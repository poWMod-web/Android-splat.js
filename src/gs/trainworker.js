// trainworker.js — the GPU-owning training core in a dedicated worker.
//
// Owns the WebGPU device, the GSTrainer, batch execution with fence gating,
// densification, evals, model download and viewport rendering into a
// transferred OffscreenCanvas. The main thread keeps the Session shell
// (cadence, batch/duty policy, metrics, UI state) and drives this worker
// through trainer-protocol.js. A host-owned device (opts.device) can never
// cross threads — the session forces local training then (see remote-trainer's
// canOffload check in the session wiring).
//
// NOTE: no DOM here (OffscreenCanvas only). `self` wiring runs solely under
// WorkerGlobalScope so Node can import the handler factory for tests.

import { GSTrainer } from './trainer.js';
import { createGpu, plainAdapterInfo } from '../gpu/context.js';
import { makeSplatTest } from './hull.js';
import { RpcEndpoint } from './trainer-protocol.js';

/** Live sets the main thread may push mid-run (anything else is rejected). */
const CONFIG_KEYS = new Set([
  'holdout', 'excluded', 'growLimit', 'protect', 'freezePos',
  'maxIters', 'iter', 'adamT0', 'hull',
]);

export function createTrainWorkerHandler(deps = {}) {
  const createGpuImpl = deps.createGpu || createGpu;
  const createTrainerImpl = deps.createTrainer || ((opts) => GSTrainer.create(opts));
  const makeSplatTestImpl = deps.makeSplatTest || makeSplatTest;
  const emit = deps.emit || (() => {});
  let gpu = null;
  let trainer = null;
  let fences = [];
  let fenceDepth = 4;
  // keyed canvases: the host swaps one stage canvas and scratch canvases
  // (snapshots, error renders); each element transfers once, then re-attaches
  // by key with its current size (a transferred element's main-side resize
  // never reaches the offscreen — the size rides the attach).
  const view = {
    canvases: new Map(), current: -1, camera: null, offset: 0,
    enabled: true, dirty: false, renderInFlight: null,
    pending: false,      // a present is owed to the rAF loop below
    rafRunning: false,   // the loop parks when idle (no perpetual wakeups)
  };

  /** Submit the pending view render now. */
  const submitViewRender = () => {
    const tr = need();
    const rec = view.canvases.get(view.current);
    if (!view.enabled || !rec || !view.camera) return false;
    // latest-wins: a render already draining its fence marks dirty and
    // returns instead of queueing stale frames
    if (view.renderInFlight) { view.dirty = true; return false; }
    view.dirty = false;
    tr.renderView(view.camera, rec.ctx, 0, view.offset);
    const t = tr.device.queue.onSubmittedWorkDone();
    view.renderInFlight = t;
    t.then(
      () => { if (view.renderInFlight === t) view.renderInFlight = null; },
      () => { if (view.renderInFlight === t) view.renderInFlight = null; });
    return true;
  };

  /** The present loop: a placeholder's frames commit in the worker's own
   *  update-the-rendering steps, so the submit lives inside a worker rAF
   *  tick — a message-driven submit presents on desktop but is NOT the
   *  path the spec commits (2026-11: silent black on Android WebView,
   *  renders accepted, nothing on screen, no error anywhere). Parks when
   *  idle; force (and hosts without worker rAF) submits synchronously. */
  const ensureViewLoop = () => {
    if (view.rafRunning) return;
    view.rafRunning = true;
    const tick = () => {
      if (!view.pending && !view.dirty) { view.rafRunning = false; return; }
      try {
        view.pending = false;
        submitViewRender();
      } catch (e) {
        // a transient submit failure (resize race, lost texture) must not
        // kill the loop: rafRunning would stick true and every later render
        // would be accepted-but-never-submitted — silent black (2026-11).
        // The owed work survives (pending back on) and the loop PARKS: the
        // next renderNow restarts it, so retries are bounded by real demand
        // instead of spinning. Device loss still surfaces via gpu.onLost.
        view.pending = true;
        view.rafRunning = false;
        emit('log', `view submit failed (${(e && e.message) || e}) — the next view frame retries`);
        return;
      }
      globalThis.requestAnimationFrame(tick);
    };
    globalThis.requestAnimationFrame(tick);
  };

  const need = () => {
    if (!trainer) throw new Error('trainworker: init() first');
    return trainer;
  };

  /** Plain-data mirror of the scalars the session shell reads. */
  const mirror = () => {
    const tr = need();
    return {
      camMeta: tr.camMeta, n: tr.n, cap: tr.cap, iter: tr.iter,
      tileGrad: tr.tileGrad, shK: tr.shK, dcMode: tr.dcMode,
      dilate: tr.dilate, mipComp: tr.mipComp, horizon: tr.horizon,
      holdout: tr.holdout, lastCam: tr.lastCam,
      gpuInfo: plainAdapterInfo(gpu && gpu.info),
    };
  };

  const applyConfig = (patch) => {
    const tr = need();
    for (const k of Object.keys(patch)) {
      if (!CONFIG_KEYS.has(k)) throw new Error(`trainworker: config key '${k}' is not offloadable`);
    }
    if (patch.holdout !== undefined) tr.holdout = patch.holdout;
    if (patch.excluded !== undefined) tr.excluded = new Set(patch.excluded);
    if (patch.growLimit !== undefined) tr.growLimit = patch.growLimit;
    if (patch.protect !== undefined) tr.protect = patch.protect && { ...patch.protect };
    if (patch.freezePos !== undefined) tr.setFreezePos(patch.freezePos.from, patch.freezePos.to);
    if (patch.maxIters !== undefined) {
      tr.opts.maxIters = patch.maxIters;
      tr.horizon = patch.maxIters;
    }
    if (patch.iter !== undefined) tr.iter = patch.iter;
    if (patch.adamT0 !== undefined) tr.adamT0 = patch.adamT0;
    if (patch.hull !== undefined) {
      tr.hullKill = patch.hull ? makeSplatTestImpl(patch.hull.hull, patch.hull.hullOpts || {}) : null;
    }
    return true;
  };

  const viewCmd = (cmd, p = {}) => {
    const tr = need();
    switch (cmd) {
      case 'attach': {
        const key = p.key ?? 0;
        let rec = view.canvases.get(key);
        if (p.canvas) {
          // first attach of this element: the OffscreenCanvas arrives transferred
          const ctx = p.canvas.getContext('webgpu');
          if (!ctx) throw new Error('trainworker: OffscreenCanvas WebGPU context unavailable in worker');
          ctx.configure({ device: tr.device, format: tr.canvasFormat, alphaMode: 'opaque' });
          rec = { canvas: p.canvas, ctx };
          view.canvases.set(key, rec);
        }
        if (!rec) throw new Error(`trainworker: unknown canvas key ${key}`);
        if (p.w > 0) rec.canvas.width = p.w;
        if (p.h > 0) rec.canvas.height = p.h;
        view.current = key;
        view.dirty = true;
        return true;
      }
      case 'setEnabled':
        view.enabled = !!p.enabled;
        view.dirty = true;
        return true;
      case 'setCamera':
        view.camera = p.cam;
        view.offset = 0;
        view.dirty = true;
        return true;
      case 'lookThrough': {
        const meta = tr.camMeta[p.i];
        if (!meta) throw new Error(`trainworker: no camera ${p.i}`);
        view.camera = { ...meta };
        view.offset = meta.offset;
        return { ...meta };
      }
      case 'renderNow': {
        const rec = view.canvases.get(view.current);
        if (!view.enabled || !rec || !view.camera) return false;
        // the present loop commits inside worker rAF ticks (see above);
        // force — and hosts without worker rAF — submit synchronously
        if (p.force || typeof globalThis.requestAnimationFrame !== 'function') {
          return submitViewRender();
        }
        view.pending = true;
        ensureViewLoop();
        return true;
      }
      case 'inspect': {
        // the no-present probe's other half (see SessionView): the worker's
        // own view state, so a black canvas diagnoses as desync (no canvas
        // / no camera / disabled / stuck in flight) vs failed present
        const rec = view.canvases.get(view.current);
        return {
          current: view.current, canvases: view.canvases.size,
          enabled: view.enabled, dirty: view.dirty,
          pending: view.pending, rafRunning: view.rafRunning,
          inFlight: !!view.renderInFlight,
          camera: view.camera ? { w: view.camera.w, h: view.camera.h } : null,
          canvasWH: rec ? { w: rec.canvas.width | 0, h: rec.canvas.height | 0 } : null,
          offset: view.offset,
        };
      }
      default:
        throw new Error(`trainworker: unknown view command '${cmd}'`);
    }
  };

  const handle = async (method, params = {}) => {
    switch (method) {
      case 'init': {
        if (trainer) throw new Error('trainworker: already initialized (dispose first)');
        const { gaussians, cams, images, maxViewW, maxViewH, radius } = params;
        const trainerOpts = params.trainerOpts || {};
        gpu = await createGpuImpl({ powerPreference: trainerOpts.powerPreference });
        gpu.onLost = (info) => {
          emit('device-lost', {
            reason: (info && info.reason) || 'unknown',
            message: (info && info.message) || '',
            iter: trainer ? trainer.iter : 0,
          });
        };
        fenceDepth = trainerOpts.fenceRing ?? 4;
        trainer = await createTrainerImpl({ ...trainerOpts, gpu });
        trainer.setup(gaussians, cams, images, maxViewW, maxViewH, radius);
        // resume-with-SH: setup zero-fills SH, a restored model brings its own
        // (clamped to the cap — a checkpoint past it truncated the params
        // but pasted all n rows into a cap-sized SH buffer)
        if (gaussians.sh && trainer.bufSH) {
          const shRows = Math.min(gaussians.n, trainer.cap);
          if (shRows < gaussians.n) {
            emit('log', `SH restore truncated to ${shRows} of ${gaussians.n} rows (splat cap)`);
          }
          trainer.device.queue.writeBuffer(trainer.bufSH, 0,
            gaussians.sh.buffer, gaussians.sh.byteOffset,
            shRows * trainer.shK * 3 * 4);
        }
        if (params.hull) {
          trainer.hullKill = makeSplatTestImpl(params.hull, params.hullOpts || {});
        }
        if (params.protect) trainer.protect = { ...params.protect };
        if (params.freezePos) trainer.setFreezePos(params.freezePos.from, params.freezePos.to);
        trainer.holdout = params.holdout ?? -1;
        trainer.excluded = new Set(params.excluded || []);
        if (params.iter != null) { trainer.iter = params.iter; trainer.adamT0 = params.iter; }
        return { value: mirror() };
      }
      case 'runBatch': {
        const tr = need();
        const steps = Math.max(1, params.steps | 0);
        // (2) the batch GPU timer brackets the steps (guarded: mock/host
        // trainers without it report null and the adapter uses wall dt)
        if (typeof tr.beginGpuBatch === 'function') tr.beginGpuBatch();
        for (let k = 0; k < steps; k++) tr.stepOnce();
        const gpuMs = typeof tr.endGpuBatch === 'function' ? await tr.endGpuBatch() : null;
        // the fence ring lives here (fences can't cross threads): pushing
        // every batch and gating past ring depth gives the main thread its
        // backpressure through this call's round trip. fenceMs is how long
        // the gated fence took (null when nothing gated): the session's
        // instant-fence detector paces timerless loops off it.
        fences.push(tr.device.queue.onSubmittedWorkDone());
        let fenceMs = null;
        if (fences.length > fenceDepth) {
          const f0 = performance.now();
          await fences.shift();
          fenceMs = performance.now() - f0;
        }
        return { value: { iter: tr.iter, n: tr.n, lastCam: tr.lastCam, gpuMs, fenceMs } };
      }
      case 'refine': {
        const tr = need();
        const r = await tr.refine();
        return { value: { ...r, iter: tr.iter, n: tr.n } };
      }
      case 'softPrune': {
        const tr = need();
        const r = await tr.softPrune(params.ratio ?? null, { skipRatchet: !!params.skipRatchet });
        return { value: { ...r, iter: tr.iter, n: tr.n } };
      }
      case 'pruneDead': {
        const tr = need();
        const r = await tr.pruneDead();
        return { value: { ...r, iter: tr.iter, n: tr.n } };
      }
      case 'readLoss': {
        const tr = need();
        return { value: { mse: await tr.readLoss(), entryOverflowTiles: tr.entryOverflowTiles || 0 } };
      }
      case 'eval': {
        const tr = need();
        const { kind, ci } = params;
        if (kind === 'psnr') return { value: await tr.evalCamPsnr(ci) };
        if (kind === 'metrics') return { value: await tr.evalCamMetrics(ci) };
        if (kind === 'refined') return { value: await tr.evalCamPsnrRefined(ci, params.steps) };
        throw new Error(`trainworker: unknown eval kind '${kind}'`);
      }
      case 'testPsnr': {
        const tr = need();
        const frames = [];
        let sum = 0;
        for (const c of params.cams) {
          const psnr = await tr.evalCamPsnr(c);
          frames.push({ cam: c, imgIdx: tr.camMeta[c].imgIdx, psnr });
          sum += psnr;
        }
        return { value: { psnr: frames.length ? sum / frames.length : null, frames } };
      }
      case 'readGaussians': {
        const tr = need();
        const g = await tr.readGaussians();
        const transfer = [];
        if (g.data) transfer.push(g.data.buffer);
        if (g.sh) transfer.push(g.sh.buffer);
        return { value: g, transfer };
      }
      case 'camDrift': {
        const tr = need();
        return {
          value: {
            report: tr.camDrift ? tr.camDrift() : null,
            poses: tr.camMeta.map((m) => ({ R: Array.from(m.R), t: Array.from(m.t) })),
          },
        };
      }
      case 'config':
        return { value: applyConfig(params.patch || {}) };
      case 'view':
        return { value: viewCmd(params.cmd, params) };
      case 'capture': {
        const tr = need();
        const shot = await tr.renderCapture(params.cam);
        return { value: shot, transfer: [shot.rgba.buffer] };
      }
      case 'sync': {
        const tr = need();
        await tr.device.queue.onSubmittedWorkDone();
        return { value: true };
      }
      case 'dispose': {
        fences = [];
        view.canvases.clear(); view.current = -1; view.camera = null;
        view.renderInFlight = null; view.dirty = false;
        if (gpu) { try { gpu.dispose(); } catch { /* already gone */ } }
        gpu = null;
        trainer = null;
        return { value: true };
      }
      default:
        throw new Error(`trainworker: unknown method '${method}'`);
    }
  };

  return { handle, _debug: () => ({ trainer, gpu }) };
}

// Real worker wiring. Guarded on WorkerGlobalScope (not `self`, which also
// exists in Node) so unit tests can import the factory without side effects.
if (typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined') {
  const rpc = new RpcEndpoint(self);
  const handler = createTrainWorkerHandler({ emit: (k, p) => rpc.notify(k, p) });
  rpc.setHandler((m, p) => handler.handle(m, p));
}
