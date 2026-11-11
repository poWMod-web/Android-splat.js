// Shared mocks for the worker-training tests (not a test itself: run.mjs
// only executes test_*.mjs). Mock ports deliver async like a real Worker;
// the mock GPU/trainer implement just the surface trainworker.js touches.

// `clone` simulates the structured-clone crossing (and fails loudly on
// uncloneable payloads, as a real port would); pass clone:false for
// transferred values, which travel by reference in the mock.
export function mockPortPair({ clone = true } = {}) {
  // transferred values cross by reference (moved, not cloned); everything
  // else clones — so uncloneable non-transferred payloads still fail loudly
  const cross = (msg, xfer) => {
    if (!clone) return msg;
    const refs = new Map((xfer || []).map((v, i) => [v, i]));
    if (!refs.size) return structuredClone(msg);
    const out = [], stack = [];
    const swap = (v) => {
      if (refs.has(v)) { out.push(v); return { $mockXfer: refs.get(v) }; }
      if (!v || typeof v !== 'object') return v;
      // views, buffers and class instances cross via the native clone
      if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer) return v;
      const proto = Object.getPrototypeOf(v);
      if (proto !== Object.prototype && proto !== null) return v;
      if (stack.includes(v)) return v;
      stack.push(v);
      const c = Array.isArray(v) ? v.map(swap) : Object.fromEntries(Object.keys(v).map((k) => [k, swap(v[k])]));
      stack.pop();
      return c;
    };
    const unswap = (v) => {
      if (!v || typeof v !== 'object') return v;
      if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer) return v;
      const proto = Object.getPrototypeOf(v);
      if (proto !== Object.prototype && proto !== null && !Array.isArray(v)) return v;
      if (!Array.isArray(v) && typeof v.$mockXfer === 'number') return out[v.$mockXfer];
      if (Array.isArray(v)) return v.map(unswap);
      return Object.fromEntries(Object.keys(v).map((k) => [k, unswap(v[k])]));
    };
    return unswap(structuredClone(swap(msg)));
  };
  const deliver = (from, to) => (msg, xfer) => {
    from.sent.push({ msg, xfer: xfer || [] });
    const data = cross(msg, xfer);
    queueMicrotask(() => { if (to.onmessage) to.onmessage({ data }); });
  };
  const a = { sent: [] }, b = { sent: [] };
  a.postMessage = deliver(a, b);
  b.postMessage = deliver(b, a);
  return [a, b];
}

export function mockGpu() {
  return { info: { vendor: 'mock', architecture: 'mock-gpu' }, onLost: null, disposed: false,
    dispose() { this.disposed = true; } };
}

export function mockTrainer() {
  const device = {
    queue: {
      submits: 0, writes: [],
      writeBuffer(buf, offset, data, dataOffset, size) {
        this.writes.push({ buf, offset, size: size ?? data.byteLength });
      },
      async onSubmittedWorkDone() { this.submits++; },
    },
  };
  return {
    device, canvasFormat: 'bgra8unorm', opts: {}, calls: [],
    n: 10, cap: 40, iter: 0, tileGrad: true, shK: 0, dcMode: 'sigmoid',
    dilate: 0.1, mipComp: true, horizon: 100, holdout: -1, lastCam: 0,
    camMeta: [
      { imgIdx: 0, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], w: 8, h: 8, offset: 0 },
      { imgIdx: 1, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [1, 0, 0], w: 8, h: 8, offset: 64 },
    ],
    bufSH: null, hullKill: null, protect: null, excluded: new Set(),
    entryOverflowTiles: 0,
    setup(g, c, images) { this.calls.push(['setup', g.n, c.length, images.length]); },
    stepOnce() { this.iter++; this.lastCam = this.iter % this.camMeta.length; },
    async refine() { this.calls.push(['refine']); return { moved: 1, grown: 2 }; },
    async readLoss() { return 0.001; },
    async evalCamPsnr(ci) { return 20 + ci; },
    async evalCamMetrics(ci) { return { psnr: 20 + ci, ssim: 0.9 }; },
    async evalCamPsnrRefined(ci, steps) { return 21 + (steps || 0) / 1000; },
    async readGaussians() {
      return { data: new Float32Array([1, 2, 3]), n: 10, sh: null, shK: 0, dc: 'sigmoid' };
    },
    camDrift() { return { n: 2 }; },
    setFreezePos(from, to) { this.calls.push(['freezePos', from, to]); },
    renderView() { this.calls.push(['renderView']); },
    async renderCapture(cam) {
      this.calls.push(['renderCapture', cam.w, cam.h]);
      return { w: cam.w, h: cam.h, rgba: new Uint8ClampedArray(cam.w * cam.h * 4).fill(7) };
    },
  };
}

import { RpcEndpoint } from '../../src/gs/trainer-protocol.js';
import { createTrainWorkerHandler } from '../../src/gs/trainworker.js';

/** A mock Worker factory for session tests: the returned main-side port is
 *  already served by a REAL handler around the mock GPU/trainer, like the
 *  production worker. `host` receives { gpu, trainer, ports, terminated }. */
export function mockWorkerFactory(host = {}, { wrapTrainer = null } = {}) {
  return () => {
    const [mainPort, workerPort] = mockPortPair();
    const gpu = mockGpu();
    const trainer = mockTrainer();
    if (wrapTrainer) wrapTrainer(trainer);
    host.gpu = gpu; host.trainer = trainer;
    host.ports = { mainPort, workerPort };
    host.terminated = false;
    const workerRpc = new RpcEndpoint(workerPort);
    const handler = createTrainWorkerHandler({
      createGpu: async () => gpu,
      createTrainer: async (opts) => { trainer.opts = opts; return trainer; },
      emit: (k, p) => workerRpc.notify(k, p),
    });
    workerRpc.setHandler((m, p) => handler.handle(m, p));
    mainPort.terminate = () => { host.terminated = true; };
    return mainPort; // the session treats the main port as the worker
  };
}
