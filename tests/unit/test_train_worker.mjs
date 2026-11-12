import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RpcEndpoint, collectBuffers, TRAIN_METHODS } from '../../src/gs/trainer-protocol.js';
import { createTrainWorkerHandler } from '../../src/gs/trainworker.js';
import { RemoteTrainer } from '../../src/gs/remote-trainer.js';

const workerSource = readFileSync(new URL('../../src/gs/trainworker.js', import.meta.url), 'utf8');
const remoteSource = readFileSync(new URL('../../src/gs/remote-trainer.js', import.meta.url), 'utf8');
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');

import { mockPortPair, mockGpu, mockTrainer } from './mock_train_worker.mjs';
import { plainAdapterInfo } from '../../src/gpu/context.js';

// ---- collectBuffers ----
{
  const a = new Float32Array([1, 2]);
  const b = new Uint8Array([3]);
  const nested = { x: [a, { y: a }], z: b, n: 5, s: 't', u: undefined };
  const out = collectBuffers(nested);
  assert.deepEqual(out, [a.buffer, b.buffer], 'nested buffers collected once each (dedupe)');
  assert.deepEqual(collectBuffers({ v: new DataView(a.buffer) }), [a.buffer]);
  assert.deepEqual(collectBuffers([1, 's', null]), []);
  class C { constructor() { this.b = new Uint8Array([1]); } }
  assert.deepEqual(collectBuffers(new C()), [], 'class instances are opaque');
  if (typeof SharedArrayBuffer !== 'undefined') {
    const sab = new Uint8Array(new SharedArrayBuffer(8));
    assert.deepEqual(collectBuffers({ v: sab }), [], 'shared buffers must never transfer');
  }
}

// ---- RpcEndpoint ----
{
  const [pa, pb] = mockPortPair();
  const ea = new RpcEndpoint(pa), eb = new RpcEndpoint(pb);
  eb.setHandler(async (m, p) => {
    if (m === 'add') return { value: p.a + p.b };
    if (m === 'boom') throw new Error('kaput');
    if (m === 'xfer') { const v = new Uint8Array([9]); return { value: v, transfer: [v.buffer] }; }
    throw new Error(`unknown ${m}`);
  });
  const [r1, r2] = await Promise.all([ea.call('add', { a: 1, b: 2 }), ea.call('add', { a: 5, b: 6 })]);
  assert.equal(r1, 3);
  assert.equal(r2, 11, 'concurrent calls resolve to their own ids');
  await assert.rejects(ea.call('boom', {}), /kaput/, 'handler throws propagate as rejections');
  await assert.rejects(ea.call('nope', {}), /unknown nope/);
  const xv = await ea.call('xfer', {});
  assert.ok(xv instanceof Uint8Array && xv[0] === 9);
  assert.equal(pb.sent.at(-1).xfer.length, 1, 'handler transfer lists ride postMessage');
  // unknown/late responses are ignored, never crash
  pa.onmessage({ data: { t: 'res', id: 9999, ok: true, v: 1 } });
  assert.equal(ea.pending, 0);
  // events
  let evt = null;
  ea.onEvent((k, p) => { evt = [k, p]; });
  eb.notify('ping', { n: 1 });
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(evt, ['ping', { n: 1 }]);
  // failAll
  const pending = ea.call('add', { a: 1, b: 1 });
  ea.failAll(new Error('torn down'));
  await assert.rejects(pending, /torn down/);
  assert.equal(ea.pending, 0);
}

// ---- full RemoteTrainer <-> handler integration ----
const [mainPort, workerPort] = mockPortPair();
const gpu = mockGpu();
const trainer = mockTrainer();
let splatArgs = null;
const events = [];
const workerRpc = new RpcEndpoint(workerPort);
const handler = createTrainWorkerHandler({
  createGpu: async () => gpu,
  createTrainer: async (opts) => { trainer.opts = opts; return trainer; },
  makeSplatTest: (hull, o) => { splatArgs = [hull, o]; return 'hull-fn'; },
  // production routes emits over the wire (rpc.notify); record + route here too
  emit: (k, p) => { events.push([k, p]); workerRpc.notify(k, p); },
});
workerRpc.setHandler((m, p) => handler.handle(m, p));
const remote = new RemoteTrainer(mainPort, { maxIters: 100, seed: 7 });

// init mirrors state and applies the seed-time sets
{
  const mirror = await remote.setupRemote({
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }],
    images: [{ tw: 8, th: 8, rgb: new Float32Array(192) }, { tw: 8, th: 8, rgb: new Float32Array(192) }],
    maxViewW: 64, maxViewH: 64, radius: 5,
    trainerOpts: { maxIters: 100, seed: 7 },
    hull: { lo: [0, 0, 0], cell: 1, dim: [2, 2, 2], alive: new Uint8Array(8) },
    hullOpts: { sigma: 3 },
    protect: { from: 1, to: 3, until: 50 },
    freezePos: { from: 0, to: 2 },
    holdout: 1, excluded: [0], iter: 10,
  });
  assert.deepEqual(trainer.calls[0], ['setup', 10, 2, 2], 'seed + 2 cams + 2 images reach setup');
  assert.deepEqual(splatArgs[1], { sigma: 3 }, 'the hull test is rebuilt worker-side from data');
  assert.equal(trainer.hullKill, 'hull-fn');
  assert.deepEqual(trainer.protect, { from: 1, to: 3, until: 50 });
  assert.deepEqual(trainer.calls.find((c) => c[0] === 'freezePos'), ['freezePos', 0, 2]);
  assert.equal(trainer.holdout, 1);
  assert.ok(trainer.excluded instanceof Set && trainer.excluded.has(0), 'excluded crosses as array, stored as Set');
  assert.equal(trainer.iter, 10);
  assert.equal(trainer.adamT0, 10);
  assert.equal(mirror.n, 10);
  assert.equal(mirror.cap, 40);
  assert.equal(mirror.camMeta.length, 2);
  assert.equal(mirror.gpuInfo.vendor, 'mock');
  assert.equal(remote.n, 10, 'RemoteTrainer mirrors scalars for session reads');
  assert.equal(remote.camMeta.length, 2);
  assert.ok(remote.excluded.has(0));
  await assert.rejects(remote.setupRemote({}), /already initialized/, 'double init is rejected');
}

// runBatch steps the mock and refreshes the mirror
{
  await remote.runSteps(5);
  assert.equal(trainer.iter, 15);
  assert.equal(remote.iter, 15);
  assert.equal(remote.n, 10);
  assert.equal(remote.lastCam, 15 % 2);
  assert.equal(trainer.device.queue.submits, 1, 'one fence per batch (ring gate lives worker-side)');
}

// runBatch reports gated-fence ms for the session's instant-fence detector
{
  const r2 = await remote.runSteps(1);
  const r3 = await remote.runSteps(1);
  const r4 = await remote.runSteps(1);
  assert.equal(r2.fenceMs, null, 'no gate inside ring depth');
  assert.equal(r3.fenceMs, null);
  assert.equal(r4.fenceMs, null);
  const r5 = await remote.runSteps(1);
  assert.equal(typeof r5.fenceMs, 'number', 'past depth the gated fence is timed');
  assert.ok(r5.fenceMs >= 0 && r5.fenceMs < 5000, `fenceMs ${r5.fenceMs}`);
}

// refine / readLoss / evals
{
  const r = await remote.refine();
  assert.equal(r.moved, 1);
  assert.equal(r.iter, 19);   // +4 batches from the gated-fence block above
  assert.equal(await remote.readLoss(), 0.001);
  assert.equal(await remote.evalCamPsnr(1), 21);
  assert.deepEqual(await remote.evalCamMetrics(0), { psnr: 20, ssim: 0.9 });
  assert.equal(await remote.evalCamPsnrRefined(0, 200), 21.2);
  await assert.rejects(remote._rpc.call('eval', { kind: 'bogus', ci: 0 }), /unknown eval kind/);
  const t = await remote.evalTestPsnr([0, 1]);
  assert.equal(t.psnr, 20.5);
  assert.deepEqual(t.frames.map((f) => f.imgIdx), [0, 1]);
}

// readGaussians transfers (zero-copy download)
{
  const before = workerPort.sent.length;
  const g = await remote.readGaussians();
  assert.equal(g.n, 10);
  assert.ok(g.data instanceof Float32Array);
  const xfer = workerPort.sent.slice(before).flatMap((s) => s.xfer);
  assert.ok(xfer.includes(g.data.buffer) || xfer.length === 1,
    'the model buffer rides the transfer list');
}

// camDrift returns the report plus plain-data poses
{
  const d = await remote.camDrift();
  assert.deepEqual(d.report, { n: 2 });
  assert.equal(d.poses.length, 2);
  assert.deepEqual(d.poses[1].t, [1, 0, 0]);
}

// config allow-list (+ mirror sync, + rejection of the unknown)
{
  await remote.setConfig({ holdout: 0, excluded: [1], maxIters: 200, iter: 20 });
  assert.equal(trainer.holdout, 0);
  assert.ok(trainer.excluded.has(1) && !trainer.excluded.has(0));
  assert.equal(trainer.opts.maxIters, 200);
  assert.equal(trainer.horizon, 200);
  assert.equal(remote.holdout, 0);
  assert.ok(remote.excluded.has(1));
  assert.equal(remote.opts.maxIters, 200);
  assert.equal(remote.iter, 20);
  await remote.setConfig({ hull: null });
  assert.equal(trainer.hullKill, null);
  await assert.rejects(remote.setConfig({ hullKill: 'fn' }), /not offloadable/,
    'functions can never cross — the hull travels as data');
  await assert.rejects(remote._rpc.call('nope', {}), /unknown method/);
}

// view commands (transferred canvas: uncloned delivery in the mock)
{
  const [mp, wp] = mockPortPair({ clone: false });
  new RpcEndpoint(wp).setHandler((m, p) => handler.handle(m, p));
  const rv = new RemoteTrainer(mp);
  let configured = null;
  const canvas = { width: 0, height: 0, getContext: () => ({ configure: (c) => { configured = c; } }) };
  await rv.viewCmd('attach', { key: 0, canvas, w: 16, h: 9 }, [canvas]);
  assert.equal(configured.device, trainer.device, 'context configures against the worker device');
  assert.equal(configured.format, 'bgra8unorm');
  assert.equal(configured.alphaMode, 'opaque');
  assert.equal(mp.sent[0].xfer[0], canvas, 'the canvas is transferred, not cloned');
  assert.equal(canvas.width, 16, 'the attach size reaches the offscreen (main-side resizes do not)');
  assert.equal(canvas.height, 9);
  // re-attaching the same element travels by key — the canvas moves only once
  canvas.width = 0;
  await rv.viewCmd('attach', { key: 0, w: 32, h: 18 });
  assert.equal(canvas.width, 32, 're-attach by key still resizes');
  await assert.rejects(rv.viewCmd('attach', { key: 7, w: 4, h: 4 }), /unknown canvas key/,
    'an untransferred key is a loud error, not a silent no-op');
  await assert.rejects(rv.viewCmd('attach', { key: 9, canvas: { getContext: () => null }, w: 4, h: 4 }),
    /WebGPU context unavailable/,
    'a worker that cannot make canvas contexts fails loudly (snapshot fallback covers it)');
  await rv.viewCmd('setEnabled', { enabled: true });
  const meta = await rv.viewCmd('lookThrough', { i: 1 });
  assert.equal(meta.imgIdx, 1);
  const renders = trainer.calls.filter((c) => c[0] === 'renderView').length;
  // hold the first render's fence open so the second lands while in flight
  const realFence = trainer.device.queue.onSubmittedWorkDone;
  let release = null;
  trainer.device.queue.onSubmittedWorkDone =
    () => new Promise((r) => { release = r; });
  assert.equal(await rv.viewCmd('renderNow', {}), true);
  assert.equal(await rv.viewCmd('renderNow', {}), false, 'latest-wins: an in-flight render marks dirty');
  assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, renders + 1);
  release();
  trainer.device.queue.onSubmittedWorkDone = realFence;
  await rv.viewCmd('setCamera', { cam: { w: 4, h: 4 } });
  assert.equal(await rv.sync(), true);
  await assert.rejects(rv.viewCmd('dolly', {}), /unknown view command/);
  // inspect: the no-present probe's worker half (state, not pixels)
  const insp = await rv.viewCmd('inspect', {});
  assert.equal(insp.current, 0);
  assert.equal(insp.canvases, 1);
  assert.equal(insp.enabled, true);
  assert.equal(insp.inFlight, false);
  assert.deepEqual(insp.camera, { w: 4, h: 4 });
  assert.deepEqual(insp.canvasWH, { w: 32, h: 18 });
  assert.equal(insp.offset, 0);

  // the present loop: with worker rAF available, renderNow marks pending
  // and the submit runs inside a worker rAF tick (the path placeholder
  // commits take); the loop parks when idle and force bypasses it
  const realRaf = globalThis.requestAnimationFrame;
  const queued = [];
  globalThis.requestAnimationFrame = (fn) => { queued.push(fn); return queued.length; };
  try {
    const renders = trainer.calls.filter((c) => c[0] === 'renderView').length;
    assert.equal(await rv.viewCmd('renderNow', {}), true);
    assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, renders,
      'accepted, not submitted — the tick submits');
    assert.equal(queued.length, 1, 'the first pending render starts the loop');
    assert.equal(await rv.viewCmd('renderNow', {}), true);
    assert.equal(queued.length, 1, 'coalesced: no second loop');
    assert.equal((await rv.viewCmd('inspect', {})).pending, true);
    queued.shift()();   // the rAF tick
    assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, renders + 1,
      'one submit for the coalesced pair (latest-wins)');
    assert.equal(queued.length, 1, 'the tick re-arms while draining');
    queued.shift()();   // nothing owed: parks
    assert.equal(queued.length, 0, 'idle parks the loop (no perpetual wakeups)');
    assert.equal((await rv.viewCmd('inspect', {})).rafRunning, false);
    assert.equal(await rv.viewCmd('renderNow', {}), true);
    assert.equal(queued.length, 1, 'a later render restarts the parked loop');
    queued.shift()(); queued.shift()();   // drain + park
    const forced = trainer.calls.filter((c) => c[0] === 'renderView').length;
    assert.equal(await rv.viewCmd('renderNow', { force: true }), true);
    assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, forced + 1,
      'force submits synchronously, never through the loop');
    assert.equal(queued.length, 0, 'force wakes nothing');
  } finally {
    if (realRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = realRaf;
  }

  // a throwing submit parks the loop with the work still owed (2026-11:
  // the throw used to kill the loop with rafRunning stuck true — every
  // later render accepted-but-never-submitted, silent black)
  globalThis.requestAnimationFrame = (fn) => { queued.push(fn); return queued.length; };
  try {
    const renders = trainer.calls.filter((c) => c[0] === 'renderView').length;
    const logs = events.length;
    const realRender = trainer.renderView;
    trainer.renderView = () => { throw new Error('transient texture loss'); };
    try {
      assert.equal(await rv.viewCmd('renderNow', {}), true);
      assert.equal(queued.length, 1);
      queued.shift()();   // the tick throws mid-submit...
      assert.equal(queued.length, 0, '...and parks instead of dying mid-loop');
      const insp = await rv.viewCmd('inspect', {});
      assert.equal(insp.rafRunning, false);
      assert.equal(insp.pending, true, 'the owed work survives the failed tick');
      assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, renders,
        'nothing submitted on the failed tick');
      assert.ok(events.length > logs && events[events.length - 1][0] === 'log',
        'the failed submit is logged, not silent');
    } finally {
      trainer.renderView = realRender;
    }
    assert.equal(await rv.viewCmd('renderNow', {}), true);
    assert.equal(queued.length, 1, 'the next render restarts the parked loop');
    queued.shift()();
    assert.equal(trainer.calls.filter((c) => c[0] === 'renderView').length, renders + 1,
      'the retry submits');
    queued.shift()();   // park
    assert.equal(queued.length, 0);
  } finally {
    if (realRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = realRaf;
  }
}

// capture round-trips pixels on the transfer list
{
  const before = workerPort.sent.length;
  const shot = await remote.capture({ w: 4, h: 2 });
  assert.equal(shot.w, 4);
  assert.equal(shot.h, 2);
  assert.ok(shot.rgba instanceof Uint8ClampedArray && shot.rgba.length === 32);
  const xfer = workerPort.sent.slice(before).flatMap((s) => s.xfer);
  assert.equal(xfer.length, 1, 'captured pixels ride transferred');
}

// error propagation from trainer methods
{
  trainer.refine = async () => { throw new Error('refine exploded'); };
  await assert.rejects(remote.refine(), /refine exploded/);
}

// device loss fails pending calls and emits (as _locked fails fast locally)
{
  assert.ok(typeof gpu.onLost === 'function', 'init arms the lost handler');
  const lost = [];
  remote.onEvent('device-lost', (p) => lost.push(p));
  const hanging = remote._rpc.call('sync', {});
  gpu.onLost({ reason: 'unit-test', message: 'reclaimed' });
  await assert.rejects(hanging, /device lost/);
  assert.equal(lost.length, 1);
  assert.equal(lost[0].reason, 'unit-test');
  const evt = events.find((e) => e[0] === 'device-lost');
  assert.ok(evt, 'the worker emits device-lost for the session shell');
}

// worker-level errors fail pending calls too
{
  const hanging = remote._rpc.call('sync', {});
  mainPort.onerror({ message: 'worker crashed' });
  await assert.rejects(hanging, /worker crashed/);
}

await remote.dispose();
assert.ok(gpu.disposed, 'dispose releases the worker device');

// calls before init are rejected, not silent
{
  const [mp, wp] = mockPortPair();
  new RpcEndpoint(wp).setHandler((m, p) => createTrainWorkerHandler({
    createGpu: async () => mockGpu(), createTrainer: async () => mockTrainer(),
  }).handle(m, p));
  const r2 = new RemoteTrainer(mp);
  await assert.rejects(r2.runSteps(3), /init\(\) first/);
}

// ---- gpuInfo crosses as a plain snapshot (raw GPUAdapterInfo is uncloneable) ----
{
  assert.deepEqual(plainAdapterInfo(null),
    { vendor: '', architecture: '', device: '', description: '' });
  assert.deepEqual(plainAdapterInfo({ vendor: 'ARM', architecture: 'Mali', extra: 1 }),
    { vendor: 'ARM', architecture: 'Mali', device: '', description: '' },
    'the four spec fields pass, extras are dropped');
  // shaped like the platform object: spec strings behind prototype getters
  // (a spread would lose them) plus an uncloneable own member, so the mock
  // port's structuredClone throws on it exactly like the phone's postMessage
  class RawAdapterInfo {
    get vendor() { return 'ARM'; }
    get architecture() { return 'Mali-G710'; }
    constructor() { this.opaque = () => {}; }
  }
  assert.throws(() => structuredClone(new RawAdapterInfo()), /could not be cloned/,
    'the stand-in really is uncloneable');
  const [mp, wp] = mockPortPair();
  const g2 = mockGpu();
  g2.info = new RawAdapterInfo();
  const t2 = mockTrainer();
  const wrpc = new RpcEndpoint(wp);
  const h2 = createTrainWorkerHandler({
    createGpu: async () => g2,
    createTrainer: async (opts) => { t2.opts = opts; return t2; },
    emit: (k, p) => wrpc.notify(k, p),
  });
  wrpc.setHandler((m, p) => h2.handle(m, p));
  const r2 = new RemoteTrainer(mp);
  const mirror = await r2.setupRemote({
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }],
    images: [{ tw: 8, th: 8, rgb: new Float32Array(192) }, { tw: 8, th: 8, rgb: new Float32Array(192) }],
  });
  assert.deepEqual(mirror.gpuInfo,
    { vendor: 'ARM', architecture: 'Mali-G710', device: '', description: '' },
    'the getters are read (not spread away) and the opaque member is dropped');
}

// ---- source pins ----
for (const m of TRAIN_METHODS) {
  assert.match(workerSource, new RegExp(`case '${m}':`),
    `trainworker must serve protocol method '${m}'`);
}
assert.match(workerSource, /typeof WorkerGlobalScope !== 'undefined'/,
  'worker self-wiring must be guarded so Node can import the handler');
assert.match(workerSource, /GSTrainer\.create/,
  'the worker must reuse GSTrainer, not fork the training logic');
assert.match(workerSource, /fences\.length > fenceDepth/,
  'the fence ring lives worker-side (fences cannot cross threads)');
assert.match(remoteSource, /failAll\(new Error\(`GPU device lost/,
  'remote device-loss must fail pending calls fast, like _locked');
assert.match(remoteSource, /collectBuffers\(rest\)/,
  'ownership transfer must be opt-in per setup (lowMem drops CPU copies)');
assert.match(workerSource, /case 'inspect':/,
  'the worker must report its own view state for the no-present probe');
assert.match(workerSource, /gpuInfo: plainAdapterInfo\(gpu && gpu\.info\),/,
  'the mirror must snapshot adapter info (raw GPUAdapterInfo cannot clone)');

assert.match(trainerSource, /async renderCapture\(camParams\) \{/,
  'GSTrainer must offer a canvas-free offscreen capture');
assert.match(trainerSource, /this\.encodeRaster\(p, camParams, false\)[\s\S]{0,400}enc\.copyBufferToBuffer\(this\.bufOut, 0, rb, 0, bytes\)/,
  'capture must raster the view path and read bufOut back directly');
assert.match(trainerSource, /Math\.min\(1, Math\.max\(0, f\[i\]\)\) \* 255/,
  'capture must quantize exactly like the blit clamp');
assert.match(trainerSource, /rb\.unmap\(\); rb\.destroy\(\);\n    return \{ w, h, rgba \};/,
  'capture must release its staging buffer');

console.log('worker training protocol (rpc, handler, proxy) verified');
