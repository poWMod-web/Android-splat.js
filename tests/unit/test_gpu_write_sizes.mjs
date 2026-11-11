// Every queue.writeBuffer in the solve/seed path, validated against the
// spec's bounds rules on a strict mock device: any upload larger than its
// buffer (or its source bytes) throws here exactly as it does in the
// browser ("Number of bytes to write is too large"), instead of only ever
// reproducing on one user's phone. Covers fresh setup (phone + desktop
// binding limits), seed truncation past maxSplats, SH resume past the splat
// cap (local + worker), the structured-clone worker crossing, and the GPU
// matcher's uploads.
import assert from 'node:assert/strict';
import { installWebGpuShims, strictMockDevice, strictMockGpu } from './mock_gpu.mjs';

installWebGpuShims();
const { GSTrainer } = await import('../../src/gs/trainer.js');
const { initGaussians } = await import('../../src/gs/init.js');
const { createSession } = await import('../../src/session.js');
const { createTrainWorkerHandler } = await import('../../src/gs/trainworker.js');
const { gpuMatchAll } = await import('../../src/sfm/gpumatch.js');

function synthPoints(np) {
  const pts = [];
  for (let i = 0; i < np; i++) {
    pts.push({
      X: [Math.sin(i * 1.7) * 2, Math.cos(i * 2.3), (i % 17) * 0.1 - 0.8],
      rgb: [0.5 + 0.1 * Math.sin(i), 0.5, 0.5],
    });
  }
  return pts;
}

function synthCams(n, w = 64, h = 48) {
  const cams = [];
  for (let i = 0; i < n; i++) {
    cams.push({
      imgIdx: i, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [i * 0.5, 0, 0],
      f: 50, cx: w / 2, cy: h / 2,
    });
  }
  return cams;
}

function synthFrames(n, w = 64, h = 48) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    const rgb = new Float32Array(w * h * 3);
    for (let p = 0; p < w * h; p++) {
      rgb[p * 3] = 0.4 + 0.2 * ((p + i) % 5) / 5;
      rgb[p * 3 + 1] = 0.5; rgb[p * 3 + 2] = 0.6;
    }
    frames.push({ tw: w, th: h, fw: w, fh: h, rgb, alpha: null, sharpness: 100 + i, emptyFrac: 0 });
  }
  return frames;
}

const bootWrites = [];

// S1: fresh phone-shaped setup validates under a 128MB binding limit and a
// desktop 2GB one — every seed upload fits its buffer by construction.
for (const bindingLimit of [128 * 1024 * 1024, 2 * 1024 * 1024 * 1024]) {
  const gpu = strictMockGpu({ bindingLimit, onWrite: (label, size) => bootWrites.push({ bindingLimit, label, size }) });
  const trainer = await GSTrainer.create({ gpu, maxIters: 100, maxSplats: 150000, capMult: 8, shDeg: 3 });
  const model = initGaussians(synthPoints(500), 2);
  trainer.setup(model, synthCams(6), synthFrames(6), 256, 144, model.radius);
  assert.ok(trainer.n > 0 && trainer.n <= trainer.cap, 'seed within cap');
}
assert.ok(bootWrites.some((w) => w.label === 'params'), 'boot uploaded params');
assert.ok(bootWrites.some((w) => w.label.startsWith('targets-')), 'boot uploaded targets');

// S2: a seed past maxSplats truncates (the 2026 fix for the silent
// zero-model boot) — and the truncated upload still fits on a real device,
// which throws where desktop Chrome used to drop the write silently.
{
  const gpu = strictMockGpu({});
  const trainer = await GSTrainer.create({ gpu, maxIters: 100, maxSplats: 2000, capMult: 8, shDeg: 2 });
  const model = initGaussians(synthPoints(2000), 8); // ~18k seed vs a 2k cap
  assert.ok(model.n > 2000, 'test needs a seed over the cap');
  trainer.setup(model, synthCams(4), synthFrames(4), 256, 144, model.radius);
  assert.equal(trainer.n, trainer.cap, 'seed truncated to cap');
}

// S3: SH resume past the splat cap — a checkpoint with more splats than
// maxSplats wrote shN rows into a cap-sized SH buffer (session.js boot):
// the tail rows are dropped exactly like the truncated params.
{
  const gpu = strictMockGpu({});
  const session = createSession({
    maxIters: 100, holdout: -1, device: gpu.device, trainer: { maxSplats: 2000, shDeg: 2 },
  });
  session.useFrames(synthFrames(4));
  const cams = synthCams(4);
  session.useReconstruction({ cams, points: [], sceneRadius: 5 });
  const n = 4000, shK = 8;
  const data = new Float32Array(n * 16);
  for (let i = 0; i < n; i++) { data[i * 16] = i * 0.01; data[i * 16 + 13] = -2; }
  const sh = new Float32Array(n * shK * 3).fill(0.01);
  await session.seedFrom({ data, n, sh, shK, dc: 'sigmoid' }, { iter: 10 });
  assert.equal(session.trainer.n, 2000, 'resumed model truncated to cap');
  const shWrites = gpu.device.queue.writes.filter((w) => w.label === 'sh');
  assert.equal(shWrites.length, 1, 'one SH restore write');
  assert.equal(shWrites[0].size, 2000 * shK * 3 * 4, 'SH write clamped to cap rows');
}

// S4: the same resume through the worker init handler (trainworker.js boot
// writes gaussians.sh itself) — clamped, not overrunning.
{
  const gpu = strictMockGpu({});
  const handler = createTrainWorkerHandler({
    createGpu: async () => gpu,
    createTrainer: (o) => GSTrainer.create(o),
  });
  const n = 4000, shK = 8;
  const res = await handler.handle('init', {
    gaussians: {
      data: new Float32Array(n * 16).fill(0.1), n,
      sh: new Float32Array(n * shK * 3).fill(0.01), shK, dc: 'sigmoid',
    },
    cams: synthCams(4), images: synthFrames(4),
    maxViewW: 256, maxViewH: 144, radius: 5,
    trainerOpts: { maxIters: 100, maxSplats: 2000, shDeg: 2 },
    holdout: -1, excluded: [],
  });
  assert.equal(res.value.n, 2000, 'worker resume truncated to cap');
  const shWrites = gpu.device.queue.writes.filter((w) => w.label === 'sh');
  assert.equal(shWrites.length, 1, 'one worker SH restore write');
  assert.equal(shWrites[0].size, 2000 * shK * 3 * 4, 'worker SH write clamped to cap rows');
}

// S5: the GPU matcher's descriptor + job uploads validate, including an
// image with zero features (empty uploads are legal).
{
  const device = strictMockDevice({});
  const mk = (n) => {
    const desc = new Uint8Array(n * 128);
    for (let i = 0; i < desc.length; i++) desc[i] = (i * 31 + 7) & 0xff;
    return { n, desc };
  };
  const results = await gpuMatchAll([mk(50), mk(0), mk(37)], [[0, 1], [0, 2], [1, 2]],
    0.8, () => {}, device, null);
  assert.equal(results.length, 3, 'one match list per pair');
  assert.ok(results.every((m) => Array.isArray(m)), 'match lists are arrays');
}

// S6: the worker crossing itself changes nothing size-wise — a
// structured-cloned seed boot validates exactly like the local one.
{
  const gpu = strictMockGpu({});
  const handler = createTrainWorkerHandler({
    createGpu: async () => gpu,
    createTrainer: (o) => GSTrainer.create(o),
  });
  const payload = {
    gaussians: initGaussians(synthPoints(500), 2),
    cams: synthCams(6), images: synthFrames(6),
    maxViewW: 256, maxViewH: 144, radius: 3,
    trainerOpts: { maxIters: 100, maxSplats: 150000, capMult: 8, shDeg: 3 },
    holdout: -1, excluded: [],
  };
  await handler.handle('init', structuredClone(payload));
  assert.ok(gpu.device.queue.writes.length > 5, 'cloned boot performed its uploads');
}

// S7: a worker that dies after a lowMem move must NOT fall back — the
// main-side seed/targets are detached, and the fallback's first upload
// throws "Number of bytes to write is too large" (the 2026-11 solve
// failure). Without the move (clone), the fallback still works, as does a
// pre-move (factory) failure under lowMem.
{
  const { mockPortPair } = await import('./mock_train_worker.mjs');
  const { RpcEndpoint } = await import('../../src/gs/trainer-protocol.js');
  const failingWorkerFactory = () => {
    const [mainPort, workerPort] = mockPortPair();
    const workerRpc = new RpcEndpoint(workerPort);
    const handler = createTrainWorkerHandler({
      createGpu: async () => { throw new Error('worker GPU unavailable'); },
      createTrainer: async () => { throw new Error('unreached'); },
      emit: (k, p) => workerRpc.notify(k, p),
    });
    workerRpc.setHandler((m, p) => handler.handle(m, p));
    mainPort.terminate = () => {};
    return mainPort;
  };
  // local fallback boots through the real createGpu (no injected device —
  // that would skip the worker attempt entirely)
  globalThis.Worker = class MockWorker {};
  globalThis.navigator.gpu.requestAdapter = async () => ({
    limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024 },
    features: { has: () => false },
    info: { vendor: 'mock', architecture: 'mock' },
    requestDevice: async () => strictMockDevice({}),
  });
  try {
    const bootSession = (opts) => {
      const s = createSession({ maxIters: 100, holdout: -1, maskTraining: false, ...opts });
      s.useFrames(synthFrames(4));
      s.useReconstruction({ cams: synthCams(4), points: synthPoints(200), sceneRadius: 5 });
      return s;
    };
    // S7a: lowMem + post-move worker death = fatal, with the ORIGINAL error
    const sA = bootSession({ lowMem: true, createTrainWorker: failingWorkerFactory });
    const err = await sA.seed().then(() => null, (e) => e);
    assert.ok(err instanceof Error, 'lowMem worker death rejects the seed');
    assert.match(err.message, /training worker failed \(worker GPU unavailable\)/, 'original error surfaces');
    assert.equal(err.workerOwned, true, 'marked worker-owned (no fallback)');
    assert.equal(sA.trainer, null, 'no doomed local trainer booted');
    assert.equal(sA._remote, false, 'not remote');
    // S7b: the same death without the move falls back and boots locally
    const sB = bootSession({ createTrainWorker: failingWorkerFactory });
    await sB.seed();
    assert.ok(sB.trainer instanceof GSTrainer, 'clone-path worker death falls back locally');
    assert.equal(sB._remote, false, 'fallback is local');
    // S7c: a pre-move (factory) failure under lowMem still falls back
    const sC = bootSession({ lowMem: true, createTrainWorker: () => { throw new Error('no workers here'); } });
    await sC.seed();
    assert.ok(sC.trainer instanceof GSTrainer, 'pre-move failure falls back even under lowMem');
  } finally {
    delete globalThis.Worker;
    delete globalThis.navigator.gpu.requestAdapter;
  }
}

// S8: a death-heavy refine caps its relocation+grown intake at half the
// plan (every clone brings a donor op) — the plan upload always fits.
{
  const gpu = strictMockGpu({});
  const trainer = await GSTrainer.create({
    gpu, maxIters: 10000, maxSplats: 6000, capMult: 2, shDeg: 0,
    refineV2: true, growRate: 1.0, moveCap: 1.0,
  });
  const model = initGaussians(synthPoints(1500), 1); // n = 3000, cap 6000
  assert.equal(model.n, 3000, 'test needs n = 3000');
  trainer.setup(model, synthCams(4), synthFrames(4), 256, 144, model.radius);
  assert.equal(trainer.cap, 6000, 'test needs cap 6000');
  const budget = Math.floor(trainer.planCap / 2) - 1;
  // gather census: 2500 dead (opacity ~0.007), 500 alive with error mass
  gpu.device.mapFillers['refine-gatherRead'] = (ab) => {
    const g = new Float32Array(ab);
    for (let i = 0; i < 3000; i++) {
      const dead = i < 2500;
      g[i * 4] = dead ? -5 : 0;
      g[i * 4 + 1] = dead ? 0 : 1;
      g[i * 4 + 2] = dead ? 0 : 1;
      g[i * 4 + 3] = 0;
    }
  };
  const r = await trainer.refine();
  assert.equal(r.moved, budget, `death-heavy refine trimmed to the plan budget (${budget})`);
  const planWrites = gpu.device.queue.writes.filter((w) => w.label === 'refine-plan');
  assert.equal(planWrites.length, 1, 'one plan upload');
  assert.ok(planWrites[0].size <= trainer.planCap * 32, 'plan upload fits the plan buffer');
}

console.log('GPU upload sizes (setup, resume SH, matcher, worker crossing, post-move fallback, refine plan) all validate');
