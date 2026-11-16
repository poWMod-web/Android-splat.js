import assert from 'node:assert/strict';
import { createSession } from '../../src/session.js';
import { RemoteTrainer } from '../../src/gs/remote-trainer.js';
import { mockTrainer, mockWorkerFactory } from './mock_train_worker.mjs';

// A failed prune reseed must leave the previous core live and (when the run
// was training) resume onto it — never the 2026-11 cascade: a half-booted
// trainer that failed readbacks as GPU TypeErrors and steps as .length.

// Node has no WebGPU: guarantee the failure path even if a future Node
// grows navigator.gpu (restore afterwards; each file runs in its own process).
const hadGpu = globalThis.navigator && globalThis.navigator.gpu;
try { if (globalThis.navigator) delete globalThis.navigator.gpu; } catch { /* non-configurable: the test still fails closed below */ }

function makeRng(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// valid stride-16 seed rows (quats normalized, sane scales): simplify eats these
function validModel(n, seed = 7) {
  const rng = makeRng(seed);
  const data = new Float32Array(n * 16);
  for (let i = 0; i < n; i++) {
    const b = i * 16;
    data[b] = (rng() - 0.5) * 4; data[b + 1] = (rng() - 0.5) * 4; data[b + 2] = (rng() - 0.5) * 4;
    data[b + 3] = Math.log(0.02); data[b + 4] = Math.log(0.02); data[b + 5] = Math.log(0.02);
    data[b + 6] = 1; data[b + 7] = 0; data[b + 8] = 0; data[b + 9] = 0;
    data[b + 10] = 0; data[b + 11] = 0; data[b + 12] = 0;
    data[b + 13] = Math.log(0.7 / 0.3);
  }
  return { data, n, sh: null, shK: 0, dc: 'sigmoid' };
}

const frame = (imgIdx) => ({
  tw: 8, th: 8, fw: 8, fh: 8, name: `f${imgIdx}`,
  rgb: new Float32Array(8 * 8 * 3).fill(0.5),
  sharpness: 1, emptyFrac: 0, sampleColor: () => [0.5, 0.5, 0.5],
});

// a local session mid-run: healthy mock core, solved cameras, pixels present
function riggedLocal({ training }) {
  const s = createSession({ trainWorker: false });
  const t = mockTrainer();
  const model = validModel(64);
  t.readGaussians = async () => ({ ...model, data: model.data.slice() });
  s.trainer = t;
  s.holdout = -1;
  s.recon = { cams: [{ imgIdx: 0, f: 100 }, { imgIdx: 1, f: 100 }], sceneRadius: 10 };
  s.frames = [frame(0), frame(1)];
  s.training = training;
  s._capAsked = true;   // the ceiling asked; a failure must re-arm it
  return s;
}

// ---- local prune, reseed fails: previous core intact, loop resumes ----
{
  const s = riggedLocal({ training: true });
  const before = s.trainer;
  await assert.rejects(() => s.pruneAndContinue(0.5),
    'a reseed with no WebGPU device must reject');
  assert.equal(s.trainer, before, 'the failed reseed publishes nothing over the live trainer');
  assert.equal(s.training, true, 'a training run resumes onto the unpruned model');
  assert.equal(s._capAsked, false, 'the failure re-arms the ceiling prompt');
  s.pause();
  await new Promise((r) => setTimeout(r, 120));   // land the scheduled frame, clear its timers
  s.dispose();
}

// ---- local prune, reseed fails while paused: stays paused, core intact ----
{
  const s = riggedLocal({ training: false });
  const before = s.trainer;
  await assert.rejects(() => s.pruneAndContinue(0.5));
  assert.equal(s.trainer, before, 'the failed reseed publishes nothing over the live trainer');
  assert.equal(s.training, false, 'a paused run (the cap-hit card) stays paused');
  assert.equal(s._capAsked, false, 'the failure re-arms the ceiling prompt');
  await new Promise((r) => setTimeout(r, 60));
  s.dispose();
}

// ---- remote reseed, replacement fails: previous worker untouched ----
{
  const hosts = [];
  const s = createSession({
    trainWorker: true,
    createTrainWorker: () => mockWorkerFactory(hosts[hosts.push({}) - 1])(),
  });
  const boot = {
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }],
    frames: [frame(0), frame(1)],
    maxW: 64, maxH: 64, radius: 5,
  };
  await s._bootRemoteTrainer({ maxIters: 100 }, boot);
  const prevTrainer = s.trainer;
  const prevWorker = s._worker;
  const prevGen = s._workerGen;
  assert.ok(prevTrainer instanceof RemoteTrainer && s._remote);

  // the replacement's worker blows up in setup (OOM, bad upload, ...)
  hosts.push('sabotage-next');
  const factory = s.opts.createTrainWorker;
  s.opts.createTrainWorker = () => mockWorkerFactory(hosts[hosts.push({}) - 1], {
    wrapTrainer: (t) => { t.setup = () => { throw new Error('worker setup blew up'); }; },
  })();
  await assert.rejects(() => s._bootRemoteTrainer({ maxIters: 100 }, boot), /worker setup blew up/);
  s.opts.createTrainWorker = factory;
  assert.equal(s.trainer, prevTrainer, 'the failed replacement publishes nothing');
  assert.equal(s._worker, prevWorker, 'the previous worker stays wired');
  assert.equal(s._remote, true, 'the session still believes its live remote core');
  assert.equal(s._workerGen, prevGen, 'no generation bump without a boot');
  assert.equal(hosts[0].terminated, false, 'the previous worker is not terminated');
  assert.equal(hosts[hosts.length - 1].terminated, true, 'only the failed replacement is terminated');
  s.dispose();
}

// ---- remote reseed success retires the previous worker (no two-device leak) ----
{
  const hosts = [];
  const s = createSession({
    trainWorker: true,
    createTrainWorker: () => mockWorkerFactory(hosts[hosts.push({}) - 1])(),
  });
  const boot = {
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }],
    frames: [frame(0), frame(1)],
    maxW: 64, maxH: 64, radius: 5,
  };
  await s._bootRemoteTrainer({ maxIters: 100 }, boot);
  const first = s.trainer;
  await s._bootRemoteTrainer({ maxIters: 100 }, boot);
  assert.notEqual(s.trainer, first, 'the replacement publishes');
  assert.equal(hosts[0].terminated, true, 'the replaced worker is terminated (no orphaned device)');
  assert.equal(hosts[1].terminated, false, 'the live worker survives');
  s.dispose();
}

if (hadGpu) globalThis.navigator.gpu = hadGpu;

console.log('prune-failure recovery (intact core + resume, local + remote): passed');
