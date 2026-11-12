import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { GSTrainer } from '../../src/gs/trainer.js';
import { strictMockDevice, installWebGpuShims } from './mock_gpu.mjs';
import { mockPortPair, mockGpu, mockTrainer } from './mock_train_worker.mjs';
import { RpcEndpoint } from '../../src/gs/trainer-protocol.js';
import { createTrainWorkerHandler } from '../../src/gs/trainworker.js';
import { RemoteTrainer } from '../../src/gs/remote-trainer.js';

// Plan item (2): one timestamp pair brackets each step's whole pass; the
// batch resolves + reads once and runBatch/runSteps reports gpuMs, which
// the session adapter paces on (wall dt fallback without timestamps).
installWebGpuShims();

// A real GSTrainer's timer methods over a strict mock device, minus the
// dataset/pipelines (stepOnce's pass is irrelevant to the timer math).
function timerTrainer(device) {
  const t = Object.create(GSTrainer.prototype);
  t.device = device;
  t._tsPeriod = device.limits.timestampPeriod ?? 1;
  t._tsSet = device.createQuerySet({ type: 'timestamp', count: 256 });
  t._tsResolve = device.createBuffer({ size: 256 * 8, usage: 0, label: 'ts-resolve' });
  t._tsRead = [0, 1].map((i) =>
    device.createBuffer({ size: 256 * 8, usage: 0, label: `ts-read${i}` }));
  t._tsCur = 0;
  t._tsSlot = 0;
  t._tsOpen = false;
  t._tsHavePrev = false;
  t._tsPrevSlots = 0;
  return t;
}

// one simulated step: stamp like stepOnce, consume the ring slot like stepOnce
function simStep(t) { t._tsStamp(); t._ringUsed++; t._tsSteps++; }

// ---- stamps: consecutive pairs, quiet outside an open batch ----
{
  const t = timerTrainer(strictMockDevice());
  assert.equal(t._tsStamp(), undefined, 'no stamp outside a batch (zero overhead)');
  t.beginGpuBatch();
  const w0 = t._tsStamp();
  assert.equal(w0.timestampWrites.querySet, t._tsSet);
  assert.equal(w0.timestampWrites.beginningOfPassWriteIndex, 0);
  assert.equal(w0.timestampWrites.endOfPassWriteIndex, 1);
  const w1 = t._tsStamp();
  assert.equal(w1.timestampWrites.beginningOfPassWriteIndex, 2);
  assert.equal(w1.timestampWrites.endOfPassWriteIndex, 3);
  t._ringUsed = 2; t._tsSteps = 2;
  assert.equal(await t.endGpuBatch(), null, 'first batch has no previous to read');
}

// ---- cadence: one batch of lag, ticks * period -> ms ----
{
  const t = timerTrainer(strictMockDevice());
  t.beginGpuBatch();
  simStep(t); simStep(t);
  assert.equal(await t.endGpuBatch(), null, 'first batch reports null');
  // the GPU would have written these ticks into slots 0..3
  t._tsSet._data.set([1000n, 3000n, 5000n, 9000n]);
  t.beginGpuBatch();
  simStep(t);
  const ms = await t.endGpuBatch();
  assert.ok(Math.abs(ms - 0.006) < 1e-12, `previous batch busy-ms (got ${ms})`);
  assert.ok(Math.abs(t._msStepEst - 0.003) < 1e-12, 'the merge cap learns per-step ms');
}

// ---- the full query set resolves without overrun; empty batches are null ----
{
  const t = timerTrainer(strictMockDevice());
  t.beginGpuBatch();
  let stamped = 0;
  for (let i = 0; i < 200; i++) {
    if (t._tsStamp()) stamped++;
    t._ringUsed++; // steps past the ring flush; the query-set cap only drops stamps
  }
  t._tsSteps = 200;
  assert.equal(stamped, 128, '128 pairs fill the set; further steps run unstamped');
  assert.equal(await t.endGpuBatch(), null);
  t.beginGpuBatch();
  assert.equal(await t.endGpuBatch(), null, 'an empty batch reports null (no resolve issued)');
}

// ---- batches past the 128-pair set scale the head measurement ----
{
  const t = timerTrainer(strictMockDevice());
  t.beginGpuBatch();
  for (let i = 0; i < 130; i++) { t._tsStamp(); t._ringUsed++; }
  t._tsSteps = 130;
  assert.equal(await t.endGpuBatch(), null);
  // every stamped pair took 1000 ticks; the 2 unstamped steps are identical
  for (let i = 0; i < 256; i += 2) t._tsSet._data[i + 1] = t._tsSet._data[i] + 1000n;
  t.beginGpuBatch();
  simStep(t);
  const ms = await t.endGpuBatch();
  const want = 128 * 1000 / 1e6 * (130 / 128);
  assert.ok(Math.abs(ms - want) < 1e-12, `scaled whole-batch ms (got ${ms}, want ${want})`);
}

// ---- no timestamp-query feature: every call is a quiet null ----
{
  const t = timerTrainer(strictMockDevice({ features: [] }));
  t._tsSet = null; // initTrainer leaves it null without the feature
  t.beginGpuBatch();
  assert.equal(t._tsStamp(), undefined);
  assert.equal(await t.endGpuBatch(), null);
}

// ---- the worker protocol carries gpuMs; timerless trainers report null ----
function servedPair(trainer) {
  const [mainPort, workerPort] = mockPortPair();
  const workerRpc = new RpcEndpoint(workerPort);
  const handler = createTrainWorkerHandler({
    createGpu: async () => mockGpu(),
    createTrainer: async (opts) => { trainer.opts = opts; return trainer; },
    emit: (k, p) => workerRpc.notify(k, p),
  });
  workerRpc.setHandler((m, p) => handler.handle(m, p));
  return new RemoteTrainer(mainPort);
}
const bootImages = () => [{ tw: 8, th: 8, rgb: new Float32Array(192) }, { tw: 8, th: 8, rgb: new Float32Array(192) }];
{
  const remote = servedPair(mockTrainer());
  await remote.setupRemote({
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }], images: bootImages(),
  });
  const s = await remote.runSteps(5);
  assert.equal(s.gpuMs, null, 'trainers without the timer report null gpuMs');
}
{
  const trainer = mockTrainer();
  trainer.beginGpuBatch = () => { trainer.calls.push(['beginGpuBatch']); };
  trainer.endGpuBatch = async () => 3.5;
  const remote = servedPair(trainer);
  await remote.setupRemote({
    gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
    cams: [{ imgIdx: 0 }, { imgIdx: 1 }], images: bootImages(),
  });
  const s = await remote.runSteps(5);
  assert.equal(s.gpuMs, 3.5, 'measured gpuMs crosses the worker boundary');
  assert.deepEqual(trainer.calls.filter((c) => c[0] === 'beginGpuBatch').length, 1,
    'runBatch brackets the steps with the batch timer');
}

// ---- source pins: stepOnce stamps its pass; the adapter paces on gpuMs ----
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
assert.match(trainerSource, /beginComputePass\(this\._tsStamp\(\)\)/,
  'stepOnce brackets its whole pass with the batch timer stamp');
const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
assert.match(sessionSource, /const paceMs = Number\.isFinite\(gpuMs\) && gpuMs > 0 \? gpuMs : dt;/,
  'the batch adapter paces on GPU-busy ms with wall-dt fallback');
assert.match(sessionSource, /const ideal = this\._batchIdeal\(batch, paceMs, gpuMs\);/,
  'the ideal batch derives from the paced cadence');
assert.match(sessionSource, /gpuMs == null && this\._gpuMsMisses <= 2 \? Math\.min\(batch, ideal\) : ideal;/,
  'unmeasured batches hold-or-shrink (stale fences must not spike growth)');

console.log('test_gpu_timer: ok');
