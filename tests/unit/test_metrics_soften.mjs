import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createSession } from '../../src/session.js';

const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');

// ---- 500ms model-touch backoff (was 2s) ----
assert.match(appSource,
  /const armResume = \(\) => \{\n    clearTimeout\(resumeT\);\n    resumeT = setTimeout\(\(\) => \{\n      resumeT = 0;\n      if \(!held\.size && S\._dragPaused\) \{\n        S\._dragPaused = false;\n        if \(S\.state === 'train'\) S\.session\.start\(\);\n      \}\n    \}, 500\);/,
  'the model-touch hold should resume training 500ms after the last touch');

// ---- score plotting rides a coalesced rAF, not the metrics event ----
assert.match(appSource, /let metricsRaf = 0;\nfunction queueMetricsPaint\(\) \{\n  if \(metricsRaf\) return;/,
  'metrics painting should coalesce through a single-slot rAF gate');
assert.match(appSource, /metricsRaf = requestAnimationFrame\(\(\) => \{/,
  'the deferred paint should run on the next animation frame');
assert.equal((appSource.match(/queueMetricsPaint\(\);/g) || []).length, 2,
  'onMetrics should queue (not perform) both the chart paint and the details repaint');
const onMetricsBody = appSource.slice(
  appSource.indexOf('function onMetrics(m) {'), appSource.indexOf('function pctOf(iter) {'));
assert.doesNotMatch(onMetricsBody, /chart\.draw\(\)/,
  'onMetrics must not paint the chart synchronously (the metrics event arrives mid-frame)');
assert.doesNotMatch(onMetricsBody, /dchart\.draw\(\)/,
  'onMetrics must not rebuild the details chart synchronously');
assert.match(appSource, /function queueMetricsPaint\(\) \{[\s\S]*?if \(chart && chart\.train\.length\) chart\.draw\(\);/,
  'the deferred paint should keep the first-samples gate (no paint before the ramp clears)');

// ---- readback softening: breathe before readLoss, defer a stacked holdout ----
assert.match(sessionSource, /await tick\(\);\n    const mse = await trainer\.readLoss\(\);/,
  'the metrics readout should yield before readLoss so a pending present goes out first');
assert.match(sessionSource, /const drainDefer = !final && \(performance\.now\(\) - tRead\) > 1500;/,
  'a >1.5s queue drain should defer the holdout eval to the next readout');
assert.match(sessionSource, /!isHidden\(\) && !drainDefer && trainer\.iter - this\._lastHoldEval >= holdEvery/,
  'the holdout cadence gate should honor the drain deferral');

// ---- functional: the deferral fires, retries, and never starves final ----
const evalCalls = [];
const s = createSession({ evalHoldEvery: 4000 });
s.trainer = {
  iter: 5000, n: 1000,
  readLoss: async () => 0.001,
  evalCamPsnr: async (ci) => { evalCalls.push(ci); return 28.5; },
};
s.holdout = 1;
s._lastHoldEval = 0;
s._lastHold = null;
s._lastStats = performance.now() - 3000;
s._itersAtStats = 4000;
{
  const m = await s._emitMetricsInner(false);
  assert.deepEqual(evalCalls, [1], 'a cheap readout should still run the due holdout eval');
  assert.equal(m.psnrHold, 28.5);
  assert.equal(s._lastHoldEval, 5000, 'the cadence should advance on an actual eval');
  assert.ok(Math.abs(m.psnrTrain - 30) < 1e-9, 'train PSNR still comes from the read loss');
}
// a 2s queue drain, simulated on a fake clock (no real waiting)
const realNow = performance.now;
let fakeNow = 50000;
performance.now = () => fakeNow;
try {
  s.trainer.iter = 9000;
  s.trainer.readLoss = async () => { fakeNow += 2000; return 0.001; };
  s._lastStats = fakeNow - 3000;
  s._itersAtStats = 8000;
  const m = await s._emitMetricsInner(false);
  assert.deepEqual(evalCalls, [1], 'a drained readout should defer the holdout instead of stacking hitches');
  assert.equal(s._lastHoldEval, 5000, 'the cadence must not advance on a deferred eval (it retries next readout)');
  assert.equal(m.psnrHold, 28.5, 'the deferred readout should report the last known holdout score');
  assert.equal(s._statsGap, 5000, 'drain pacing still prices the readout that actually happened');
  const fin = await s._emitMetricsInner(true);
  assert.deepEqual(evalCalls, [1, 1], 'final always evaluates — the done screen needs its number');
  assert.equal(s._lastHoldEval, 9000);
  assert.equal(fin.psnrHold, 28.5);
} finally {
  performance.now = realNow;
}

console.log('ok — metrics soften: 500ms backoff, rAF paint, drain deferral');
