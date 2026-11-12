import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Chart, psnrToApproxSsimPercent } from '../../browser/app/js/chart.js';

const appSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- wiring pins: the live curve + the details sheet ----
assert.match(appSource,
  /import \{ Chart, psnrToApproxSsimPercent \} from '\.\/app\/js\/chart\.js';/,
  'the score curve is the vendored Chart');
assert.match(appSource, /chart = new Chart\(ui\.chartCv\);/,
  'the chart attaches to the run-section canvas');
assert.match(appSource, /chart\.push\(m\.iter, m\.psnrTrain, m\.psnrHold \?\? null\);/,
  'every scored readout lands on the curve');
assert.match(appSource, /chart\.events = S\.chartEvents\.map\(\(e\) => \(\{\s*\.\.\.e, at: e\.iter \/ S\.maxIters\s*\}\)\);/,
  'refine markers ride the curve');
assert.match(appSource, /S\.chartEvents\.push\(\{ iter: e\.iter, kind: 'refine', label: 'refine' \}\);/,
  'refines mark the curve');
assert.match(appSource, /\(S\.perfMetrics \?\?= \[\]\)\.push\(\[Math\.round\(performance\.now\(\)\), m\.iter/,
  'metrics rows accumulate for the perf log');
assert.match(appSource, /function buildPerfReport\(\) \{/, 'the perf log builder is present');
assert.match(appSource, /function pctl\(arr, p\) \{/, 'percentiles summarize frame costs');
assert.match(appSource, /function etaText\(\) \{/, 'the countdown reads 04:32-style');
assert.match(appSource, /S\.etaAt = S\.etaAt == null \? est : S\.etaAt \+ 0\.35 \* \(est - S\.etaAt\);/,
  'pace re-anchors the countdown, blended');
assert.match(appSource, /el\.addEventListener\('click', \(\) => \{ S\.detailTab = k; renderDetails\(\); \}\);/,
  'detail tabs switch + render');
assert.match(appSource, /S\.scoreScale = S\.scoreScale === 'similarity' \? 'db' : 'similarity';/,
  'the score scale toggles dB/%');
assert.match(appSource, /globalThis\.addEventListener\?\.\('resize', \(\) => chart\?\.resize\(\)\);/,
  'the chart follows layout');
assert.match(appSource, /c\.train = \(stats\.chart \|\| \[\]\)\.map/,
  'a restore replays its saved curve');

// ---- the vendored chart draws in Node behind a no-op 2d context ----
const noop2d = () => new Proxy({}, { get: () => () => {} });
const fakeCanvas = () => ({ width: 300, height: 150, addEventListener() {}, getContext: () => noop2d() });
{
  const c = new Chart(fakeCanvas());
  c.push(100, 20, 18);
  c.push(200, 25, 22);
  c.maxIter = 1000;
  c.events = [{ at: 0.1, kind: 'refine', label: 'refine' }];
  c.draw();
  assert.deepEqual(c.train, [[100, 20], [200, 25]], 'train samples kept');
  assert.deepEqual(c.hold, [[100, 18], [200, 22]], 'holdout samples kept');
  c.scale = 'similarity';
  c.draw();
  assert.equal(c._displayValue(18), 50, 'the % scale calibrates 18 dB to 50%');
  c.reset();
  assert.deepEqual([c.train, c.hold], [[], []], 'reset clears both series');
}
assert.equal(psnrToApproxSsimPercent(18), 50);
assert.equal(psnrToApproxSsimPercent(Infinity), 100);
assert.equal(psnrToApproxSsimPercent(-1), 0);

console.log('browser score curve + details sheet: passed');
