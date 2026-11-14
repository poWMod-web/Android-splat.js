import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { scorePlateaued } from '../../src/session.js';
import { FastRobustZLEMA as EngineZLEMA, SKIP_FIRST } from '../../src/zlema.js';
import { FastRobustZLEMA as ChartZLEMA } from '../../app/js/chart.js';

const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const zlemaSource = readFileSync(new URL('../../src/zlema.js', import.meta.url), 'utf8');
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// ---- the detector on synthetic score curves --------------------------------
const line = (n, i0, step, p0, slopePerKiter) =>
  Array(n).fill(0).map((_, k) => [i0 + k * step, p0 + (k * step * slopePerKiter) / 1000]);
const NOW = 10000, W = 1200;

assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 1.0), NOW, { window: W }), false,
  'a rising score never plateaus');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.0), NOW, { window: W }), true,
  'a flat score plateaus');
const noisyFlat = line(9, NOW - W, 150, 20, 0.0).map((r, i) => [r[0], r[1] + (i % 2 ? 0.35 : -0.35)]);
assert.equal(scorePlateaued(noisyFlat, NOW, { window: W }), true,
  'a noisy under-capacity ceiling is still a plateau');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.05), NOW, { window: W }), true,
  'a crawl (+0.05 dB/kiter) plateaus');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.25), NOW, { window: W }), false,
  'a climb (+0.25 dB/kiter) does not');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, -0.5), NOW, { window: W }), true,
  'a falling score trips triage too');
assert.equal(scorePlateaued(line(3, NOW - W, 400, 20, 0.0), NOW, { window: W }), false,
  'three points are warming up, never a plateau');
assert.equal(scorePlateaued(line(8, NOW - 200, 25, 20, 0.0), NOW, { window: W }), false,
  'a window that is not covered is not a plateau');
assert.equal(scorePlateaued([], NOW), false);
assert.equal(scorePlateaued([[9000, 20], [9200, NaN], [9400, 20], [9600, 20], [9800, 20], [10000, 20]], NOW, { window: W }), true,
  'a NaN readout is skipped, not fatal');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.0), NOW - 5000, { window: W }), false,
  'points past the horizon do not count');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.0), NOW,
  { window: W, after: NOW - 300 }), false,
  'score evidence from before the prior growth must not trigger another growth');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.19), NOW, { window: W }), true,
  'the default trips under 0.2 dB/kiter');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.21), NOW, { window: W }), false,
  'the default holds at and above 0.2 dB/kiter');
const filteredCrawl = line(9, NOW - W, 150, 20, 0.19);
const risingNoise = filteredCrawl.map((r, i) => [r[0], r[1] + (i % 2 ? 1 : -1) * (i < 4 ? 0.02 : 0.2)]);
assert.equal(scorePlateaued(filteredCrawl, NOW, { window: W, rawHistory: risingNoise }), true,
  'increasing raw variance accompanies a filtered under-capacity plateau below 0.2 dB/kiter');
const healthyFiltered = line(9, NOW - W, 150, 20, 1.0);
const healthyNoisy = healthyFiltered.map((r, i) => [r[0], r[1] + (i % 2 ? 1 : -1) * (i < 4 ? 0.02 : 0.2)]);
assert.equal(scorePlateaued(healthyFiltered, NOW, { window: W, rawHistory: healthyNoisy }), false,
  'increasing noise cannot interrupt a strongly improving score');

// ---- wiring pins: growth follows recovery and robust plateaus --------------
assert.match(sessionSource, /export function scorePlateaued\(history, nowIter/,
  'the detector is exported for tests and reuse');
assert.doesNotMatch(sessionSource, /stdDev <= maxStdDev/,
  'noise is diagnostic, not a hard veto on under-capacity plateaus');
assert.match(sessionSource, /sinceRefine >= convergenceGap/,
  'a full convergence gap must elapse after the prior growth');
assert.match(sessionSource, /const recovered = this\._refineRecoveryTarget == null[\s\S]*after: trainer\.lastRefine \|\| 0/,
  'the next plateau needs recovery to its reference and fresh evidence');
assert.match(sessionSource, /if \(trainer\.iter >= plateauWindow && plateau\) \{/,
  'automatic growth should run only after a full measured score window');
assert.match(sessionSource, /rawHistory: this\.lossHistory/,
  'the live controller evaluates rising variance in raw score samples');
assert.match(sessionSource, /lateNoise >= Math\.max\(earlyNoise \* noiseRatio, earlyNoise \+ 0\.03\)/,
  'raw noise must materially increase, not merely remain noisy');
assert.doesNotMatch(sessionSource, /sinceRefine >= refineEvery \|\| plateau|_plateauGap \* 2/,
  'neither fixed cadence nor exponential delays should bypass repeated convergence');
assert.match(sessionSource, /\[score plateaued\]/,
  'plateau-triggered refines say so in the log');
assert.match(sessionSource, /slope: this\.opts\.plateauSlope \?\? 0\.2,/,
  'the loop trips under 0.2 dB/kiter');
assert.equal((trainerSource.match(/growUntil \?\? 0\.90 \* this\.horizon/g) || []).length, 4,
  'growth runs to 90% of the horizon on every refine path');

// ---- the mobile ceiling is 1M splats ----------------------------------------
assert.match(appSource, /Math\.min\(phoneClass \? 1000000 : 2000000,/,
  'the app budgets mobiles to 1M splats');

// ---- growth already plants splats where the error is ------------------------
assert.match(trainerSource, /error-weighted donor draw \(only when the window carries any mass\)/,
  'donors draw by error mass');
assert.match(trainerSource, /if \(cdf\[mid\] < r\) lo = mid \+ 1; else hi = mid;/,
  'the draw is an inverse-CDF sample over donors');

// ---- overflow past the entries binding warns, never fails silent ------------
assert.match(sessionSource, /const ov = trainer\.entryOverflowTiles \|\| 0;/,
  'every metrics readout checks the cumulative drop count');
assert.match(sessionSource, /this\._em\.emit\('event', \{ kind: 'overflow', iter: trainer\.iter, tiles: ov \}\);/,
  'each increase emits once');
assert.match(appSource, /if \(e\.kind === 'overflow' && S\._ovWarned !== \(S\.runId \?\? 'live'\)\) \{/,
  'the app flashes once per run');

// ---- the trigger reads the graph's history, not the raw readouts -------------
assert.equal(ChartZLEMA, EngineZLEMA,
  'the graph and the trigger should share one filter class');
assert.equal(SKIP_FIRST, 3,
  'the trigger should hold back the same cold-start samples as the graph');
assert.match(sessionSource, /import \{ FastRobustZLEMA, SKIP_FIRST \} from '\.\/zlema\.js';/,
  'the session should import the shared filter');
assert.match(sessionSource, /this\.zlemaHistory = \[\];   \/\/ \[iter, filteredPsnrTrain\]/,
  'the constructor should open an empty filtered history');
assert.match(sessionSource, /if \(this\.lossHistory\.length > SKIP_FIRST\) \{/,
  'every raw readout past the hold should also feed the shared filter');
assert.match(sessionSource, /this\.zlemaHistory\.push\(\[trainer\.iter, this\._zlema\.update\(m\.psnrTrain\)\]\);/,
  'the trigger history should store filtered values per iteration');
assert.match(sessionSource, /setLossHistory\(rows\) \{[\s\S]*this\._zlema\.series\(this\.lossHistory\.slice\(SKIP_FIRST\)\)/,
  'a restored history should replay the graph pass through a fresh filter');
assert.match(appSource, /session\.setLossHistory\(history\);/,
  'reopening a run should hydrate the filtered history with it');
// the engine's incremental feed is the graph's fresh pass, sample for sample
const spiky = Array(8).fill(0).map((_, k) => [NOW - W + k * 150, 20 + (k * 150 * 0.2) / 1000]);
spiky[6][1] -= 2.5; spiky[7][1] -= 2.0;
const incrFeed = new EngineZLEMA();
const triggerHist = [];
spiky.forEach(([i, v], k) => { if (k >= SKIP_FIRST) triggerHist.push([i, incrFeed.update(v)]); });
assert.deepEqual(triggerHist, new EngineZLEMA().series(spiky.slice(SKIP_FIRST)),
  'incremental per-push updates should equal the graph fresh pass');
assert.equal(scorePlateaued(spiky, NOW, { window: W }), true,
  'raw transient noise may resemble an under-capacity plateau');
assert.equal(scorePlateaued(triggerHist, NOW, { window: W }), false,
  'the filtered slope should see through them to the healthy climb');

console.log('plateau-driven refine + 1M mobile ceiling + overflow warnings: passed');
