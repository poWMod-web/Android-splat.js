import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { scorePlateaued } from '../../src/session.js';

const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// ---- the detector on synthetic score curves --------------------------------
const line = (n, i0, step, p0, slopePerKiter) =>
  Array(n).fill(0).map((_, k) => [i0 + k * step, p0 + (k * step * slopePerKiter) / 1000]);
const NOW = 10000, W = 1200;

const eligible = (extra = {}) => ({ window: W, ...extra });

const rising = line(9, NOW - W, 150, 20, 1.0);
assert.equal(scorePlateaued(rising, NOW, eligible()), false,
  'a healthy raw epoch-score climb must not trigger growth');
const flat = line(9, NOW - W, 150, 20, 0.0);
assert.equal(scorePlateaued(flat, NOW, eligible()), true,
  'a raw epoch-score plateau alone triggers growth');
const crawl = line(9, NOW - W, 150, 20, 0.05);
assert.equal(scorePlateaued(crawl, NOW, eligible()), true,
  'a crawl below the slope threshold triggers growth');
const climb = line(9, NOW - W, 150, 20, 0.25);
assert.equal(scorePlateaued(climb, NOW, eligible()), false,
  'a climb at or above the slope threshold does not trigger growth');
const falling = line(9, NOW - W, 150, 20, -0.5);
assert.equal(scorePlateaued(falling, NOW, eligible()), true,
  'a falling raw epoch score triggers triage');
const tooFew = line(3, NOW - W, 400, 20, 0.0);
assert.equal(scorePlateaued(tooFew, NOW, eligible()), false,
  'three points are warming up, never a plateau');
const short = line(8, NOW - 200, 25, 20, 0.0);
assert.equal(scorePlateaued(short, NOW, eligible()), false,
  'a window that is not covered is not a plateau');
assert.equal(scorePlateaued([], NOW), false);
const withNan = [[9000, 20], [9200, NaN], [9400, 20], [9600, 20], [9800, 20], [10000, 20]];
assert.equal(scorePlateaued(withNan, NOW, eligible()), true,
  'a NaN epoch readout is skipped, not fatal');
assert.equal(scorePlateaued(flat, NOW - 5000, eligible()), false,
  'points past the horizon do not count');
assert.equal(scorePlateaued(flat, NOW, eligible({ after: NOW - 300 })), false,
  'score evidence from before the prior growth must not trigger another growth');
const postGrowth = line(8, NOW - 700, 100, 20, 0.0);
assert.equal(scorePlateaued(postGrowth, NOW, {
  window: W, minSpan: 700, after: NOW - 700,
}), true, 'fresh plateau evidence may trigger after the 700-cycle backoff');
const under = line(9, NOW - W, 150, 20, 0.19);
assert.equal(scorePlateaued(under, NOW, eligible()), true,
  'the default trips under 0.2 dB/kiter');
const over = line(9, NOW - W, 150, 20, 0.21);
assert.equal(scorePlateaued(over, NOW, eligible()), false,
  'the default holds at and above 0.2 dB/kiter');

// ---- wiring pins: growth follows recovery and robust plateaus --------------
assert.match(sessionSource, /export function scorePlateaued\(history, nowIter/,
  'the detector is exported for tests and reuse');
assert.match(sessionSource, /return robustSlope < slope;/,
  'growth should use only the robust raw epoch-score plateau slope');
assert.match(sessionSource, /const convergenceGap = this\.opts\.plateauMinGap \?\? 700;/,
  'the post-growth backoff defaults to at least 700 iterations');
assert.match(sessionSource, /sinceRefine >= convergenceGap/,
  'the convergence gap must elapse after the prior growth');
assert.match(sessionSource, /minSpan: \(trainer\.lastRefine \|\| 0\) > 0 \? convergenceGap : plateauWindow \* 0\.75/,
  'post-growth fresh evidence uses the same 700-cycle minimum');
assert.match(sessionSource, /const recovered = this\._refineRecoveryTarget == null[\s\S]*after: trainer\.lastRefine \|\| 0/,
  'the next plateau needs recovery to its reference and fresh evidence');
assert.match(sessionSource, /if \(trainer\.iter >= plateauWindow && plateau\) \{/,
  'automatic growth should run only after a full measured score window');
assert.doesNotMatch(sessionSource, /rawHistory: this\.lossHistory|noiseRatio|noiseFloor|noiseRising/,
  'raw score variance must not participate in automatic growth');
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

// ---- the trigger reads the exact raw series plotted by the graph -----------
assert.doesNotMatch(sessionSource, /FastRobustZLEMA|SKIP_FIRST|zlemaHistory|_zlema/,
  'the session should perform no ZLEMA calculation');
assert.match(sessionSource, /currentScore = this\.lossHistory\.length/,
  'growth recovery should use the latest raw complete-epoch score');
assert.match(sessionSource, /scorePlateaued\(this\.lossHistory, trainer\.iter/,
  'the plateau detector should read the same raw series as the graph');
assert.match(sessionSource, /setLossHistory\(rows\) \{[\s\S]*this\.lossHistory = rows\.map/,
  'restored history should stay raw');
assert.match(appSource, /session\.setLossHistory\(history\);/,
  'reopening a run should hydrate that raw history');
const spiky = Array(8).fill(0).map((_, k) => [NOW - W + k * 150, 20 + (k * 150 * 0.2) / 1000]);
spiky[6][1] -= 2.5; spiky[7][1] -= 2.0;
assert.equal(scorePlateaued(spiky, NOW, { window: W }), true,
  'the robust raw slope should judge the supplied complete-epoch samples directly');

console.log('plateau-driven refine + 1M mobile ceiling + overflow warnings: passed');
