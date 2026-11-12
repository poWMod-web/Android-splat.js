import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { scorePlateaued } from '../../src/session.js';

const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browserSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- the detector on synthetic score curves --------------------------------
const line = (n, i0, step, p0, slopePerKiter) =>
  Array(n).fill(0).map((_, k) => [i0 + k * step, p0 + (k * step * slopePerKiter) / 1000]);
const NOW = 10000, W = 1200;

assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 1.0), NOW, { window: W }), false,
  'a rising score never plateaus');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.0), NOW, { window: W }), true,
  'a flat score plateaus');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.05), NOW, { window: W }), true,
  'a crawl (+0.05 dB/kiter) plateaus');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.2), NOW, { window: W }), false,
  'a climb (+0.2 dB/kiter) does not');
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
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.09), NOW, { window: W }), true,
  'the default trips under 0.1 dB/kiter');
assert.equal(scorePlateaued(line(8, NOW - W, 150, 20, 0.11), NOW, { window: W }), false,
  'the default holds at and above 0.1 dB/kiter');

// ---- wiring pins: plateau pulls refines forward, cadence backstops ---------
assert.match(sessionSource, /export function scorePlateaued\(history, nowIter/,
  'the detector is exported for tests and reuse');
assert.match(sessionSource, /if \(this\.opts\.refinePlateau !== false && trainer\.iter > 1500 &&/,
  'plateau refines gate on the warmup iters and stay switchable');
assert.match(sessionSource, /sinceRefine >= \(this\.opts\.plateauMinGap \?\? 400\)/,
  'a minimum gap keeps a flat score from storming refines');
assert.match(sessionSource, /plateau = scorePlateaued\(this\.lossHistory, trainer\.iter, \{/,
  'the live curve feeds the detector');
assert.match(sessionSource, /if \(trainer\.iter > 1500 && \(sinceRefine >= refineEvery \|\| plateau\)\) \{/,
  'the fixed cadence stays as the maximum gap');
assert.match(sessionSource, /\[score plateaued\]/,
  'plateau-triggered refines say so in the log');
assert.match(sessionSource, /slope: this\.opts\.plateauSlope \?\? 0\.1,/,
  'the loop trips under 0.1 dB/kiter');
assert.equal((trainerSource.match(/growUntil \?\? 0\.90 \* this\.horizon/g) || []).length, 4,
  'growth runs to 90% of the horizon on every refine path');

// ---- the mobile ceiling is 1M splats ----------------------------------------
assert.match(appSource, /Math\.min\(phoneClass \? 1000000 : 2000000,/,
  'the app budgets mobiles to 1M splats');
assert.match(browserSource, /Math\.min\(phone \? 1000000 : 2000000,/,
  'the browser matches it');

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
assert.match(browserSource, /else if \(e\.kind === 'overflow' && S\._ovWarned !== \(S\.runId \?\? 'live'\)\) \{/,
  'the browser logs once per run');

console.log('plateau-driven refine + 1M mobile ceiling + overflow warnings: passed');
