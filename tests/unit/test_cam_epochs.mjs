import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { shuffledCamDeck } from '../../src/gs/trainer.js';
import { makeRng } from '../../src/sfm/geometry.js';

const trainer = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../../src/gs/shaders.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// ---- the deck is a permutation, dealt once per pass -------------------------
const N = 100;
const deck = shuffledCamDeck(N, makeRng(42));
assert.equal(deck.length, N, 'the deck should hold every camera');
assert.deepEqual([...deck].sort((a, b) => a - b), Array.from({ length: N }, (_, i) => i),
  'the first pass should cover every camera exactly once');
assert.deepEqual(shuffledCamDeck(N, makeRng(42)), deck,
  'a seeded run should reproduce the dealing order');
assert.notDeepEqual(shuffledCamDeck(N, makeRng(7)), deck,
  'a fresh shuffle should re-deal (no two passes alike)');
assert.deepEqual(shuffledCamDeck(1, makeRng(1)), [0], 'a one-camera deck still deals');
assert.deepEqual(shuffledCamDeck(0, makeRng(1)), [], 'an empty scene deals nothing');
// uniformity spot-check: over many shuffles each camera leads ~1/N of the time
const leads = new Array(10).fill(0);
for (let s = 0; s < 2000; s++) leads[shuffledCamDeck(10, makeRng(s))[0]]++;
assert.ok(leads.every((c) => c > 100 && c < 300),
  `each camera should lead a fair share of shuffles (got ${leads})`);

// ---- wiring: the step deals, the probe does not -----------------------------
assert.match(trainer, /this\._camDeck = \[\];   \/\/ shuffled-epoch deck/,
  'the trainer should open with an empty deck');
assert.match(trainer, /if \(!this\._camDeck\.length\) \{[\s\S]*shuffledCamDeck\(this\.camMeta\.length, this\.rand\)[\s\S]*\.filter\(\(ci\) => ci !== this\.holdout/,
  'a spent deck should reshuffle every trainable camera exactly once');
assert.match(trainer, /const ci = this\._camDeck\.length \? this\._camDeck\.pop\(\) : 0;/,
  'each step should deal one camera');
assert.equal((trainer.match(/\(this\.rand\(\) \* this\.camMeta\.length\) \| 0/g) || []).length, 1,
  'only the timing probe should still sample with replacement');
assert.match(trainer, /deliberately NOT the epoch deck: a timing probe must not deal training's hand/,
  'the probe should say why it samples on its own');

// ---- one graph point and per-image scores per complete shuffled deck --------
assert.match(session, /batch = Math\.min\(batch, Math\.max\(1, this\._scoreGroupLeft\)\)/,
  'a GPU batch must stop exactly at the camera-group boundary');
assert.match(session, /this\._scoreGroupLeft -= batch;[\s\S]*if \(this\._scoreGroupLeft <= 0\)[\s\S]*await this\._emitMetrics\(\)/,
  'metrics should read and plot only after the whole shuffled group');
assert.doesNotMatch(session, /now - this\._lastStats >|_statsGap/,
  'score accumulation must not be scheduled by elapsed time');
assert.match(trainer, /this\._statsWords = 4 \+ cams\.length \* 2/,
  'GPU statistics should retain error and pixel count for every image');
assert.match(shaders, /atomicAdd\(&stats\[scoreAt\], scoreErr\)[\s\S]*atomicAdd\(&stats\[scoreAt \+ 1u\], 1u\)/,
  'training pixels should feed their image score without an extra render');
assert.match(session, /if \(!final && mse != null && mse > 0\)[\s\S]*m\.frameScores = Array\.isArray\(trainer\.cameraScores\)/,
  'only complete scheduled epochs should expose graph and per-image scores');
assert.match(app, /function frameScoreColor\(value, scores\)[\s\S]*\[239, 91, 76\][\s\S]*\[242, 193, 78\][\s\S]*\[47, 212, 193\]/,
  'filmstrip scores should grade worst red through yellow to best teal');
assert.doesNotMatch(app, /c\.state === 'holdout' \? S\.psnrHold : S\.psnrTrain/,
  'filmstrip lines must not copy the global score onto every image');

console.log('shuffled-epoch camera dealing and epoch scores: passed');
