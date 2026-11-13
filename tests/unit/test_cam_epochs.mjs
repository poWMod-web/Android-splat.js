import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { shuffledCamDeck } from '../../src/gs/trainer.js';
import { makeRng } from '../../src/sfm/geometry.js';

const trainer = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');

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
assert.match(trainer, /if \(!this\._camDeck\.length\) this\._camDeck = shuffledCamDeck\(this\.camMeta\.length, this\.rand\);/,
  'a spent deck should reshuffle from the full camera set');
assert.match(trainer, /let ci = this\._camDeck\.pop\(\);/,
  'each step should deal one camera');
assert.equal((trainer.match(/\(this\.rand\(\) \* this\.camMeta\.length\) \| 0/g) || []).length, 1,
  'only the timing probe should still sample with replacement');
assert.match(trainer, /deliberately NOT the epoch deck: a timing probe must not deal training's hand/,
  'the probe should say why it samples on its own');

console.log('shuffled-epoch camera dealing: passed');
