import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// Continuing a finished run reselects the training size first (like a new
// training); a paused run resumes its original configuration directly.
assert.match(app, /if \(r && r\.status === 'finished'\) \{ continueConfigCard\(r\); return; \}/,
  'keep-training on a finished run should open the configuration card');

const cardStart = app.indexOf('function continueConfigCard(');
const cardEnd = app.indexOf('/** Continue training a stored local run:', cardStart);
assert.ok(cardStart >= 0 && cardEnd > cardStart, 'the continue-config card should exist');
const card = app.slice(cardStart, cardEnd);
assert.match(card, /id="ltc-res"/, 'the card should offer a training resolution select');
assert.match(card, /id="ltc-splats"/, 'the card should offer a splat budget select');
assert.match(card, /id="ltc-iters"/, 'the card should offer an extra-cycles select');
assert.match(card, /Same as before/,
  'each knob should default to the finished run’s own configuration');
assert.match(card, /continueLocalRun\(false, \{\s*res:[\s\S]*splats:[\s\S]*extraIters,/,
  'confirming the card should continue with the chosen configuration');

const contStart = app.indexOf('async function continueLocalRun(viewOnly = false, opts = {})');
assert.ok(contStart >= 0, 'the continuation should accept configuration overrides');
const contEnd = app.indexOf('function continueTraining(', contStart);
const cont = app.slice(contStart, contEnd);
assert.match(cont, /opts\.extraIters[\s\S]*defaultExtra/,
  'the continuation horizon should honor configured extra cycles, defaulting to the automatic extension');
assert.match(cont, /opts\.res === 'auto' \? undefined[\s\S]*opts\.res === 'full' \? nativeDim[\s\S]*: prevDim/,
  'the continuation should honor a reselected resolution, defaulting to the run’s own');
assert.match(cont, /Math\.max\(g\.n, lr\.splats \|\| 0,[\s\S]*opts\.splats \|\| 0\)/,
  'the continuation budget should honor a reselected splat count, never below the trained size');
assert.match(cont, /frames: \{ trainMaxDim \}/,
  'the rebuilt session should train at the selected resolution');

console.log('continue-training configuration reselection: passed');
