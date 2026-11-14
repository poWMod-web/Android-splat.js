import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const render = app.slice(app.indexOf('function renderControls()'), app.indexOf('/** The local-run viewer'));
const toggle = app.slice(app.indexOf('function toggleTrainingVisuals()'), app.indexOf('// the OAuth popup'));
const continuation = app.slice(app.indexOf('async function continueLocalRun'), app.indexOf('/** Resume from done'));

assert.match(render, /S\.atFrame >= 0 && \(S\.state !== 'train' \|\| S\.trainVisuals\)/,
  'photo comparison controls must disappear with training visuals');
assert.match(toggle, /syncTrainingVisualsControl\(\);\s*renderControls\(\);/,
  'hiding visuals must repaint controls immediately');
assert.match(continuation, /viewOnly \? 'Opening saved model …' : 'Preparing to continue training …'/,
  'view-only checkpoint loading must not claim training is resuming');

console.log('training visual controls + view-only messaging: passed');
