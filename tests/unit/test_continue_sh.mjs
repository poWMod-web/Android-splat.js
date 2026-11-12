import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// ---- the full app resamples SH on continue (existing path, pinned) -----------
assert.match(appSource, /if \(curDeg !== opts\.sh\) convertSHDegree\(g, opts\.sh\);/,
  'a reselected degree resamples the decoded harmonics');
assert.match(appSource, /maxSplats: cap, shDeg: contShDeg \?\? recipeTrainer\.shDeg \?\? 3,/,
  'the continuation trains the chosen degree');


console.log('continue at a reselected SH degree (one UI): passed');
