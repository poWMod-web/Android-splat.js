import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browserSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- the full app resamples SH on continue (existing path, pinned) -----------
assert.match(appSource, /if \(curDeg !== opts\.sh\) convertSHDegree\(g, opts\.sh\);/,
  'a reselected degree resamples the decoded harmonics');
assert.match(appSource, /maxSplats: cap, shDeg: contShDeg \?\? recipeTrainer\.shDeg \?\? 3,/,
  'the continuation trains the chosen degree');

// ---- the browser mirrors it --------------------------------------------------
assert.match(browserSource, /convertSHDegree \} from '\.\/app\/js\/session_io\.js';/,
  'the resampler comes from the vendored session_io');
assert.match(browserSource, /const curDeg = g\.shK != null \? DEG_OF_SHK\[g\.shK\] : recipeTrainer\.shDeg;/,
  'the current degree is read off its band count');
assert.match(browserSource, /if \(curDeg !== st\.sh\) \{\n          convertSHDegree\(g, st\.sh\);/,
  'a reselected settings-row degree resamples before the trainer boots');
assert.match(browserSource, /maxSplats: cap, shDeg: contShDeg \?\? recipeTrainer\.shDeg \?\? 3,/,
  'the continuation trains the chosen degree');
assert.match(browserSource, /if \(\(contShDeg \?\? recipeTrainer\.shDeg \?\? 3\) > 0\) recipeTrainer\.shLr = 3e-4;/,
  'active SH keeps the continued-run learning rate');
assert.match(browserSource, /ui\.sh\.value = String\(DEG_OF_SHK\[gaussians\.shK\]\);/,
  'a restore opens the settings panel on the stored degree');

console.log('continue at a reselected SH degree (app + browser): passed');
