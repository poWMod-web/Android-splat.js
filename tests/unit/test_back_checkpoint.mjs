import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');

// Pausing is free: toggleTrain must not checkpoint (the write moved to leaving).
const toggle = app.slice(app.indexOf('function toggleTrain()'), app.indexOf('function toggleTrain()') + 1400);
assert.ok(!toggle.includes('checkpointRun'), 'pause must not write a checkpoint');
assert.ok(toggle.includes('headlessStatusWrite'), 'pause still follows the handoff status');
assert.ok(toggle.includes('splatjs_pause_hint'), 'pause teaches the new model once');

// The checkpoint reason is the leave, and it keeps the saved-on-device promise.
assert.ok(!app.includes("checkpointRun('pause')"), "no 'pause' checkpoints remain");
assert.ok(app.includes("if (reason === 'back') flash('Progress saved on this device"),
  'the leave checkpoint keeps the saved-on-device promise');

// Both in-page exits secure a live run, then boot the deliberate-leave home.
assert.ok(app.includes("await checkpointRun('back')"), 'the header Back awaits the checkpoint');
assert.ok(app.includes("location.href = (S.state === 'train' || S.state === 'prep') ? 'index.html?back=1' : 'index.html';"),
  'leaving a live run boots the deliberate-leave home');
assert.ok(app.includes("checkpointRun('back').catch(() => {})\n      .then(() => { location.href = 'index.html?back=1'; });"),
  'popstate secures the IDB checkpoint before leaving too');
// Back is IDB-only: the Yours resume is the way back, no file copy.
const back = app.slice(app.indexOf('async function goBack()'), app.indexOf('function backToSetup()'));
assert.ok(!back.includes('secureFileCheckpoint'), 'goBack secures IDB only, never files');
const pop = app.slice(app.indexOf("addEventListener('popstate'"), app.indexOf('function detailClose()'));
assert.ok(!pop.includes('secureFileCheckpoint'), 'system Back secures IDB only, never files');
assert.ok(app.includes("history.pushState({ sj: 'run' }, '')"),
  'training owns a history entry so Back pops into the checkpoint-leave');
assert.ok(app.includes("else if (mp.get('back')) headlessClearMarker().finally(() => mountWall());"),
  'the deliberate-leave boot stands the handoff down instead of recovering');
assert.ok(app.includes('async function headlessClearMarker()'), 'the marker stand-down exists');

// The brand title IS Back off the home tile: one shared routine, same effect.
// (On the home tile it still opens About.)
assert.ok(app.includes('async function goBack()'), 'Back lives in one shared routine');
assert.match(app, /\$\('btn-new'\)\.addEventListener\('click', async \(e\) => \{\s*e\.stopPropagation\(\);\s*await goBack\(\);/,
  'the header Back runs the shared routine');
assert.match(app, /\$\('brand'\)\.addEventListener\('click', async \(\) => \{[\s\S]*?else await goBack\(\);/,
  'the brand title runs the same routine off the home tile');

// Android system Back routes through the page history (same checkpoint-leave),
// finishing the activity only with no history left.
assert.match(main, /public void onBackPressed\(\) \{[\s\S]*?if \(webView\.canGoBack\(\)\) \{[\s\S]*?webView\.goBack\(\);/,
  'system Back should traverse page history first');
assert.match(main, /public void onBackPressed\(\) \{[\s\S]*?super\.onBackPressed\(\);/,
  'system Back should finish the activity when no history is left');

console.log('test_back_checkpoint: OK');
