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
assert.ok(app.includes("checkpointRun('back').catch(() => {}).finally(() => { location.href = 'index.html?back=1'; });"),
  'popstate checkpoints before leaving too');
assert.ok(app.includes("history.pushState({ sj: 'run' }, '')"),
  'training owns a history entry so Back pops into the checkpoint-leave');
assert.ok(app.includes("else if (mp.get('back')) headlessClearMarker().finally(() => mountWall());"),
  'the deliberate-leave boot stands the handoff down instead of recovering');
assert.ok(app.includes('async function headlessClearMarker()'), 'the marker stand-down exists');

// Android system Back routes through the page history (same checkpoint-leave),
// finishing the activity only with no history left.
assert.match(main, /public void onBackPressed\(\) \{[\s\S]*?if \(webView\.canGoBack\(\)\) \{[\s\S]*?webView\.goBack\(\);/,
  'system Back should traverse page history first');
assert.match(main, /public void onBackPressed\(\) \{[\s\S]*?super\.onBackPressed\(\);/,
  'system Back should finish the activity when no history is left');

console.log('test_back_checkpoint: OK');
