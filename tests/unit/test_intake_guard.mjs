// Two on-device bugs, pinned at the source level (the suite's style for app.js):
// (1) swiped-away training froze: hidden pages lose rAF and their worker
// timers throttle, so frames pump through a MessageChannel and headless
// progress rides the metrics beat, not an interval; (2) Start silently died
// when a landing set's open() bumped the generation mid-startPrep, and a
// zombie service-owned marker blocked every new start until the monitor
// intervened.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');
const xr = readFileSync(resolve(root, 'app/js/xr-ui.js'), 'utf8');
const driver = readFileSync(resolve(root, 'app/js/train-headless.js'), 'utf8');
const session = readFileSync(resolve(root, 'src/session.js'), 'utf8');
const mainAct = readFileSync(resolve(root, 'android/app/src/main/java/space/arrival/splatxr/MainActivity.java'), 'utf8');
const service = readFileSync(resolve(root, 'android/app/src/main/java/space/arrival/splatxr/TrainingService.java'), 'utf8');

// --- the intake gate: the first tap queues while a set is still landing ---
const prepStart = app.indexOf('async function startPrep()');
const prep = app.slice(prepStart, app.indexOf('document.getElementById(\'failcard\')', prepStart));
assert.match(prep, /if \(S\._startBusy\) return false;/,
  'one Start request should own the button and suppress double taps');
assert.match(prep, /go\.disabled = true;[\s\S]*Preparing photos…/,
  'a fast Start tap should immediately show visible work on the button');
assert.match(prep, /while \(\(S\._intakeN \|\| 0\) > 0[\s\S]*setTimeout\(resolve, 50\)/,
  'Start should wait for the intake wrapper instead of rejecting the first tap');
assert.match(app, /await startPrep\(\);\n  \}\);/,
  'the click handler should await the whole Start request');
assert.match(app, /if \(autostart\) await startPrep\(\);/,
  'picker autostart should await and surface the same guarded request');

assert.match(app,
  /async function useOwnPhotos\(list, capture = null\) \{\n  S\._intakeN = \(S\._intakeN \|\| 0\) \+ 1;\n  try \{ return await useOwnPhotosInner\(list, capture\); \}\n  finally \{ S\._intakeN--; \}/,
  'photo intake should hold the counter through open()');
assert.match(app,
  /async function useOwnVideo\(file\) \{\n  S\._intakeN = \(S\._intakeN \|\| 0\) \+ 1;\n  try \{ return await useOwnVideoInner\(file\); \}\n  finally \{ S\._intakeN--; \}/,
  'video intake should hold the counter through open()');
assert.match(app, /async function useOwnPhotosInner\(list, capture = null\) \{/,
  'the photo body should keep its name + Inner');
assert.match(app, /async function useOwnVideoInner\(file\) \{/,
  'the video body should keep its name + Inner');

const openFn = app.slice(app.indexOf('async function open(preset'), app.indexOf('const BEATS_BASE'));
assert.match(openFn, /S\._intakeN = \(S\._intakeN \|\| 0\) \+ 1;\n    try \{ await loadPresetList\(preset\); \}/,
  "open()'s own pre-gen await should hold the counter");

assert.match(app, /setBusy: \(on\) => \{ S\._intakeN = Math\.max\(0, \(S\._intakeN \|\| 0\) \+ \(on \? 1 : -1\)\); \}/,
  'the capture UI should get a clamped counter setter');
assert.match(xr, /initXRUI\(\{ onCapture, flash, setBusy \}\)/, 'initXRUI should accept setBusy');
assert.equal((xr.match(/setBusy\?\.\(true\)/g) || []).length, 2,
  'both capture paths (browser pre-modal, native import) should raise the counter');
assert.equal((xr.match(/setBusy\?\.\(false\)/g) || []).length, 2,
  'both capture paths should lower the counter in a finally');
assert.match(xr, /setBusy\?\.\(true\);\n      let dataset = null;\n      try \{ dataset = await captureXR\(\); \}/,
  'browser capture should hold the counter across its pre-modal await');
assert.match(xr, /splatxr-native-capture', async \(\) => \{\n      \/\/[\s\S]{0,300}?setBusy\?\.\(true\);/,
  'the native import should hold the counter from the event');

// --- service handoff retired: Start can never be blocked by a stale marker ---
assert.match(app, /const headlessBridge = \(\) => null;/,
  'the unreliable cross-renderer training handoff should remain disabled');
assert.doesNotMatch(mainAct, /headlessCommand|handoffDone|ACTION_HANDOFF_TRAINING/,
  'the Activity bridge should expose no service trainer controls');
assert.doesNotMatch(service, /new WebView|AndroidHeadless|headless-ping|RunFiles/,
  'the notification service should never create a second trainer');

// The Activity WebView may still make progress while Android leaves it alive;
// unlike the retired service twin this uses the one original session and no
// state migration.
assert.match(session, /if \(isHidden\(\) && this\._sched\.pump\) this\._sched\.pump\(\);/,
  'a hidden Activity page should pump frames through the channel when allowed');
assert.match(session, /ch\.port1\.onmessage = \(\) => runFrame\(\);/,
  'the pump should deliver frames, not just ticks');
assert.match(session, /ch\.port1\.unref\?\.\(\); ch\.port2\.unref\?\.\(\);/,
  'the pump must not hold a Node event loop open by itself');

console.log('intake guard + single-Activity background fallback pinned');
