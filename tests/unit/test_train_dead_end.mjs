import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { litSample } from '../../src/index.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const slice = (from, to) => {
  const s = app.indexOf(from);
  const e = app.indexOf(to, s);
  assert.ok(s >= 0 && e > s, `${from} should precede ${to}`);
  return app.slice(s, e);
};

// ---- trainDeadEnd: an unrecoverable failure tears down to setup ----
const dead = slice('function trainDeadEnd() {', 'function backToSetup() {');
assert.match(dead, /S\.session && S\.session\.dispose\(\);/,
  'the dead end disposes the faulted session');
assert.match(dead, /S\.gen\+\+;\n  S\.state = 'ready';/,
  'the dead end invalidates stragglers and leaves the train state');
assert.match(dead, /clearTimeout\(S\._presentWd && S\._presentWd\.timer\);/,
  'the dead end parks the present watchdog');
assert.match(dead, /dock\(''\);\n  backToSetup\(\);/,
  'the dead end clears the dock and lands on the setup card');

// the fresh pipeline's catch and its placed<3 bail share the helper
assert.match(app, /reconstruction\.'\)\);\n    trainDeadEnd\(\);/,
  'a failed solve/seed should tear down through trainDeadEnd');
assert.match(app, /retrying usually clears it\.'\);\n      trainDeadEnd\(\);/,
  'the placed<3 bail should tear down through trainDeadEnd');

// device loss the recovery cannot bring back: actionable card, then teardown
const rec = slice('async function deviceLostRecovery(', 'iOS background guard');
assert.match(rec, /back at the setup screen/,
  'an unrecoverable device should say where the user landed');
assert.match(rec, /lower Training resolution/,
  'an unrecoverable device should name the actionable retry');
assert.match(rec, /else \{[\s\S]*back at the setup screen[\s\S]*trainDeadEnd\(\);/,
  'a non-file-backed failed recovery should tear the dead train screen down');
assert.match(rec, /if \(S\._hlStaged\)[\s\S]*showRecovery\(runId/,
  'a file-backed handback failure should remain recoverable instead of returning Home');

// continuations rebuild after tearing the viewer down — a failure in the
// rebuild (e.g. higher resolution OOMs the seed) must land on setup too,
// not in the torn-down limbo with the old chart still up
const cont = slice('async function continueLocalRun(', 'async function trainAfterDone()');
assert.match(cont, /const gen = S\.gen;\n  try \{/,
  'the continuation rebuild should run guarded');
assert.match(cont, /\} catch \(e\) \{\n    if \(S\.gen !== gen\) throw e;/,
  'a superseded continuation must not tear down the screen that replaced it');
assert.match(cont, /\n    trainDeadEnd\(\);\n    throw e;/,
  'a failed rebuild should tear down, then let the caller flash the why');

// ---- the silent-black watchdog: accepted renders, black screen ----
const arm = slice('function armPresentWatchdog() {', 'function syncTrainingVisualsControl()');
assert.match(arm, /!S\.session\._remote/,
  'local views present in-thread — the watchdog is remote-only');
assert.match(arm, /S\.state !== 'train' && S\.state !== 'done'/,
  'the watchdog only arms where the model surface is live');
assert.match(arm, /S\._capView && S\._capView\.active/,
  'an engaged snapshot view needs no watch');
assert.match(arm, /S\._presentWd\.canvas === gpuCanvas/,
  'arming is idempotent per canvas');
assert.match(arm, /setTimeout\(\(\) => presentWatchdogFire\(canvas, gen, 0\), 8000\)/,
  'the check fires once, shortly after the canvas path goes live');

const fire = slice('async function presentWatchdogFire(', 'function syncTrainingVisualsControl()');
assert.match(fire, /view\._diagnoseNoPresent\(\)/,
  'the fire probes the raster through the canvas-free diagnosis');
assert.match(fire, /d\.kind === 'unknown'\) \{ rehear\(\); return; \}/,
  'an unconcluded probe rehears once instead of disarming silently');
assert.match(fire, /S\._presentWd = null;/,
  'a spent arm releases the canvas so a later transition re-arms');
assert.match(fire, /blackProbes < 1/,
  'a black probe gets one rehearing before crying raster');
assert.match(fire, /presentWatchdogFire\(canvas, gen, 1\), 15000\)/,
  'the rehearing comes 15s later');
assert.match(fire, /Model view is black/,
  'a twice-black raster surfaces as words (snapshots could not help)');
assert.match(fire, /drawImage\(canvas, 0, 0, w, h\)/,
  'a lit raster is checked against the placeholder pixels');
assert.match(fire, /litSample\(sctx\.getImageData/,
  'the placeholder reads through the shared sampler');
assert.match(fire, /engageCaptureView\('the live canvas stays black on this device'\)/,
  'raster-lit but placeholder-black engages the snapshot view');

const sync = slice('function syncTrainingVisualsControl() {', 'function toggleTrainingVisuals()');
assert.match(sync, /\n  armPresentWatchdog\(\);/,
  'every transition into the live canvas path re-arms the check');

assert.match(app, /litSample \} from '\.\.\/\.\.\/src\/index\.js';/,
  'the app should sample through the engine export');
assert.equal(typeof litSample, 'function', 'the engine should export litSample');

console.log('train dead-end teardown + silent-black watchdog: passed');
