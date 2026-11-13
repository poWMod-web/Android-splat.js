import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  RUNS_BASE, runUrl, photoUrl, activeUrl, fresh,
  validRunId, validName, buildManifest, validateManifest,
  isLiveStatus, isOwnedBy,
  HEARTBEAT_STALE_MS, heartbeatFresh, encodeB64,
  snapshotSize, snapshotCam,
} from '../../app/js/headless.js';

// ── URL + validation ────────────────────────────────────────────────────────
assert.equal(RUNS_BASE, 'https://appassets.androidplatform.net/runs/');
assert.equal(runUrl('run_1_2', 'state.bin'),
  'https://appassets.androidplatform.net/runs/run_1_2/state.bin');
assert.equal(photoUrl('run_1_2', 'frame_0001.jpg'),
  'https://appassets.androidplatform.net/runs/run_1_2/photos/frame_0001.jpg');
assert.equal(activeUrl(), 'https://appassets.androidplatform.net/runs/active.json');
assert.ok(fresh(runUrl('r', 's')).includes('?t='));
// traversal never reaches the filesystem on either side
assert.ok(validRunId('run_1720000000000_123456'));
assert.ok(!validRunId('../x') && !validRunId('a/b') && !validRunId('') && !validRunId(null));
assert.ok(validName('frame_0001.jpg') && validName('state.bin'));
assert.ok(!validName('../state.bin') && !validName('..') && !validName('a/b') && !validName(''));

// ── manifest ────────────────────────────────────────────────────────────────
const m = buildManifest({
  runId: 'run_1_2', name: 'Local Scene', maxIters: 20000, gpu: 100,
  srcKey: 'k', photos: ['a.jpg', 'b.jpg', 'c.jpg'], photoSource: 'files',
  settings: { buf: 1 },
});
assert.equal(validateManifest(m), m);
assert.equal(m.version, 1);
assert.throws(() => validateManifest({ ...m, version: 2 }), /version/);
assert.throws(() => validateManifest({ ...m, runId: '../x' }), /runId/);
assert.throws(() => validateManifest({ ...m, maxIters: 500 }), /horizon/);
assert.throws(() => validateManifest({ ...m, photos: ['a.jpg'] }), /photo list/);
assert.throws(() => validateManifest({ ...m, photos: ['a.jpg', 'b.jpg', '../c'] }), /photo list/);
assert.throws(() => validateManifest({ ...m, photoSource: 'blobs' }), /photoSource/);
assert.throws(() => validateManifest(null), /object/);

// ── status helpers ──────────────────────────────────────────────────────────
for (const s of ['staging', 'booting', 'active', 'paused']) {
  assert.ok(isLiveStatus({ status: s }), `${s} is live`);
}
for (const s of ['finished', 'interrupted', null, undefined, 'bogus']) {
  assert.ok(!isLiveStatus({ status: s }), `${s} is not live`);
}
assert.ok(isOwnedBy({ owner: 'service' }, 'service'));
assert.ok(!isOwnedBy({ owner: 'activity' }, 'service'));

// ── heartbeat ───────────────────────────────────────────────────────────────
assert.equal(HEARTBEAT_STALE_MS, 30000);
assert.ok(heartbeatFresh({ alive: true, at: Date.now() - 5000 }));
assert.ok(!heartbeatFresh({ alive: true, at: Date.now() - 60000 }));
assert.ok(!heartbeatFresh({ alive: false, at: Date.now() }));
assert.ok(!heartbeatFresh(null));

// ── base64 ──────────────────────────────────────────────────────────────────
const bytes = new Uint8Array(30000);
for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 37 + 11) & 255;
assert.equal(encodeB64(bytes), Buffer.from(bytes).toString('base64'));
assert.equal(encodeB64(new Uint8Array(0)), '');
assert.equal(encodeB64(new Uint8Array([255, 0, 1])), '/wAB');

// ── snapshot geometry ───────────────────────────────────────────────────────
assert.deepEqual(snapshotSize(1920, 1080), { w: 512, h: 288 });
assert.deepEqual(snapshotSize(400, 400), { w: 400, h: 400 });
const recon = {
  cams: [{ imgIdx: 0, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 1], f: 1000, cx: 960, cy: 540 }],
  frames: [{ name: 'a.jpg', tw: 1920, th: 1080 }],
};
const cam = snapshotCam(recon);
assert.deepEqual([cam.w, cam.h], [512, 288]);
assert.ok(Math.abs(cam.f - 1000 * 512 / 1920) < 1e-9);
assert.ok(Math.abs(cam.cx - 256) < 1e-9 && Math.abs(cam.cy - 144) < 1e-9);
assert.equal(snapshotCam({ cams: [], frames: [] }), null);
assert.equal(snapshotCam(null), null);

// ── wiring pins: the driver + app implement the vocabulary this module defines
const driver = readFileSync(new URL('../../app/js/train-headless.js', import.meta.url), 'utf8');
for (const hook of ['__headlessToggle', '__headlessPause', '__headlessResume', '__headlessStop', '__headlessSnapshot']) {
  assert.ok(driver.includes(hook), `driver exposes ${hook}`);
}
for (const call of ['postProgress', 'writeRunJson', 'writeRunChunk', 'writeSnapshotChunk', 'finishRun', 'stopRun', 'abortRun']) {
  assert.ok(driver.includes(call), `driver calls bridge.${call}`);
}
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
for (const fn of ['stageHeadlessRun', 'secureFileCheckpoint', 'enterMonitor', 'adoptHeadlessRun']) {
  assert.ok(app.includes(`function ${fn}`), `app implements ${fn}`);
}
// transitions only: no periodic file checkpoints anywhere on either side
for (const [src, name] of [[app, 'app'], [driver, 'driver']]) {
  assert.ok(!src.includes('maybeFileCheckpoint') && !src.includes('maybeCheckpoint'),
    `${name} writes file checkpoints on transitions only, never periodically`);
}
assert.ok(!readFileSync(new URL('../../app/js/headless.js', import.meta.url), 'utf8').includes('checkpointDue'),
  'the cadence gate is gone with the periodics');
// foreground-service handoff: Activity background/task removal asks, the page secures + relinquishes GPU
assert.ok(app.includes('globalThis.__splatxrHandoff'), 'app exposes the handoff hook');
assert.ok(app.includes('b.handoffDone(runId, ok)'), 'the handoff verdict rides handoffDone');
const hook = app.slice(app.indexOf('globalThis.__splatxrHandoff'), app.indexOf('const trainBridge'));
assert.ok(hook.includes('secureFileCheckpoint'), 'the handoff hook is the app side that still secures files');
assert.match(hook, /if \(ok[\s\S]*S\.session\.pause\(\); S\.session\.dispose\(\)[\s\S]*enterMonitor\(runId\)/,
  'a successful handoff should dispose the Activity GPU before becoming a monitor');
assert.match(hook, /activity\.json[^]*alive: false/,
  'the successful handoff should retire the Activity heartbeat before service boot');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
assert.ok(main.includes('ACTION_HANDOFF_TRAINING'), 'the handoff broadcast exists');
assert.ok(main.includes('public void handoffDone(String runId, boolean ok)'), 'the bridge carries the verdict');
assert.match(main, /void onStop\(\)[^]*activity-background/,
  'normal Activity backgrounding should ask the foreground service to own training');
assert.match(main, /void onStart\(\)[^]*notifyActivityVisible\(\)/,
  'a quick foreground return should cancel only the debounced handoff');
assert.ok(app.includes("localStorage.getItem('splatjs_monitor_visuals') === '1'"),
  'the monitor visuals toggle defaults off, like the in-app training visuals');
const svc = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');
assert.match(svc, /case "activity-visible":[\s\S]*removeCallbacks\(backgroundHandoff\)[\s\S]*break;/,
  'foreground visibility must not stop an already-running service trainer');
const runFiles = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/RunFiles.java', import.meta.url), 'utf8');
for (const s of ['setDataDirectorySuffix("training")', 'AndroidHeadless', 'headless-snapshot', 'onTaskRemoved',
    'headless-ack', 'headless-ready', 'settleHandoff', 'HANDOFF_ACK_MS', 'handoffGen',
    'activity-background', 'BACKGROUND_HANDOFF_DELAY_MS', 'SUCCESS_TAKEOVER_DELAY_MS',
    'ORPHAN_TAKEOVER_DELAY_MS', 'beginHandoff']) {
  assert.ok(svc.includes(s), `service implements ${s}`);
}

// Notification return is monitor-only until explicit user consent.
assert.match(main, /void onNewIntent\(Intent intent\)[\s\S]*notifyActivityVisible\(\)[\s\S]*__splatxrOpenMonitor/,
  'a notification tap should bring the monitor forward without taking ownership');
assert.match(app, /__splatxrOpenMonitor = \(\) => \{[\s\S]*headlessBootCheck\(\)/,
  'an existing Activity should reopen only the file-backed monitor');
assert.doesNotMatch(main, /requestTrainingReturn|activity-foreground/,
  'foregrounding must not request an automatic service-to-Activity transfer');
assert.doesNotMatch(svc, /case "activity-foreground"|returnToActivity|returnHandoff/,
  'the service must have no automatic foreground-return path');
assert.match(svc, /beat\.put\("alive", false\)[\s\S]*RunFiles\.writeJson[\s\S]*service\.json/,
  'Java should acknowledge service GPU teardown before Activity adoption');
assert.match(driver, /const secured = await secureFileCheckpoint\(\);[\s\S]*if \(!secured\)[\s\S]*ses\.start\(\)/,
  'a failed handback checkpoint should keep service training instead of returning stale state');
assert.match(app, /status\.status === 'interrupted' && status\.stopped[\s\S]*showRecovery\(runId, status\)/,
  'a stopped service checkpoint should wait on recovery UI for Continue here');
assert.match(app, /'mon-continue'\)\.addEventListener\('click'[\s\S]*returnHeadlessRun\(S\._monRun\)/,
  'only the Continue here button should request service ownership');
assert.match(app, /service\.alive === false[\s\S]*Headless\.heartbeatFresh\(service\)/,
  'Activity adoption should wait for the post-teardown service acknowledgment');

const adoptStart = app.indexOf('async function adoptHeadlessRun(runId, mode)');
const adopt = app.slice(adoptStart, app.indexOf('/** The run died out of reach', adoptStart));
assert.match(adopt, /closeOverlays\(\);[\s\S]*dock\(''\);[\s\S]*cv-monitor[\s\S]*window\.scrollTo\(0, 0\)/,
  'explicit adoption should clear the monitor, old dock, and mobile home layout first');
const monitorStart = app.indexOf('function enterMonitor(runId)');
const monitor = app.slice(monitorStart, app.indexOf('function wireMonitorOnce()', monitorStart));
assert.match(monitor, /closeOverlays\(\);[\s\S]*dock\(''\);[\s\S]*cv-model/,
  'the monitor should not sit over the Activity training chart or model canvas');
assert.match(adopt, /if \(mode === 'finished'\) \{[\s\S]*await continueLocalRun\(true\);[\s\S]*headlessClear\(runId\);/,
  'a finished handback should clear its marker only after the viewer rebuilds');
assert.match(adopt, /await continueLocalRun\(false\);[\s\S]*S\.state !== 'train'[\s\S]*owner: 'activity'/,
  'a continuing handback should publish Activity ownership only after a live trainer exists');
const continueBranch = adopt.indexOf('await continueLocalRun(false)');
assert.ok(continueBranch > adopt.indexOf('headlessClear(runId)') &&
    !adopt.slice(adopt.lastIndexOf('} else {', continueBranch), continueBranch).includes('headlessClear(runId)'),
  'a continuing handback must never delete its recovery marker before rebuild');
assert.match(adopt, /Transaction rollback:[\s\S]*showRecovery\(runId/,
  'a failed local rebuild should retain files and show recovery instead of Home');
assert.match(app, /source: 'urls', complete: true/,
  'native AR handoff should mark archive references complete without copying images');
assert.match(main, /public String recoverActiveRun\(\)[\s\S]*RunFiles\.recoverActive/,
  'the Activity bridge should expose orphaned handback recovery');
assert.match(runFiles, /String recoverActive\(Context ctx\)[\s\S]*state\.length\(\) < 16[\s\S]*photos\.optBoolean\("complete"/,
  'orphan recovery should require exact state and retained AR photos');
assert.match(runFiles, /"finished"\.equals\(value\)[\s\S]*"discarded"\.equals\(value\)[\s\S]*"dismissed"\.equals\(value\)/,
  'orphan recovery must not resurrect completed or deliberately abandoned runs');
assert.match(app, /typeof b\.recoverActiveRun === 'function' \? b\.recoverActiveRun\(\) : null/,
  'boot should recover a durable run directory when old code deleted active.json');
assert.match(app, /status: 'discarded'[\s\S]*headlessClear\(runId\)/,
  'Discard should mark its directory so orphan recovery respects the choice');
assert.match(app, /A file-backed service handback cannot use Session\.recover[\s\S]*restartForGpuRecovery[\s\S]*showRecovery\(runId/,
  'a handback device loss should rebuild in a fresh Android renderer or remain on file-backed recovery');
assert.match(main, /public void restartForGpuRecovery\(String runId\)[\s\S]*gpuRendererRestart = true;[\s\S]*recreate\(\)/,
  'Android should replace the poisoned visible WebView for one handback retry');
assert.match(main, /WebView oldWebView = webView;[\s\S]*oldWebView\.destroy\(\)/,
  'Activity teardown should explicitly destroy the old WebView renderer');
assert.match(app, /service\.alive === false[\s\S]*return true;[\s\S]*headlessCommand\('headless-stop'/,
  'a dead service should not be restarted and allowed to overwrite GPU-retry metadata');

// Graph + source durability span Activity, service, Android files, and IndexedDB.
const sessionIo = readFileSync(new URL('../../app/js/session_io.js', import.meta.url), 'utf8');
assert.match(sessionIo, /stats:[\s\S]*chart:[\s\S]*holds:[\s\S]*events:/,
  'the reconstruction transaction should carry score history and graph events');
assert.match(driver, /const priorChart = \(recon\.stats && recon\.stats\.chart\) \|\| \[\];[\s\S]*ses\.setLossHistory\(priorChart\)/,
  'service training should append metrics to the Activity score graph');
assert.match(driver, /holdHist: \(recon\.stats && recon\.stats\.holds\)[\s\S]*chartEvents/,
  'service checkpoints should carry holds and graph annotations forward');
const fileCheckpoint = app.slice(app.indexOf('async function secureFileCheckpoint()'),
  app.indexOf('/** Best-effort status write', app.indexOf('async function secureFileCheckpoint()')));
assert.match(fileCheckpoint, /writeRunJson\(S\.runId, 'recon\.json'[\s\S]*patchRun\(S\.runId,[\s\S]*state: new Blob[\s\S]*detailProgress/,
  'every Android-file handoff checkpoint should be mirrored into IndexedDB with its graph');
assert.match(app, /async function mirrorHeadlessRun\(runId, status\)[\s\S]*state\.bin[\s\S]*saveRun\(rec\)/,
  'boot should rediscover an Android run directory as a Home library record');
assert.match(app, /async function nativeCaptureTiles\(\)[\s\S]*listCaptureArchives\(\)[\s\S]*importNativeCapture\(b, row\.captureId\)/,
  'Home should enumerate native Android scan archives independently of IndexedDB');
assert.match(main, /public String listRecoverableRuns\(\)[\s\S]*RunFiles\.listDurableRuns/,
  'the bridge should expose complete Android run directories to Home');
assert.match(runFiles, /List<String> listDurableRuns\(Context ctx\)[\s\S]*state\.bin[\s\S]*photos\.optBoolean\("complete"/,
  'Android discovery should require a checkpoint and complete retained images');
assert.match(app, /async function mirrorAndroidRunDirs\(\)[\s\S]*listRecoverableRuns\(\)[\s\S]*mirrorHeadlessRun/,
  'Home should rebuild missing IndexedDB run records from Android files');
const stage = app.slice(app.indexOf('function stageHeadlessRun()'), app.indexOf('/** Upload photo bytes'));
assert.doesNotMatch(stage, /cleanRunDirs|resetCaptureSession/,
  'handoff must not delete old run directories or native scan archives');
assert.match(stage, /photoSource = native \? 'urls' : 'files'[\s\S]*photoUrls: files\.map/,
  'native handoff manifests should reference canonical archive URLs');
assert.doesNotMatch(stage, /retainCapture/,
  'native handoff must not copy canonical photos into each run');
const store = readFileSync(new URL('../../app/js/store.js', import.meta.url), 'utf8');
assert.doesNotMatch(store, /KEEP\s*=|pruneRuns|slice\(KEEP\)/,
  'training history must not be implicitly evicted after a fixed count');
assert.doesNotMatch(app, /a dead tombstone:[\s\S]*deleteRun\(r\.id\)/,
  'Home must not silently delete incomplete training records');
console.log('test_headless_files: OK');
