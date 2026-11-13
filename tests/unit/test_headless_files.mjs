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

// ── retired service handoff ────────────────────────────────────────────────
// Keep the parser/validator covered above for compatibility with files left by
// older APKs, but never activate cross-WebView WebGPU ownership again.
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const svc = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');
const manifest = readFileSync(new URL('../../android/app/src/main/AndroidManifest.xml', import.meta.url), 'utf8');
assert.match(app, /const headlessBridge = \(\) => null;/,
  'service-owned WebGPU handoff must stay disabled');
const boot = app.slice(app.indexOf('function boot()'), app.indexOf('// WebGPU probe'));
assert.doesNotMatch(boot, /headlessBootCheck|headlessClearMarker/,
  'app boot must ignore stale service-run markers and open the local wall');
const start = app.slice(app.indexOf('function startTraining()'), app.indexOf('function buildSceneFromSession'));
assert.doesNotMatch(start, /stageHeadlessRun/,
  'starting a run must not stage a second-process trainer');
const preflightAt = app.indexOf('async function startPrep()');
const preflight = app.slice(preflightAt, preflightAt + 3000);
assert.doesNotMatch(preflight, /headlessMarkerLive|headlessBootCheck/,
  'stale service files must never block a new local Start');
assert.doesNotMatch(main, /ACTION_HANDOFF_TRAINING|activity-background|handoffDone|headlessCommand/,
  'the Activity must never transfer a live trainer to the service');
assert.doesNotMatch(main, /writeRunJson|writeRunChunk|setActiveRun|recoverActiveRun/,
  'the disabled bridge must not stage or adopt headless run files');
assert.doesNotMatch(svc, /import android\.webkit|new WebView|AndroidHeadless|headless-|RunFiles|activity-foreground/,
  'the notification service must not own a WebView, GPU, or run files');
assert.doesNotMatch(manifest, /android:process=":training"/,
  'the notification-only service should not create a second app process');
assert.match(svc, /onTaskRemoved\(Intent rootIntent\)[\s\S]*teardown\(\);/,
  'swiping away the Activity should stop the notification, not start a trainer');
assert.match(main, /onDestroy\(\)[\s\S]*stopTrainingService\(\);/,
  'destroying the Activity should always stop its notification service');
console.log('test_headless_files: retired handoff compatibility OK');
