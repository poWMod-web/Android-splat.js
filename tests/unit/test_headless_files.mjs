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
// the swipe-away handoff: the service asks, the page secures + reports back
assert.ok(app.includes('globalThis.__splatxrHandoff'), 'app exposes the handoff hook');
assert.ok(app.includes('b.handoffDone(runId, ok)'), 'the handoff verdict rides handoffDone');
const hook = app.slice(app.indexOf('globalThis.__splatxrHandoff'), app.indexOf('const trainBridge'));
assert.ok(hook.includes('secureFileCheckpoint'), 'the handoff hook is the app side that still secures files');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
assert.ok(main.includes('ACTION_HANDOFF_TRAINING'), 'the handoff broadcast exists');
assert.ok(main.includes('public void handoffDone(String runId, boolean ok)'), 'the bridge carries the verdict');
assert.ok(app.includes("localStorage.getItem('splatjs_monitor_visuals') === '1'"),
  'the monitor visuals toggle defaults off, like the in-app training visuals');
const svc = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');
for (const s of ['setDataDirectorySuffix("training")', 'AndroidHeadless', 'headless-snapshot', 'onTaskRemoved',
    'headless-ack', 'headless-ready', 'settleHandoff', 'HANDOFF_ACK_MS', 'handoffGen']) {
  assert.ok(svc.includes(s), `service implements ${s}`);
}
console.log('test_headless_files: OK');
