import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { zipStore } from '../../app/js/zip.js';
import * as appIntake from '../../app/js/xr-dataset.js';
import * as mirrorIntake from '../../browser/app/js/xr-dataset.js';

// A byte-faithful Android native export (NativeCaptureActivity.writeDataset):
// frame_%04d.jpg + transforms.json (per-frame + top-level intrinsics) +
// seed_points.json ({points, metadata}) + seed.ply + capture.json, nested
// under a folder the way OS compressors pack it. Intake is method-agnostic
// (deflate itself is pinned in test_zip_codec), so STORED fixtures do.
const enc = new TextEncoder();
const J = (o) => enc.encode(JSON.stringify(o));
const jpeg = (seed) => new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 16, seed, 1, 2, 3, 0xff, 0xd9]);
const I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const INTR = { fl_x: 1500.5, fl_y: 1500.5, cx: 960, cy: 540, w: 1920, h: 1080 };
const FRAMES = ['frame_0001.jpg', 'frame_0002.jpg', 'frame_0003.jpg'];
const nativeTransforms = (names) => ({ camera_angle_x: 1.1, ...INTR,
  frames: names.map((p, k) => ({ file_path: p,
    transform_matrix: I4.map((r, i) => r.map((v, j) => (i === 0 && j === 3 ? k * 0.1 : v))),
    ...INTR })) });
const nativeSeeds = { points: [
  { X: [0.1, 0.2, -0.5], rgb: [0.5, 0.4, 0.3], frameName: 'frame_0001.jpg' },
  { X: [0.2, 0.1, -0.6], rgb: [0.2, 0.8, 0.1], frameName: 'frame_0002.jpg' },
], metadata: { seedSource: 'arcore-point-cloud-rgb' } };
const nativeCapture = { frames: 3, pointCount: 2, seedSource: 'arcore-point-cloud-rgb',
  leveling: { method: 'arcore-horizontal-plane', applied: true, tiltDegrees: 1.2, planeArea: 3.5 },
  coordinateConvention: 'ARCore world meters; OpenGL camera-to-world transforms' };
const nativeEntries = () => [
  ...FRAMES.map((n, i) => ({ name: `capture/${n}`, data: jpeg(i + 1) })),
  { name: 'capture/transforms.json', data: J(nativeTransforms(FRAMES)) },
  { name: 'capture/seed_points.json', data: J(nativeSeeds) },
  { name: 'capture/seed.ply', data: enc.encode('ply\nformat ascii 1.0\n') },
  { name: 'capture/capture.json', data: J(nativeCapture) },
];
const zipOf = (entries, name = 'export.zip') => new File([zipStore(entries)], name);

// every intake runs twice: the app source and the browser mirror must BOTH
// accept the export (byte parity is pinned separately in test_browser_vendor;
// this pins the mirror's BEHAVIOR on the Android layout)
for (const [label, intake] of [['app', appIntake], ['browser mirror', mirrorIntake]]) {
  // 1. the clean native layout trains as-is
  const clean = await intake.unpackZipFile(zipOf(nativeEntries()));
  assert.equal(clean.kind, 'capture', `[${label}] the native layout should route to training intake`);
  assert.deepEqual(clean.files.map((f) => f.name).sort(),
    ['capture.json', 'frame_0001.jpg', 'frame_0002.jpg', 'frame_0003.jpg',
      'seed_points.json', 'transforms.json'],
    `[${label}] entries should unpack to basenamed files, seed.ply dropped`);
  assert.deepEqual(clean.skipped, [], `[${label}] a clean export should skip nothing`);
  const ds = await intake.datasetFromSelection(clean.files, { fromZip: true });
  assert.equal(ds.kind, 'ar', `[${label}] poses should mean the solve is skipped`);
  assert.equal(ds.files.length, 3, `[${label}] all three views should survive shaping`);
  assert.equal(ds.transforms.frames.length, 3, `[${label}] all three cameras should match by basename`);
  assert.equal(ds.points.length, 2, `[${label}] the ARCore seed points should survive`);
  assert.equal(ds.metadata.seedSource, 'arcore-point-cloud-rgb', `[${label}] the seed source should survive`);
  assert.equal(ds.metadata.leveling.method, 'arcore-horizontal-plane', `[${label}] the leveling report should survive`);
  assert.equal(ds.dropped, undefined, `[${label}] a clean export should drop nothing`);

  // 2. the real-world folder zip: macOS metadata, a thumbnail, a second
  // capture's stale copy (listed FIRST, so first-wins would keep the wrong
  // bytes), a second camera file, and a stray saved run
  const nasty = await intake.unpackZipFile(zipOf([
    { name: 'old/frame_0001.jpg', data: jpeg(9) },
    ...nativeEntries(),
    { name: '__MACOSX/capture/._frame_0001.jpg', data: jpeg(7) },
    { name: 'capture/.DS_Store', data: new Uint8Array(8) },
    { name: 'capture/preview.jpg', data: jpeg(8) },
    { name: 'capture/cameras.json', data: J({ cameras: [] }) },
    { name: 'model.sog', data: new Uint8Array(8) },
    { name: 'recon.json', data: J({ version: 1, frames: [{ name: 'x.jpg' }] }) },
    { name: 'state.bin', data: new Uint8Array(8) },
  ]));
  assert.equal(nasty.kind, 'capture', `[${label}] photos beside a stray run should still train`);
  const kept = nasty.files.find((f) => f.name === 'frame_0001.jpg');
  assert.deepEqual(new Uint8Array(await kept.arrayBuffer()), jpeg(1),
    `[${label}] the photo next to the camera file should win over old/`);
  const nds = await intake.datasetFromSelection(nasty.files, { fromZip: true });
  assert.equal(nds.kind, 'ar', `[${label}] the messy zip should still train as one capture`);
  assert.equal(nds.files.length, 3, `[${label}] the three posed views should survive the mess`);
  assert.equal(nds.transforms.frames.length, 3, `[${label}] transforms.json should win over cameras.json`);
  assert.equal(nds.points.length, 2, `[${label}] the seeds should survive the mess`);
  const leftOut = [...nasty.skipped, ...(nds.dropped || [])].join('\n');
  for (const n of ['._frame_0001.jpg', '.DS_Store', 'preview.jpg', 'cameras.json',
    'old/frame_0001.jpg', 'recon.json']) {
    assert.ok(leftOut.includes(n), `[${label}] should say it left out ${n}`);
  }
  assert.ok(leftOut.includes('next to the camera file'),
    `[${label}] should say WHY the stale copy lost`);

  // 3. the sniffer: photos-first, junk-blind, session zips still restore
  assert.equal(intake.zipEntryKind(['shots/a.jpg', 'recon.json', 'state.bin']), 'capture',
    `[${label}] photos beside session markers should train, not restore`);
  assert.equal(intake.zipEntryKind(['recon.json', 'state.bin', 'model.sog']), 'session',
    `[${label}] a pure session zip should still restore`);
  assert.equal(intake.zipEntryKind(['__MACOSX/._a.jpg', 'shots/.DS_Store']), null,
    `[${label}] OS metadata alone should sniff as neither`);
  assert.equal(intake.isZipJunk('__MACOSX/capture/._frame_0001.jpg'), true);
  assert.equal(intake.isZipJunk('capture/Thumbs.db'), true);
  assert.equal(intake.isZipJunk('capture/frame_0001.jpg'), false);
}

// 4. hand picks stay strict: the leniency is for folder zips only
{
  const clean = await appIntake.unpackZipFile(zipOf(nativeEntries()));
  const stray = new File([jpeg(5)], 'preview.jpg', { type: 'image/jpeg' });
  await assert.rejects(appIntake.datasetFromSelection([...clean.files, stray]), /No known camera for preview/,
    'a hand-picked stray photo should still fail loudly');
  const extra = new File([J({ cameras: [] })], 'cameras.json', { type: 'application/json' });
  await assert.rejects(appIntake.datasetFromSelection([...clean.files, extra]), /only one camera JSON/,
    'hand-picked second camera files should still fail loudly');
}

// 5. wiring: the app passes zip provenance through and names the left-outs
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
assert.match(app, /datasetFromSelection\(all, fromZip \? \{ fromZip: true \} : \{\}\)/,
  'zip intake should shape leniently, hand picks strictly');
assert.match(app, /Left out of the zip import:/,
  'the app should name what the zip import left out');

// 6. the parity script must keep carrying the fix to the mirror (it copies
// app/js whole, so a future "simplification" that drops the intake files
// would silently re-break the browser — fail loudly instead)
const syncSh = readFileSync(new URL('../../tools/sync-browser.sh', import.meta.url), 'utf8');
assert.match(syncSh, /cp -r app\/js\/\. browser\/app\/js\//,
  'the parity script must keep mirroring the intake sources');
assert.match(syncSh, /check_dir app\/js browser\/app\/js/,
  'the parity script must keep verifying the mirrored sources');

console.log('android capture zips (app + browser mirror) accepted, strays named, hand picks strict: passed');
