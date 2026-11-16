import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { zipStore } from '../../app/js/zip.js';
import { unpackZipFile, zipEntryKind, datasetFromSelection } from '../../app/js/xr-dataset.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const appHtml = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const enc = new TextEncoder();

// an exported capture zip: photos + camera JSON (+ seeds, manifest, readme),
// nested under a folder the way OS compressors do it
const I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const transforms = { camera_angle_x: 0.7, frames: ['f0.jpg', 'f1.jpg', 'f2.jpg'].map((p, k) => ({
  file_path: p, transform_matrix: I4.map((r, i) => r.map((v, j) => (i === 0 && j === 3 ? k : v))),
})) };
const fakeJpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
const capZip = zipStore([
  { name: 'set/f0.jpg', data: fakeJpeg },
  { name: 'set/f1.jpg', data: fakeJpeg },
  { name: 'set/f2.jpg', data: fakeJpeg },
  { name: 'set/transforms.json', data: enc.encode(JSON.stringify(transforms)) },
  { name: 'set/seed_points.json', data: enc.encode(JSON.stringify({ points: [] })) },
  { name: 'set/capture.json', data: enc.encode(JSON.stringify({ frames: 3 })) },
  { name: 'set/README.txt', data: enc.encode('hi') },
]);
const cap = await unpackZipFile(new File([capZip], 'splatxr-capture-1.zip'));
assert.equal(cap.kind, 'capture', 'a photos+poses zip should route to training intake');
assert.deepEqual(cap.files.map((f) => f.name).sort(),
  ['capture.json', 'f0.jpg', 'f1.jpg', 'f2.jpg', 'seed_points.json', 'transforms.json'],
  'entries should unpack to basenamed files, READMEs dropped');
assert.ok(cap.files.filter((f) => f.type === 'image/jpeg').length === 3,
  'photos should type as images by extension');
assert.ok(cap.files.filter((f) => f.type === 'application/json').length === 3,
  'camera/seed/manifest JSONs should come along');
assert.ok(cap.bytes.length > 0, 'the raw bytes should ride along for session restores');

// end to end: unpacked files flow through the known-pose intake shaping
const dataset = await datasetFromSelection(cap.files);
assert.equal(dataset.kind, 'ar', 'poses in the zip should mean the solve is skipped');
assert.equal(dataset.files.length, 3, 'all three views should survive shaping');
assert.equal(dataset.transforms.frames.length, 3, 'all three cameras should match by basename');

// session zips keep restoring; junk zips are skipped, loudly at the call site
const sessZip = zipStore([
  { name: 'recon.json', data: enc.encode('{}') },
  { name: 'state.bin', data: new Uint8Array(4) },
]);
const sess = await unpackZipFile(new File([sessZip], 'run.zip'));
assert.equal(sess.kind, 'session', 'a session zip should restore, not train');
assert.equal(sess.files.length, 0, 'a session zip should unpack no photo files');
assert.equal(zipEntryKind(['README.txt', 'notes.md']), null,
  'a zip with no photos or session should sniff as neither');

// ---- wiring: picker + drop both ride the photo flow -------------------------
assert.match(appHtml, /accept="image\/\*,video\/\*,\.json,\.zip,/,
  'the file picker should accept capture zips');
assert.match(app, /unpackZipFile \} from '\.\/xr-dataset\.js';/,
  'intake should import the zip unpacker');
assert.match(app, /if \(f && \/\\.zip\$\/i\.test\(f\.name \|\| ''\)\) \{/,
  'picked zips should expand before the photo flow');
assert.match(app, /if \(u\.kind === 'session'\) \{ restoreSession\(\{ bytes: u\.bytes \}\); return; \}/,
  'a picked session zip should restore instead of training');
assert.match(app, /if \(files\.length === 1 && \/\\\.\(sog\|ply\)\$\/i\.test\(files\[0\]\.name\)\) \{/,
  'drops should reserve the restore fast-path for ply/sog; zips sniff contents');

console.log('capture-zip intake (photos + poses in, known-pose training): passed');
