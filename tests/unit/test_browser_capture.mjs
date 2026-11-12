import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateDataset, measuredSeeds, levelingNote } from '../../browser/app/js/xr-dataset.js';
import { gravityUpForKnownRun } from '../../browser/app/js/session_io.js';
import { parseCameras } from '../../browser/src/io/cameras.js';
import { zipStore } from '../../browser/app/js/zip.js';
import { readZip } from '../../browser/zip.js';

const appSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- wiring pins: the ARCore path goes through the vendored validators ----
assert.match(appSource,
  /import \{ validateDataset, measuredSeeds, levelingNote \} from '\.\/app\/js\/xr-dataset\.js';/,
  'capture validation comes from the vendored xr-dataset module');
assert.match(appSource,
  /import \{ gravityUpForKnownRun,[\s\S]*?\} from '\.\/app\/js\/session_io\.js';/,
  'known-run gravity comes from the vendored session_io module');
assert.match(appSource, /function assembleCapture\(files, jsons\) \{/,
  'loose/zip JSONs assemble into the app capture shape');
assert.match(appSource, /return validateDataset\(\{/,
  'assembled captures validate exactly like the phone path');
assert.match(appSource, /seedCount: phone \? 8000 : 20000/,
  'known-camera runs seed the app-sized random fallback');
assert.match(appSource, /Some images lost their known cameras during decoding/,
  'a known run refuses to substitute estimated poses');
assert.match(appSource, /levelingNote\(capture\.metadata/,
  'the capture leveling report reaches the log');
assert.match(appSource, /gravityUpForKnownRun\(\{ persisted: cap\.levelingUp \|\| null,[\s\S]*?leveling: cap\.metadata/,
  'known runs level the orbit to capture gravity, not sensor-up');
assert.match(appSource, /Use at least 3 overlapping views/,
  'intake enforces the multi-view minimum');
assert.match(appSource, /await sortByCapture\(staged\.images\)/,
  'loose picks restore the capture sequence');
assert.match(appSource, /function dominantUp\(cams\) \{/,
  'solver runs keep the dominant-camera-up orbit');

// ---- boot the UI module in Node to drive the exported intake helpers ----
const els = new Map();
function mkEl(id, defaults = {}) {
  const el = {
    id, value: '', textContent: '', disabled: false, hidden: false,
    style: {}, scrollTop: 0, scrollHeight: 0,
    classList: { add() {}, remove() {} },
    addEventListener() {},
    querySelectorAll: () => [],
    replaceChildren() {},
    append() {},
    ...defaults,
  };
  els.set(id, el);
  return el;
}
const DEFAULTS = { 'opt-q': 'standard', 'opt-res': '', 'opt-feat': '', 'opt-tier': 'standard',
  'opt-buf': '1', 'opt-sh': '3', 'opt-splats': '', 'opt-lod': '', 'opt-lod-count': '0',
  'opt-mcmc': '', 'opt-iters': '30000', 'opt-gpu': '100', 'opt-gap': '0' };
globalThis.document = {
  getElementById: (id) => els.get(id) || mkEl(id, { value: DEFAULTS[id] ?? '' }),
  createElement: () => mkEl('x'),
};
Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
const { assembleCapture, sortByCapture } = await import('../../browser/app.js');

// ---- a synthetic ARCore capture zip, end to end through browser modules ----
const I4 = (x = 0) => [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const NAMES = ['f0.jpg', 'f1.jpg', 'f2.jpg'];
const transforms = {
  fl_x: 1000, fl_y: 1000, cx: 500, cy: 500, w: 1000, h: 1000,
  frames: NAMES.map((file_path, i) => ({ file_path, transform_matrix: I4(i * 0.5) })),
};
const seedDoc = { points: Array.from({ length: 10 }, (_, i) => ({ X: [i * 0.1, 0, 0], rgb: [0.5, 0.5, 0.5] })) };
const captureDoc = {
  frames: 3,
  leveling: { method: 'arcore-horizontal-plane', applied: true, tiltDegrees: 2,
    planeArea: 1.5, rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1] },
};
const enc = new TextEncoder();
{
  const photo = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]); // names validate; pixels decode later
  const blob = zipStore([
    ...NAMES.map((name) => ({ name, data: photo })),
    { name: 'transforms.json', data: enc.encode(JSON.stringify(transforms)) },
    { name: 'seed_points.json', data: enc.encode(JSON.stringify(seedDoc)) },
    { name: 'capture.json', data: enc.encode(JSON.stringify(captureDoc)) },
    { name: 'README.txt', data: enc.encode('SplatXR capture dataset') },
  ]);
  const entries = await readZip(new Uint8Array(await blob.arrayBuffer()));
  assert.equal(entries.size, 7, 'the whole capture export unpacks');
  const base = (p) => String(p).split(/[\\/]/).pop();
  const images = [];
  const jsons = [];
  for (const [name, bytes] of entries) {
    const b = base(name);
    if (/\.jpe?g$/i.test(b)) images.push(new File([bytes], b, { type: 'image/jpeg' }));
    else if (/\.json$/i.test(b)) jsons.push({ name: b, json: JSON.parse(new TextDecoder().decode(bytes)) });
  }
  const capture = assembleCapture(images, jsons);
  assert.equal(capture.files.length, 3);
  assert.deepEqual(measuredSeeds(capture), seedDoc.points, '10 measured seeds survive assembly');
  assert.match(levelingNote(capture.metadata.leveling), /ARCore ground plane/,
    'the leveling report reads as the phone wrote it');
  assert.deepEqual(gravityUpForKnownRun({ leveling: capture.metadata.leveling }), [0, 1, 0],
    'identity leveling keeps world +Y as orbit up');
  const cams = parseCameras(capture.transforms);
  assert.equal(cams.length, 3, 'three poses parse');
  for (const c of cams) {
    assert.ok(c.R.every(Number.isFinite) && c.t.every(Number.isFinite), 'poses are finite');
    assert.ok(Number.isFinite(c.fx) && c.fx > 0, 'focal present');
  }
  assert.deepEqual(validateDataset({ ...capture }), capture, 'validateDataset is idempotent');
}

// ---- the rejections, with the app's own messages ----
const files3 = NAMES.map((name) => new File([new Uint8Array([1])], name, { type: 'image/jpeg' }));
const J = (transforms, points = seedDoc.points, metadata = captureDoc) => ([
  { name: 'transforms.json', json: transforms },
  { name: 'seed_points.json', json: { points } },
  { name: 'capture.json', json: metadata },
]);
assert.throws(() => assembleCapture(files3.slice(0, 2), J(transforms)), /at least 3/,
  'fewer than 3 views are refused');
assert.throws(() => assembleCapture(
  [new File([new Uint8Array([1])], 'D.jpg'),
    new File([new Uint8Array([1])], 'd.jpg'), files3[0]],
  J(transforms)), /unique/,
  'case-insensitive duplicate basenames are refused');
assert.throws(() => assembleCapture(files3,
  J({ ...transforms, frames: transforms.frames.slice(0, 2) })), /No known camera for f2/,
  'a photo without its pose names the photo');
assert.throws(() => assembleCapture(files3,
  J({ ...transforms, frames: transforms.frames.map((f, i) => i ? f : { ...f, skew: 0.01 }) })),
  /skew/, 'nonzero projection skew is refused, not silently trained');
assert.throws(() => assembleCapture(files3,
  J(transforms, [{ X: [0, 0, 0], rgb: [2, 0, 0] }])), /Seed points/,
  'out-of-range seed colors are refused');
assert.throws(() => assembleCapture(files3,
  [...J(transforms), { name: 'cameras.json', json: { cameras: [] } }]), /only one camera JSON/,
  'two camera files are refused');
assert.deepEqual(measuredSeeds({ points: seedDoc.points.slice(0, 7) }), [],
  'fewer than 8 seeds fall back to random-volume init');
assert.equal(assembleCapture(files3, [{ name: 'notes.json', json: { a: 1 } }]), null,
  'no camera file means a plain photo set, not an error');

// ---- capture sequence: EXIF time wins, numeric names are the fallback ----
function jpegWithExif(dateStr) {
  const s = enc.encode(`${dateStr}\0`); // 20 bytes: "YYYY:MM:DD HH:MM:SS\0"
  const tiff = new Uint8Array(62);
  const dv = new DataView(tiff.buffer);
  tiff.set([0x49, 0x49], 0); dv.setUint16(2, 42, true); dv.setUint32(4, 8, true);
  dv.setUint16(8, 1, true); // IFD0: one entry -> EXIF IFD at 26
  dv.setUint16(10, 0x8769, true); dv.setUint16(12, 4, true);
  dv.setUint32(14, 1, true); dv.setUint32(18, 26, true);
  dv.setUint16(26, 1, true); // EXIF IFD: DateTimeOriginal -> string at 42
  dv.setUint16(28, 0x9003, true); dv.setUint16(30, 2, true);
  dv.setUint32(32, 20, true); dv.setUint32(36, 42, true);
  tiff.set(s, 42);
  const app1 = new Uint8Array(2 + 6 + tiff.length);
  const av = new DataView(app1.buffer);
  av.setUint16(0, 6 + tiff.length + 2, false);
  app1.set([0x45, 0x78, 0x69, 0x66, 0, 0], 2);
  app1.set(tiff, 8);
  const jpg = new Uint8Array(2 + 2 + app1.length + 2);
  jpg.set([0xff, 0xd8, 0xff, 0xe1], 0);
  jpg.set(app1, 4);
  jpg.set([0xff, 0xd9], 4 + app1.length);
  return jpg;
}
{
  const late = new File([jpegWithExif('2024:03:01 10:00:00')], 'a_first.jpg', { type: 'image/jpeg' });
  const early = new File([jpegWithExif('2024:01:01 10:00:00')], 'z_second.jpg', { type: 'image/jpeg' });
  const files = [late, early];
  await sortByCapture(files);
  assert.deepEqual(files.map((f) => f.name), ['z_second.jpg', 'a_first.jpg'],
    'EXIF time orders the set, not the pick order');
  const noExif = (name) => new File([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])], name, { type: 'image/jpeg' });
  const plain = [noExif('img10.jpg'), noExif('img2.jpg')];
  await sortByCapture(plain);
  assert.deepEqual(plain.map((f) => f.name), ['img2.jpg', 'img10.jpg'],
    'without EXIF anywhere, numeric name order rules');
}

console.log('browser ARCore capture ingest (assemble, validate, level, sequence): passed');
