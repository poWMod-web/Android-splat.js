import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const appSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- wiring pins: tour, strip, session + photos exports --------------------
assert.match(appSource, /import \{ plyToSog \} from '\.\/app\/js\/sog\.js';/,
  'the session export compresses through the vendored SOG codec');
assert.match(appSource, /import \{ zipStore \} from '\.\/app\/js\/zip\.js';/,
  'photo zips use the shared stored-zip writer');
assert.match(appSource, /import \{ saveBlob \} from '\.\/app\/js\/download\.js';/,
  'downloads go through the shared saver (plain anchor, no bridge)');
assert.match(appSource, /function tourPath\(cams\) \{/,
  'the fly-through path is shared path math');
assert.match(appSource, /if \(S\.tour\) stopTour\(\);\n  else startTour\(\);/,
  'the tour button toggles');
assert.match(appSource, /stopTour\(\);   \/\/ a grab takes over from the fly-through/,
  'grabbing the view stops the tour');
assert.match(appSource, /renderStrip\(\);\n  wireOrbit\(\);/,
  'every viewer gets its filmstrip');
assert.match(appSource, /const k = cams\.findIndex\(\(c\) => c\.imgIdx === i\);/,
  'strip taps find the photo\'s own camera by imgIdx');
assert.equal((appSource.match(/setPhotoSources\(files\);/g) || []).length, 2,
  'fresh and continued runs both keep their photo pixels');
assert.match(appSource, /URL\.createObjectURL\(f\.source \|\| f\)/,
  'strip thumbs cover Files and video-frame sources alike');
assert.match(appSource, /if \(!s \|\| S\.restored\) return;   \/\/ the raw state needs its live trainer/,
  'the session export needs a trained run, not a restore');
assert.match(appSource, /const blob = await plyToSog\(new Uint8Array\(await ply\.arrayBuffer\(\)\), \{/,
  'the session packs a freshly compressed SOG');
assert.match(appSource, /const zip = await buildSessionZip\(reconFacade\(\), sog\);/,
  'the session zip reuses the shared builder + the run facade');
assert.match(appSource, /if \(S\.sogBlob\) return S\.sogBlob;/,
  'the compression runs once per run');
assert.match(appSource, /const blob = f\.source \|\| f;/,
  'the photos zip covers Files and video-frame sources alike');
assert.match(appSource, /ui\.ply\.disabled = false;\n  ui\.simplify\.disabled = false;\n  ui\.sess\.disabled = false;\n  setStatus\('Training finished/,
  'finishing unlocks the session export');
assert.match(appSource, /S\.stripUrls = \[\]; S\.photoBlobs = \[\];/,
  'a reset drops the photo pixels');
assert.match(appSource, /S\.sogBlob = null; S\._sogJob = null;/,
  'a reset drops the compressed model');

// ---- boot the UI module for the exported path builder ----
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
if (!globalThis.navigator) {
  Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
}
const { tourPath } = await import('../../browser/app.js');

// ---- the path builder on synthetic walks -----------------------------------
const I9 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const cam = (x, y, z) => ({ R: I9, t: [-x, -y, -z] });   // identity R: centre = -t

assert.equal(tourPath([]), null);
assert.equal(tourPath([cam(0, 0, 0)]), null);
assert.equal(tourPath([cam(0, 0, 0), { t: [0, 0, 0] }]), null,
  'a camera without rotation is not a node');
{
  const line = [0, 1, 2, 3, 4].map((x) => cam(x, 0, 0));
  const p = tourPath(line);
  assert.equal(p.count, 5, 'a straight walk flies whole');
  assert.equal(p.samples.length, (5 - 1) * 8 + 1, '8 subdivisions per span, plus the end');
  assert.equal(p.cum.length, p.samples.length);
  assert.ok(p.cum.every((v, i) => i === 0 || v >= p.cum[i - 1]), 'arc length never runs back');
  assert.ok(p.total > 0 && p.total <= 4, `smoothing shortens a straight line, never past zero (${p.total})`);
  const long = Array(20).fill(0).map((_, x) => cam(x, 0, 0));
  assert.ok(tourPath(long).total > 17, 'a long walk keeps nearly its whole length');
}
{
  const twoWalks = [...[0, 1, 2, 3].map((x) => cam(x, 0, 0)), ...[100, 101, 102].map((x) => cam(x, 0, 0))];
  assert.equal(tourPath(twoWalks).count, 4, 'a break in the capture flies the longest leg');
}
{
  const pano = [...Array(6).fill(0).map(() => cam(0, 0, 0)), cam(1, 0, 0), cam(2, 0, 0)];
  assert.equal(tourPath(pano).count, 3, 'co-located pano orientations collapse to one node');
}

console.log('browser tour + strip + session/photos exports: passed');
