import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { zipStore } from '../../browser/app/js/zip.js';
import { readZip } from '../../browser/zip.js';

const appSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');

// ---- wiring pins: model intake, restore, continue ---------------------------
assert.match(appSource,
  /gravityUpForKnownRun, decodeModel, parseState, parsePlyGaussians,[\s\S]*?sogToGaussians, continuedInputDim, continuedBuf, buildReconJson, buildSessionZip,[\s\S]*?convertSHDegree \} from '\.\/app\/js\/session_io\.js';/,
  'restore/continue come from the vendored session_io module');
assert.match(appSource, /buildLODPlan, undistortFrames,/,
  'continuations undistort against the original run');
assert.match(appSource, /function trainWorkerOpt\(\) \{/,
  'the worker/local decision mirrors the full app (phones train locally)');
assert.equal((appSource.match(/trainWorker: trainWorkerOpt\(\)/g) || []).length, 3,
  'fresh, restore, and continue sessions all place the training core');
assert.match(appSource, /async function modelFromEntries\(entries\) \{/,
  'zip entries dispatch to checkpoint / SOG / PLY');
assert.match(appSource, /const st = parseState\(byBase\.get\('state\.bin'\)\);/,
  'the checkpoint wins when present (bit-exact)');
assert.match(appSource, /gaussians = await sogToGaussians\(byBase\.get\('model\.sog'\)\);/,
  'a packed SOG decodes without one');
assert.match(appSource, /gaussians = parsePlyGaussians\(byBase\.get\('model\.ply'\)\);/,
  'a packed PLY parses as the last resort');
assert.match(appSource, /nothing loadable in the model file/,
  'a model-less archive fails honestly');
assert.match(appSource, /S\.staged\.kind === 'model'/, 'loose .ply/.sog stage as models');
assert.match(appSource, /inputs\.restoreEntries\) return restoreFromEntries\(/,
  'session zips restore instead of training');
assert.match(appSource, /seedFrom\(gaussians, \{\n    viewOnly: true,/,
  'restores seed view-only');
assert.match(appSource, /trainer: \{ maxSplats: gaussians\.n, capMult: 1 \},/,
  'the viewer allocates exactly the restored size');
assert.match(appSource, /continuedBuf\(recon\.trainScale, undefined, prevDim\)/,
  'continuations replay the recorded buffer');
assert.match(appSource, /continuedInputDim\(prevDim, prevBuf\)/,
  'continuations divide the buffer back out of the stored size');
assert.match(appSource, /\.\.\.\(!hasState && wasBaked \? \{ unbake: true \} : \{\}\)/,
  'exported models unbake opacities; checkpoints never do');
assert.match(appSource, /seedFrom\(g, \{ iter: baseIter, sceneRadius/,
  'continuations resume at the checkpoint iteration');
assert.match(appSource, /matchPhotosByName\(ui\.photosFile\.files, names\)/,
  'URL-less exports match dropped photos by name');
assert.match(appSource, /Still missing \$\{missing\.length\} photo/,
  'missing source photos name the gap');
assert.match(appSource, /ui\.cont\.addEventListener\('click', continueClick\);/,
  'the continue button is wired');
assert.match(appSource, /q\.get\('model'\)/, '?model= opens straight in the viewer');

// ---- boot the UI module for the exported restore helpers ----
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
const { modelFromEntries, matchPhotosByName } = await import('../../browser/app.js');

// ---- a synthetic phone session zip restores through browser modules ----
const enc = new TextEncoder();
const I9 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
function stateBin(n, shK, iter, recipe) {
  const head = enc.encode(JSON.stringify({ magic: 'splatjs-state', version: 1,
    n, shK, iter, dc: 'sigmoid', engine: 'v1', recipe }));
  const out = new Uint8Array(4 + head.length + n * 16 * 4 + n * shK * 3 * 4);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  const params = new Float32Array(n * 16).fill(0.1);
  out.set(new Uint8Array(params.buffer), 4 + head.length);
  return out;
}
const recon = {
  version: 1, knownCameras: false, trainScale: 1, iter: 100, splats: 2, shK: 0,
  sceneRadius: 5, center: [0, 0, 0], k1: 0, k2: 0, fFeat: 1000,
  cams: [{ imgIdx: 0, name: 'f0.jpg', R: I9, t: [0, 0, 0], f: 1000, cx: 500, cy: 500 }],
  frames: [{ name: 'f0.jpg', fw: 1000, fh: 1000, tw: 500, th: 500 }],
  source: { preset: null, names: ['f0.jpg'], urls: [null] },
  stats: { minutes: 1, psnrTrain: 20, psnrHold: 18, chart: [[100, 20]], holds: [] },
  cloud: { xyz: [], rgb: [] },
};
{
  const blob = zipStore([
    { name: 'recon.json', data: enc.encode(JSON.stringify(recon)) },
    { name: 'state.bin', data: stateBin(2, 0, 100, { growRate: 0.05 }) },
  ]);
  const entries = await readZip(new Uint8Array(await blob.arrayBuffer()));
  const got = await modelFromEntries(entries);
  assert.equal(got.gaussians.n, 2, 'checkpoint splats decode');
  assert.equal(got.gaussians.data.length, 2 * 16, 'params land in trainer layout');
  assert.equal(got.reconJson.cams.length, 1, 'cameras restore');
  assert.equal(got.state.iter, 100, 'the checkpoint iteration restores');
  assert.equal(got.state.engine, 'v1', 'the writing engine restores');
  assert.equal(got.state.recipe.growRate, 0.05, 'the trainer recipe restores');
}

// ---- a packed PLY restores with cameras but no checkpoint ----
function tinyPly() {
  const props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const head = `ply\nformat binary_little_endian 1.0\nelement vertex 1\n` +
    props.map((p) => `property float ${p}`).join('\n') + `\nend_header\n`;
  const hb = enc.encode(head);
  const out = new Uint8Array(hb.length + props.length * 4);
  out.set(hb, 0);
  const body = new Float32Array(props.length).fill(0.25);
  out.set(new Uint8Array(body.buffer), hb.length);
  return out;
}
{
  const got = await modelFromEntries(new Map([
    ['model.ply', tinyPly()],
    ['nested/recon.json', enc.encode(JSON.stringify(recon))],
  ]));
  assert.equal(got.gaussians.n, 1, 'packed PLY parses');
  assert.equal(got.gaussians.dc, 'sh', 'PLY colors keep the standard SH-DC tag');
  assert.equal(got.reconJson.frames.length, 1, 'nested recon.json still found');
  assert.equal(got.state, null, 'no checkpoint means no state');
}
await assert.rejects(modelFromEntries(new Map([['notes.txt', new Uint8Array([1])]])),
  /nothing loadable/, 'a model-less archive fails honestly');

// ---- photo matching follows the run's own frame order ----
{
  const names = ['b.jpg', 'a.jpg'];
  const fb = new File([new Uint8Array([1])], 'b.jpg', { type: 'image/jpeg' });
  const fa = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' });
  const ok = matchPhotosByName([fa, fb], names);
  assert.deepEqual(ok.files, [fb, fa], 'matched photos follow recon order, not drop order');
  const miss = matchPhotosByName([fa], names);
  assert.deepEqual(miss.missing, ['b.jpg'], 'missing photos are named');
}

console.log('browser session restore + continue intake: passed');
