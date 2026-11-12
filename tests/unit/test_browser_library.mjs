import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const appSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');
const storeSource = readFileSync(new URL('../../browser/app/js/store.js', import.meta.url), 'utf8');

// ---- the shelf is the vendored store --------------------------------------
for (const fn of ['saveRun', 'patchRun', 'listRuns', 'deleteRun',
  'saveLastCapture', 'loadLastCapture', 'deleteLastCapture']) {
  assert.match(storeSource, new RegExp(`export async function ${fn}\\(`),
    `store.js keeps its ${fn}`);
}
assert.match(appSource,
  /import \{ saveRun, patchRun, listRuns, deleteRun,[\s\S]*?saveLastCapture, loadLastCapture, deleteLastCapture \} from '\.\/app\/js\/store\.js';/,
  'the browser imports the run shelf from the vendored store');
assert.match(appSource, /continuedBuf, buildReconJson, buildSessionZip,[\s\S]*?convertSHDegree \} from '\.\/app\/js\/session_io\.js';/,
  'finished runs keep a recon built by the shared builder');

// ---- wiring pins: records open, persist, secure ---------------------------
assert.equal((appSource.match(/saveRunStart\(files\[0\]\);/g) || []).length, 2,
  'fresh and continued runs both open a record at start');
assert.match(appSource, /S\.runId = null; S\.loadedFiles = \[\]; S\.lastPersist = 0;/,
  'a fresh staging starts record-less');
assert.match(appSource, /S\.loadedFiles = files\.map\(\(f, i\) => \(\{\s*name: f\.name, url: \(urls && urls\[i\]\) \|\| null\s*\}\)\);/,
  'continuations keep the photo source list (URLs where they exist)');
assert.match(appSource,
  /S\.runId && performance\.now\(\) - \(S\.lastPersist \|\| 0\) > 30000/,
  'long runs patch progress every 30 s, never paced by storage');
assert.match(appSource, /setStatus\('Paused — resume anytime; nothing is lost\.'\);\n    persistRunDetails\(\);/,
  'pausing secures the progress line');
assert.match(appSource, /document\.addEventListener\('visibilitychange', \(\) => \{/,
  'a hidden tab secures progress before it can die');
assert.match(appSource, /\n  secureFinishedRun\(\);/,
  'finishing secures the run');
assert.match(appSource, /status: 'finished',[\s\S]*?\.\.\.\(ply \? \{ ply \} : \{\}\), recon,/,
  'the finished record keeps model + recon');
assert.match(appSource, /thumbFromViewer\(\)/,
  'the finished tile shows the trained result, not a photo');
assert.match(appSource, /gaussians = parsePlyGaussians\(new Uint8Array\(await rec\.ply\.arrayBuffer\(\)\)\);/,
  'shelf tiles reopen from their saved model');
assert.match(appSource, /restoreRunDetails\(rec\);/,
  'reopened runs get their curve memory back');
assert.match(appSource, /await deleteRun\(rec\.id\)/,
  'tiles delete');
assert.match(appSource, /: \(rec\.status === 'training' \? 'interrupted' :/,
  'a training record with no live session reads as interrupted');
assert.equal((appSource.match(/rememberCapture\('/g) || []).length, 3,
  'photo, video, and zip intakes all remember the capture');
assert.match(appSource, /stageFiles\(rec\.files\.map\(\(\{ name, blob \}\) => new File\(\[blob\], name/,
  'the last capture restages from storage');
assert.match(appSource, /ui\.lastcapForget\.addEventListener\('click', async \(\) => \{/,
  'the last capture can be forgotten');
assert.match(appSource, /\nrefreshLastCapture\(\);\nrenderLibrary\(\)\.catch\(\(\) => \{\}\);/,
  'the shelf and the last set load with the page, GPU or not');

// ---- boot the UI module for the exported shelf helpers ----
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
const { runNameFor, describeRun } = await import('../../browser/app.js');

// ---- pure shelf text --------------------------------------------------------
assert.equal(runNameFor({ kind: 'photos', images: [1, 2, 3] }), '3 photos');
assert.equal(runNameFor({ kind: 'zip', file: { name: 'walk.zip' } }), 'walk.zip');
assert.equal(runNameFor({ kind: 'model-url', url: 'https://x/y/model.sog?f=1' }), 'model.sog');
assert.equal(runNameFor(null), 'run');
assert.equal(
  describeRun({ status: 'finished', iter: 30000, maxIters: 30000, splats: 250000, psnr: 28.5, minutes: 12 }),
  'finished · 30,000 / 30,000 · 250,000 splats · 28.50 dB · 12 min');
assert.match(describeRun({ status: 'training', iter: 5 }), /^interrupted/,
  'a dead training record reads as interrupted');
assert.match(describeRun({ status: 'training', iter: 5 }, true), /^training/,
  'the live run still reads as training');

console.log('browser run library + last capture: passed');
