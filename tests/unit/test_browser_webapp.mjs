import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'browser');
const read = (f) => readFileSync(join(dir, f), 'utf8');

// 1. the folder ships the whole webapp: five native files plus the vendored
// engine + app modules (byte-identity pinned by test_browser_vendor.mjs)
for (const f of ['index.html', 'app.js', 'zip.js', 'styles.css', 'README.md']) {
  assert.ok(existsSync(join(dir, f)), `browser/${f} exists`);
}
for (const f of ['src/index.js', 'app/js/session_io.js', 'app/js/xr-dataset.js',
    'app/js/store.js', 'app/js/chart.js', 'app/models/splat-transform.bundle.mjs',
    'app/vendor/st-worker.mjs', 'app/vendor/webp.wasm']) {
  assert.ok(existsSync(join(dir, f)), `browser/${f} is vendored`);
}

// 2. index.html wires the module + stylesheet and owns every id app.js needs
const html = read('index.html');
assert.match(html, /<script type="module" src="\.\/app\.js"><\/script>/);
assert.match(html, /<link rel="stylesheet" href="\.\/styles\.css">/);
assert.match(html, /<input id="files" type="file" multiple/);
assert.match(html, /<canvas id="view"><\/canvas>/);
for (const id of ['nogl', 'drop', 'opt-q', 'opt-res', 'opt-feat', 'opt-tier',
    'opt-buf', 'opt-sh', 'opt-splats', 'opt-lod', 'opt-lod-count', 'opt-mcmc',
    'opt-iters', 'opt-gpu', 'opt-gpu-v', 'opt-gap', 'input-summary',
    'btn-start', 'run', 'stage-label', 'stage-bar', 'metrics', 'btn-pause',
    'btn-finish', 'btn-ply', 'btn-continue', 'needphotos', 'photos-hint',
    'photos-file', 'lodlinks', 'status', 'viewer', 'btn-recenter',
    'logwrap', 'log', 'chart', 'btn-scale', 'details', 'tab-score', 'tab-perf',
    'tab-cams', 'tab-gpu', 'detail-body', 'lastcap', 'lastcap-text', 'lastcap-use',
    'lastcap-forget', 'library', 'runs']) {
  assert.ok(html.includes(`id="${id}"`), `index.html owns #${id}`);
}

// 3. app.js drives the vendored engine and nothing outside the folder
const app = read('app.js');
assert.match(app, /from '\.\/src\/index\.js'/, 'engine surface comes from the vendored ./src/index.js');
assert.match(app, /createSession.*parseCameras.*extractSharpFrames.*isVideoFile.*solveTierOpts.*buildLODPlan/s);
assert.match(app, /from '\.\/zip\.js'/, 'zip intake is the local DOM-free reader');
assert.ok(!/from '\.\.\//.test(app) && !/import\('\.\.\//.test(app),
  'no imports escape the browser folder (standalone-servable)');
assert.ok(!/from ['"][^.'"/][^'"]*['"]/.test(app.replace(/from '\.\//g, '')),
  'no bare-package imports — relative only, no build step, no import map');

// 4. the three intakes + the full pipeline are present
assert.match(app, /readZip\(new Uint8Array\(await .*\.arrayBuffer\(\)\)\)/, 'zip files are unpacked');
assert.match(app, /recon\.json.*state\.bin.*model\.sog/s, 'session exports are detected and explained');
assert.match(app, /extractSharpFrames\(file, \{/, 'video goes through the sharp-frame extractor');
assert.match(app, /targetGapSec/, 'video spacing control reaches the extractor');
assert.match(app, /pairMinInliers: 100, pairMinInliersAdj: 15/, 'video frames get the neighbour-pair gates');
assert.match(app, /readVideoMeta/, 'container lens data becomes the focal prior');
assert.match(app, /useKnownCameras\(parseCameras\(capture\.transforms\)/, 'known poses skip the solve');
assert.match(app, /await session\.solve\(\)/, 'plain photos are solved');
assert.match(app, /await session\.seed\(\)/, 'the run is seeded');
assert.match(app, /train-complete/, 'completion is handled');
assert.match(app, /await s\.exportPlyBlob\(\)/, 'the result downloads as .ply');

// 4b. the full app's training knobs reach the session identically
assert.match(app, /trainMaxDim/, 'training resolution caps the input scale');
assert.match(app, /trainScale: st\.buf/, 'the working buffer scales the supervision grid');
assert.match(app, /featMaxDim/, 'solve resolution reaches the solver');
assert.match(app, /trainerOpts\.shDeg = st\.sh/, 'the SH degree is always explicit');
assert.match(app, /mcmcNoise: true/, 'the MCMC set matches the full app');
assert.match(app, /refineEvery: 500/, 'MCMC runs refine on its own cadence');
assert.match(app, /gpuUsage: st\.gpu \/ 100/, 'the GPU slider becomes the duty cycle');
assert.match(app, /S\.session\.opts\.gpuUsage = /, 'the GPU slider stays live mid-run');
assert.match(app, /QMACROS = \{/, 'the quality macro drives the rows');
assert.match(app, /buildLODPlan\(st\.splats, st\.lodLevels/, 'LOD plans come from the shared builder');
assert.match(app, /growLimit = .*levels\[0\]/, 'LOD runs start capped at the first level');
assert.match(app, /LP\.snaps\.push\(/, 'LOD levels snapshot as they complete');
assert.match(app, /splat-lod-/, 'LOD snapshots download as their own .ply files');

// 5. the preview is a self-contained orbit (no app viewport dependency)
assert.match(app, /session\.view\.attach\(cv\)/, 'the session view attaches to the canvas');
assert.match(app, /s\.view\.setCamera\(\{ R, t, f: o\.f/, 'orbit drags drive a free camera');
assert.match(app, /acc\[0\] -= c\.R\[3\]/, 'orbit up is the dominant camera up');

// 6. browser-only: the NATIVE files carry no device, capture-hardware, or
// native-bridge references. (The vendored mirrors are byte-copies of reviewed
// app sources — inert comments, format strings, and guarded fallbacks only;
// test_browser_vendor.mjs pins the byte-identity so no live device dep can
// sneak in without also changing the app source.)
// 'arcore' is NOT banned: the browser ingests the ARCore capture export
// format (photos + poses + pointcloud zip) — the word names an accepted
// input, not a hardware capability. Live bridges and capture APIs stay out.
const banned = ['android', 'webxr', 'xrdevice', 'xr-device', 'splatxr',
  'androidbridge', 'nativebridge', 'appassets', 'capacitor', 'cordova',
  'androidarbridge', 'getusermedia', 'mediarecorder'];
for (const f of ['index.html', 'app.js', 'zip.js', 'styles.css']) {
  const lower = read(f).toLowerCase();
  for (const b of banned) assert.ok(!lower.includes(b), `browser/${f} must not mention ${b}`);
}

// 7. zip.js stays DOM-free so Node tests it directly
const zip = read('zip.js');
assert.ok(!/\b(document|window|navigator|localStorage|indexedDB)\b/.test(zip), 'zip.js has no DOM access');
assert.match(zip, /export async function readZip/);

// 8. the README documents serving + scope honestly
const readme = read('README.md');
assert.match(readme, /python3 -m http\.server/, 'serving from the repo root is documented');
assert.match(readme, /file:\/\//, 'the file:// refusal is documented');
assert.match(readme, /WebGPU/, 'the WebGPU requirement is documented');
assert.match(readme, /\.ply/, 'the .ply result is documented');

console.log('browser webapp structure + scope: passed');
