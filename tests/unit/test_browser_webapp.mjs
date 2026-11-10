import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'browser');
const read = (f) => readFileSync(join(dir, f), 'utf8');

// 1. the folder ships the whole webapp
for (const f of ['index.html', 'app.js', 'zip.js', 'styles.css', 'README.md']) {
  assert.ok(existsSync(join(dir, f)), `browser/${f} exists`);
}

// 2. index.html wires the module + stylesheet and owns every id app.js needs
const html = read('index.html');
assert.match(html, /<script type="module" src="\.\/app\.js"><\/script>/);
assert.match(html, /<link rel="stylesheet" href="\.\/styles\.css">/);
assert.match(html, /<input id="files" type="file" multiple/);
assert.match(html, /<canvas id="view"><\/canvas>/);
for (const id of ['nogl', 'drop', 'opt-tier', 'opt-iters', 'opt-gap', 'input-summary',
    'btn-start', 'run', 'stage-label', 'stage-bar', 'metrics', 'btn-pause',
    'btn-finish', 'btn-ply', 'status', 'viewer', 'btn-recenter', 'logwrap', 'log']) {
  assert.ok(html.includes(`id="${id}"`), `index.html owns #${id}`);
}

// 3. app.js drives the shared engine and nothing else
const app = read('app.js');
assert.match(app, /from '\.\.\/src\/index\.js'/, 'engine surface comes from ../src/index.js');
assert.match(app, /createSession.*parseCameras.*extractSharpFrames.*isVideoFile.*solveTierOpts/s);
assert.match(app, /from '\.\/zip\.js'/, 'zip intake is the local DOM-free reader');
assert.ok(!/from '\.\.\/app\//.test(app), 'no imports from the full app (it carries device UI)');
assert.ok(!/from ['"][^.'"/][^'"]*['"]/.test(app.replace(/from '\.\.\//g, '').replace(/from '\.\//g, '')),
  'no bare-package imports — relative only, no build step, no import map');

// 4. the three intakes + the full pipeline are present
assert.match(app, /readZip\(new Uint8Array\(await .*\.arrayBuffer\(\)\)\)/, 'zip files are unpacked');
assert.match(app, /recon\.json.*state\.bin.*model\.sog/s, 'session exports are detected and explained');
assert.match(app, /extractSharpFrames\(file, \{/, 'video goes through the sharp-frame extractor');
assert.match(app, /targetGapSec/, 'video spacing control reaches the extractor');
assert.match(app, /pairMinInliers: 100, pairMinInliersAdj: 15/, 'video frames get the neighbour-pair gates');
assert.match(app, /readVideoMeta/, 'container lens data becomes the focal prior');
assert.match(app, /useKnownCameras\(parseCameras\(knownJson\)/, 'known poses skip the solve');
assert.match(app, /await session\.solve\(\)/, 'plain photos are solved');
assert.match(app, /await session\.seed\(\)/, 'the run is seeded');
assert.match(app, /train-complete/, 'completion is handled');
assert.match(app, /await s\.exportPlyBlob\(\)/, 'the result downloads as .ply');

// 5. the preview is a self-contained orbit (no app viewport dependency)
assert.match(app, /session\.view\.attach\(cv\)/, 'the session view attaches to the canvas');
assert.match(app, /s\.view\.setCamera\(\{ R, t, f: o\.f/, 'orbit drags drive a free camera');
assert.match(app, /acc\[0\] -= c\.R\[3\]/, 'orbit up is the dominant camera up');

// 6. browser-only: no device, capture-hardware, or native-bridge references
const banned = ['arcore', 'android', 'webxr', 'xrdevice', 'xr-device', 'splatxr',
  'androidbridge', 'nativebridge', 'appassets', 'capacitor', 'cordova'];
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
