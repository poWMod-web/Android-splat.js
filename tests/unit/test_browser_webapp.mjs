import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'browser');
const read = (f) => readFileSync(join(dir, f), 'utf8');

// 1. the folder ships the whole UI: a generated entry plus the vendored app
// and engine (byte-identity pinned by test_browser_vendor.mjs)
for (const f of ['index.html', 'README.md', 'app/js/app.js', 'app/js/store.js',
    'app/js/chart.js', 'app/js/session_io.js', 'app/js/zip.js', 'app/css/app.css',
    'app/css/xr.css', 'src/index.js', 'app/models/splat-transform.bundle.mjs',
    'app/vendor/playcanvas.min.mjs', 'app/vendor/st-worker.mjs', 'app/vendor/webp.wasm']) {
  assert.ok(existsSync(join(dir, f)), `browser/${f} exists`);
}
// the fork is gone: no parallel shell, stylesheet, or zip reader at the
// top level, and no flat mirror dirs either — browser/app/ is the whole app
for (const f of ['app.js', 'styles.css', 'zip.js', 'js', 'css', 'avatar', 'vendor', 'models']) {
  assert.ok(!existsSync(join(dir, f)), `browser/${f} was retired with the fork`);
}

// 2. the generated entry boots the vendored app with the browser marker
const html = read('index.html');
assert.match(html, /<script>window\.__SPLAT_SHELL="browser";<\/script><script type="module" src="app\/js\/app\.js"><\/script>/,
  'the entry should set the shell marker before the app module');
assert.match(html, /<link rel="stylesheet" href="app\/css\/app\.css">/,
  'the entry should load the vendored app stylesheet');
assert.ok(!html.includes('manifest.webmanifest'), 'the browser is not installable');
// same UI, same mount points: the entry owns the ids the app boots against
for (const id of ['brand', 'btn-new', 'start', 'detail', 'detail-back',
    'btn-go', 'settings', 'dock', 'strip', 'details', 'd-chart', 'about',
    'btn-ar', 'btn-record', 'gallery', 'file-input']) {
  assert.ok(html.includes(`id="${id}"`), `index.html owns #${id}`);
}

// 3. the README documents serving + scope honestly
const readme = read('README.md');
assert.match(readme, /python3 -m http\.server/, 'serving the folder is documented');
assert.match(readme, /file:\/\//, 'the file:// refusal is documented');
assert.match(readme, /WebGPU/, 'the WebGPU requirement is documented');
assert.match(readme, /\.ply/, 'the .ply result is documented');
assert.match(readme, /same UI|same app/, 'the shared-UI scope is documented');
assert.match(readme, /local-only|local only/, 'the local-only profile is documented');

console.log('browser shell structure + scope: passed');
