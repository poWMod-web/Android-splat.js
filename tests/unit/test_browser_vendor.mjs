import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const browser = join(root, 'browser');

// Shared files are byte-identical. Viewer/background boundary files are
// Browser-owned and deliberately survive sync instead of following Android.
const walk = (dir) => {
  const out = [];
  const rec = (d) => {
    for (const e of readdirSync(d).sort()) {
      const p = join(d, e);
      if (statSync(p).isDirectory()) rec(p);
      else out.push(p);
    }
  };
  rec(dir);
  return out;
};
const DIRS = [['src', 'src'], ['app/css', 'app/css'], ['app/avatar', 'app/avatar']];
const BROWSER_OWNED_JS = new Set(['app.js', 'pcview.js', 'headless.js']);
const ANDROID_ONLY_JS = new Set(['train-headless.js']);
const pairs = [];
for (const [s, d] of DIRS) {
  const srcBase = join(root, s);
  for (const p of walk(srcBase)) pairs.push([p, join(browser, d, p.slice(srcBase.length + 1))]);
}
const appJs = join(root, 'app', 'js');
for (const src of walk(appJs)) {
  const name = src.slice(appJs.length + 1);
  if (!BROWSER_OWNED_JS.has(name) && !ANDROID_ONLY_JS.has(name))
    pairs.push([src, join(browser, 'app', 'js', name)]);
}
for (const f of ['playcanvas.min.mjs', 'st-worker.mjs', 'webp.wasm']) {
  pairs.push([join(root, 'app', 'vendor', f), join(browser, 'app', 'vendor', f)]);
}
pairs.push([join(root, 'app', 'models', 'splat-transform.bundle.mjs'),
  join(browser, 'app', 'models', 'splat-transform.bundle.mjs')]);
assert.ok(pairs.length > 75, `the mirror should hold the engine + shared app (saw ${pairs.length})`);
for (const [src, dst] of pairs) {
  assert.ok(existsSync(dst), `${dst.slice(root.length + 1)} is vendored`);
  assert.ok(readFileSync(src).equals(readFileSync(dst)),
    `${dst.slice(root.length + 1)} drifted from its source — run sh tools/sync-browser.sh`);
}
for (const f of BROWSER_OWNED_JS)
  assert.ok(existsSync(join(browser, 'app', 'js', f)), `Browser-owned app/js/${f} exists`);
assert.ok(!existsSync(join(browser, 'app', 'js', 'train-headless.js')),
  'the Android background-service driver must not be copied into Browser');
const syncScript = readFileSync(join(root, 'tools', 'sync-browser.sh'), 'utf8');
assert.match(syncScript, /BROWSER_OWNED_JS="app\.js pcview\.js headless\.js"/,
  'sync explicitly protects Browser viewer/background boundaries');
assert.match(syncScript, /ANDROID_ONLY_JS="train-headless\.js"/,
  'sync explicitly omits the Android service driver');
const browserPcview = readFileSync(join(browser, 'app', 'js', 'pcview.js'), 'utf8');
assert.doesNotMatch(browserPcview, /yUp/,
  'Browser pcview should retain arrival-space coordinate behavior, not Android’s y-up branch');
// No stale files: shared directories match exactly; Browser JS equals source
// names minus Android-only files (Browser-owned names replace their sources).
for (const [s, d] of DIRS) {
  const inMirror = walk(join(browser, d)).map((p) => p.slice(join(browser, d).length + 1)).sort();
  const inSrc = walk(join(root, s)).map((p) => p.slice(join(root, s).length + 1)).sort();
  assert.deepEqual(inMirror, inSrc, `browser/${d}/ lists exactly ${s}/`);
}
const browserJsNames = walk(join(browser, 'app', 'js')).map((p) => p.slice(join(browser, 'app', 'js').length + 1)).sort();
const expectedJsNames = walk(appJs).map((p) => p.slice(appJs.length + 1))
  .filter((f) => !ANDROID_ONLY_JS.has(f)).sort();
assert.deepEqual(browserJsNames, expectedJsNames, 'Browser JS excludes only Android service drivers');
// The avatar ML models stay out; the SOG codec is the only models/ file.
assert.deepEqual(readdirSync(join(browser, 'app', 'models')).sort(), ['splat-transform.bundle.mjs'],
  'browser/app/models/ carries only the SOG codec');

// the generated entry: app/index.html minus install bits plus the shell marker
const entry = readFileSync(join(browser, 'index.html'), 'utf8');
const appHtml = readFileSync(join(root, 'app', 'index.html'), 'utf8');
assert.ok(entry.includes('<script>window.__SPLAT_SHELL="browser";</script><script type="module" src="app/js/app.js"></script>'),
  'the generated entry should set the shell marker before booting the app');
assert.ok(!entry.includes('manifest.webmanifest') && !entry.includes('icon-192') && !entry.includes('icon-180'),
  'the generated entry should drop the install bits');
const norm = (s) => s.split('\n').filter((l) => !/manifest|icon-192|icon-180/.test(l)).join('\n')
  .replace('<script>window.__SPLAT_SHELL="browser";</script>', '')
  .replaceAll('src="app/js/', 'src="js/').replaceAll('href="app/css/', 'href="css/');
assert.equal(norm(entry), norm(appHtml),
  'browser/index.html should be app/index.html plus the marker — regenerate, never hand-edit');

// every mirrored import resolves INSIDE browser/ (the folder serves standalone)
const jsFiles = [...new Set([...pairs.map(([, dst]) => dst).filter((p) => p.endsWith('.js')),
  ...walk(join(browser, 'app', 'js')).filter((p) => p.endsWith('.js'))])];
const importRe = /(?:from\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\))/g;
for (const f of jsFiles) {
  // comments lie (e.g. "from 'splat.js'" in prose) — strip them; no mirrored
  // import specifier contains //, so the naive pass can't eat a real one
  const src = readFileSync(f, 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|\s)\/\/.*$/gm, '$1');
  let m;
  while ((m = importRe.exec(src))) {
    const spec = m[1] || m[2];
    assert.ok(spec.startsWith('.'), `${f} must not use bare import ${spec}`);
    const target = resolve(dirname(f), spec);
    assert.ok(target === browser || target.startsWith(browser + sep),
      `${f} escapes the browser folder via ${spec}`);
    assert.ok(existsSync(target), `${f} imports ${spec}, which is missing from the mirror`);
  }
}

console.log(`browser shared mirror (${pairs.length} shared files) + protected viewer/background boundaries: passed`);
