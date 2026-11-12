import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const browser = join(root, 'browser');

// every mirrored file byte-identical to its source (drift = run tools/sync-browser.sh)
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
const DIRS = [['src', 'src'], ['app/js', 'app/js'], ['app/css', 'app/css'], ['app/avatar', 'app/avatar']];
const pairs = [];
for (const [s, d] of DIRS) {
  const srcBase = join(root, s);
  for (const p of walk(srcBase)) pairs.push([p, join(browser, d, p.slice(srcBase.length + 1))]);
}
for (const f of ['playcanvas.min.mjs', 'st-worker.mjs', 'webp.wasm']) {
  pairs.push([join(root, 'app', 'vendor', f), join(browser, 'app', 'vendor', f)]);
}
pairs.push([join(root, 'app', 'models', 'splat-transform.bundle.mjs'),
  join(browser, 'app', 'models', 'splat-transform.bundle.mjs')]);
assert.ok(pairs.length > 80, `the mirror should hold the engine + the whole app (saw ${pairs.length})`);
for (const [src, dst] of pairs) {
  assert.ok(existsSync(dst), `${dst.slice(root.length + 1)} is vendored`);
  assert.ok(readFileSync(src).equals(readFileSync(dst)),
    `${dst.slice(root.length + 1)} drifted from its source — run sh tools/sync-browser.sh`);
}
// no extra files hiding in the mirror (stale copies lie about what's live)
for (const [s, d] of DIRS) {
  const inMirror = walk(join(browser, d)).map((p) => p.slice(join(browser, d).length + 1)).sort();
  const inSrc = walk(join(root, s)).map((p) => p.slice(join(root, s).length + 1)).sort();
  assert.deepEqual(inMirror, inSrc, `browser/${d}/ lists exactly ${s}/`);
}
// the avatar ML models stay out (the profile never runs that route);
// the SOG codec bundle is the only models/ file
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
const jsFiles = pairs.map(([, dst]) => dst).filter((p) => p.endsWith('.js'));
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

console.log(`browser vendor mirror (${pairs.length} files) byte-identical + self-contained: passed`);
