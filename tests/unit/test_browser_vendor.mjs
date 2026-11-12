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
const pairs = walk(join(root, 'src'))
  .map((p) => [p, join(browser, 'src', p.slice(join(root, 'src').length + 1))]);
const APP_JS = ['chart', 'data', 'develop', 'download', 'img', 'scene-cloud',
  'session_io', 'sog', 'store', 'xr-dataset', 'zip'];
for (const f of APP_JS) pairs.push([join(root, 'app', 'js', `${f}.js`), join(browser, 'app', 'js', `${f}.js`)]);
pairs.push([join(root, 'app', 'models', 'splat-transform.bundle.mjs'),
  join(browser, 'app', 'models', 'splat-transform.bundle.mjs')]);
for (const f of ['st-worker.mjs', 'webp.wasm']) {
  pairs.push([join(root, 'app', 'vendor', f), join(browser, 'app', 'vendor', f)]);
}
assert.ok(pairs.length > 40, `the mirror should hold the engine + app modules (saw ${pairs.length})`);
for (const [src, dst] of pairs) {
  assert.ok(existsSync(dst), `${dst.slice(root.length + 1)} is vendored`);
  assert.ok(readFileSync(src).equals(readFileSync(dst)),
    `${dst.slice(root.length + 1)} drifted from its source — run sh tools/sync-browser.sh`);
}
// no extra files hiding in the mirror (stale copies lie about what's live)
const inMirror = walk(join(browser, 'src')).map((p) => p.slice(join(browser, 'src').length + 1)).sort();
const inSrc = walk(join(root, 'src')).map((p) => p.slice(join(root, 'src').length + 1)).sort();
assert.deepEqual(inMirror, inSrc, 'browser/src/ lists exactly src/');

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
  }
}

console.log(`browser vendor mirror (${pairs.length} files) byte-identical + self-contained: passed`);
