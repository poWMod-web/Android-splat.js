import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');

assert.match(app, /const HOME_BENCHMARK_IDS = new Set\(\[[\s\S]*preset\.id === 'synthetic'[\s\S]*'42485456_7518', \/\/ Camping/,
  'the home benchmark allowlist should contain Synthetic Corner and Camping only');
assert.match(app, /const presets = rest\.filter\(\(x\) => HOME_BENCHMARK_IDS\.has\(String\(x\.id\)\)\)/,
  'the home Benchmarks tab should filter all other benchmark tiles');
assert.match(html, /id="btn-xr-example"[\s\S]*Synthetic Corner|known cameras/i,
  'the Synthetic Corner example should remain on the home screen');
assert.doesNotMatch(html, /id="btn-install"|id="xr-install-app"|Install app|Install on Android/,
  'all install actions should be removed from the home UI');

console.log('Home benchmark allowlist and removed install actions: passed');
