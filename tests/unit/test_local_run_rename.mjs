import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');
const store = readFileSync(resolve(root, 'app/js/store.js'), 'utf8');
const from = app.indexOf('async function localRunTiles(');
const to = app.indexOf('/** the first `cnt` photos', from);
assert.ok(from >= 0 && to > from, 'local run tile renderer should be present');
const localTiles = app.slice(from, to);
assert.match(localTiles, /\{ label: 'Rename', act: renameLocalModel \}/,
  'local model menus should include Rename regardless of cloud-share state');
assert.match(localTiles, /textCard\('Rename this model', previousName\)/,
  'renaming should use the app rename dialog');
assert.match(localTiles, /patchRun\(r\.id, \{ name \}\)/,
  'the updated model name must be persisted to local storage');
assert.match(localTiles, /title\.textContent = name/,
  'the device model tile should update immediately after rename');
assert.match(store, /export async function patchRun\(id, patch\)[\s\S]*?return updated;/,
  'patchRun should report whether the requested model record was updated');
console.log('This device model rename action and persistence: passed');
