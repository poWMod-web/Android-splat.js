import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browser = readFileSync(new URL('../../browser/app/js/app.js', import.meta.url), 'utf8');
const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const headless = readFileSync(new URL('../../app/js/train-headless.js', import.meta.url), 'utf8');

for (const [name, source] of [['Android', app], ['Browser', browser]]) {
  const wireStart = source.indexOf('function wireFrameLongPress(');
  const wireEnd = source.indexOf('function buildStrip(', wireStart);
  assert.ok(wireStart >= 0 && wireEnd > wireStart, `${name}: filmstrip long-press wiring should exist`);
  const wire = source.slice(wireStart, wireEnd);
  assert.match(wire, /setTimeout\([\s\S]*cam\.state === 'removed' \? restoreTrainingImage\(i\) : removeTrainingImage\(i\)[\s\S]*1000/,
    `${name}: a one-second hold should remove active images or restore removed ones`);
  assert.match(wire, /Math\.hypot\([\s\S]*> 10\) cancel\(\)/,
    `${name}: scrolling the filmstrip should cancel the hold`);
  assert.match(wire, /if \(!fired\) return;[\s\S]*preventDefault\(\)[\s\S]*stopImmediatePropagation\(\)/,
    `${name}: the release click after a completed hold must not select the image`);

  const removeStart = source.indexOf('async function removeTrainingImage(');
  const removeEnd = source.indexOf('async function restoreTrainingImage(', removeStart);
  const remove = source.slice(removeStart, removeEnd);
  assert.match(remove, /inAppConfirm\('Remove image from training'[\s\S]*excludeCameraImage\(cam\.ci\)/,
    `${name}: removal requires a titled in-app confirmation and uses stable camera exclusion`);
  assert.doesNotMatch(remove, /\bconfirm\(/,
    `${name}: image removal must not show the WebView origin as its title`);
  assert.match(remove, /cam\.state = 'removed'[\s\S]*persistReconEdit\(\)/,
    `${name}: removed state should be visible and durable through metadata only`);
  assert.doesNotMatch(remove, /checkpointRun|packState/,
    `${name}: removing one camera must not rewrite the full splat model`);
  assert.doesNotMatch(remove, /S\.photos\.splice|S\.scene\.cams\.splice/,
    `${name}: removal must not shift persisted photo/camera indices`);
  const restoreStart = source.indexOf('async function restoreTrainingImage(');
  const restoreEnd = source.indexOf('function wireFrameLongPress(', restoreStart);
  const restore = source.slice(restoreStart, restoreEnd);
  assert.match(restore, /inAppConfirm\('Restore image to training'[\s\S]*includeCameraImage\(cam\.ci\)[\s\S]*cam\.state = 'placed'/,
    `${name}: removed thumbnails should confirm and restore the stable camera entry`);
  assert.match(source, /c\?\.state === 'removed' \? 'Long press to restore to training'/,
    `${name}: removed thumbnails should advertise their reversible action`);
  assert.match(source, /async function persistReconEdit\(\)[\s\S]*buildReconJson\(S\)[\s\S]*'recon\.json'[\s\S]*patchRun\(S\.runId/,
    `${name}: camera/training-set edits should update recon metadata in both durable stores`);
  const editPersist = source.slice(source.indexOf('async function persistReconEdit('), source.indexOf('function formatTrainingScore'));
  assert.doesNotMatch(editPersist, /packState|state\.bin/,
    `${name}: metadata persistence must not read or write the full splat state`);
}

assert.match(session, /async includeCameraImage\(ci\)[\s\S]*rc\.excluded = false[\s\S]*excluded\.delete\(i\)[\s\S]*_commitSplit\(excluded\)/,
  'restoring an image should re-enable every camera backed by it and reset the epoch deck');
assert.doesNotMatch(session.slice(session.indexOf('async includeCameraImage('), session.indexOf('async growNow()')),
  /\.splice\(/, 'restoring an image must preserve camera and reconstruction indices');
assert.ok((app.match(/\.\.\.\(c\.excluded \? \{ excluded: true \} : \{\}\)/g) || []).length >= 2,
  'finished-model loading and continuation should both carry removed-image markers into the session');
assert.match(headless, /useReconstruction\([\s\S]*c\.excluded \? \{ excluded: true \}/,
  'background continuation should carry removed-image markers into the service session');

console.log('filmstrip long-press image removal and restoration: passed');
