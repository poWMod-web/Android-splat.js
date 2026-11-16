import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { captureSrcKey } from '../../app/js/store.js';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');

// the fingerprint: stable across entry shapes, sensitive to content
const fresh = [{ name: 'a.jpg', size: 10 }, { name: 'b.jpg', size: 20 }];
const restored = [{ name: 'b.jpg', blob: { size: 20 } }, { name: 'a.jpg', blob: { size: 10 } }];
assert.equal(captureSrcKey('photos', fresh), captureSrcKey('photos', restored),
  'a restored capture should fingerprint like the fresh pick (order- and shape-proof)');
assert.notEqual(captureSrcKey('photos', fresh), captureSrcKey('video', fresh),
  'the kind should separate keys');
assert.notEqual(captureSrcKey('photos', fresh),
  captureSrcKey('photos', [{ name: 'a.jpg', size: 11 }, { name: 'b.jpg', size: 20 }]),
  'file sizes should separate keys');
assert.notEqual(captureSrcKey('photos', fresh),
  captureSrcKey('photos', [{ name: 'a.jpg', size: 10 }]),
  'the file count should separate keys');

// same-session re-entry (fault recovery, double Start) keeps the live record
assert.match(app, /const keepId = resumeId \|\| S\.runId \|\| null;/,
  'starting training should reuse the live run id when one is set');
assert.match(app, /S\.runId = keepId \|\| `run_\$\{Date\.now\(\)\}/,
  'only a genuinely fresh run should mint a record id');

// fresh own-capture runs adopt their stale same-source record — finished
// results and live-in-another-tab records are never adopted
assert.match(app, /r\.srcKey === srcKey && r\.status !== 'finished'/,
  'adoption should only consider unfinished same-source records');
assert.match(app, /Date\.now\(\) - \(r\.updatedAt \|\| 0\) > 60000/,
  'adoption should leave records written in the last minute alone');
assert.match(app, /S\.runId = stale\.id;/,
  'an adopted record should become the run’s identity');
const liveStart = app.indexOf('const livePatch = () => ({');
const livePatch = app.slice(liveStart, app.indexOf('});', liveStart));
assert.match(livePatch, /status: 'training'/, 'going live should mark the record training');
assert.doesNotMatch(livePatch, /name:|thumb|recon|poses|createdAt/,
  'going live should move only the training head — name, thumb, poses and photos stay');
assert.match(app, /id: runId, name: runName, \.\.\.\(srcKey \? \{ srcKey \} : \{\}\),/,
  'new records should carry the source key for the next run to adopt by');


console.log('run identity: replace, never duplicate, never clobber: passed');
