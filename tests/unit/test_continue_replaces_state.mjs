import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

const continueStart = app.indexOf('async function continueLocalRun(');
const continueEnd = app.indexOf('function continueTraining(', continueStart);
assert.ok(continueStart >= 0 && continueEnd > continueStart,
  'the local-run continuation function should exist');
const cont = app.slice(continueStart, continueEnd);

// Continuing — paused or finished — trains back into the run's own record,
// so the new result replaces the previous trained state instead of minting
// a second record next to it.
assert.match(cont, /S\._resumeRunId = lr\.id;/,
  'continuing a local run should keep the run identity for its record');
assert.doesNotMatch(cont, /status !== 'finished'\) S\._resumeRunId/,
  'finished runs should continue in place, not only paused ones');

// The continuation rebuilds from the record's own solved poses (the root
// camera data stays the run's poses) and resolves its photos by name from
// the already-stored capture — re-saving them would duplicate the images.
assert.match(cont, /ses\.useReconstruction\(\{[\s\S]*cams: recon\.cams\.map\(/,
  'the continued session should rebuild from the saved run camera poses');
assert.match(cont, /const rec = await loadLastCapture\(\);[\s\S]*const byName = new Map\(rec\.files\.map/,
  'continued photos should come from the stored capture, matched by name');
assert.doesNotMatch(cont, /persistCapture|saveLastCapture/,
  'continuing must reuse the stored photo blobs, never re-save copies');

// Identity reuse lands in startTraining's resume branch (mark live, no new
// record); checkpoints and compression then overwrite the trained state,
// poses-stats bundle, and compressed result within that same record.
assert.match(app, /const keepId = resumeId \|\| S\.runId \|\| null;[\s\S]*S\.runId = keepId \|\| `run_/,
  'a resumed run id should become the run id instead of minting a new record');
assert.match(app, /if \(keepId\) \{[\s\S]*patchRun\(keepId, livePatch\(\)\)/,
  'resuming should mark the existing record live without creating one');
assert.match(app, /state, recon, ckptAt: Date\.now\(\), minutes:/,
  'checkpoints should overwrite the raw trained state and recon in place');
assert.match(app, /sog: blob, recon: buildReconJson\(S\), splats: S\.splats,/,
  'compression should replace the stored sog result in the same record');
assert.match(app, /status: 'finished', iter: S\.iter, splats: S\.splats,/,
  'finishing should mark the same record finished with the new result');

console.log('continued runs replace their trained state in place: passed');
