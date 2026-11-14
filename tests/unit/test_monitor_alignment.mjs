import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const headless = readFileSync(new URL('../../app/js/train-headless.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const service = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');
const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const align = readFileSync(new URL('../../app/js/align.js', import.meta.url), 'utf8');

assert.match(headless, /scoreHistory: \(ses\.lossHistory \|\| \[\]\)\.slice\(-400\)/);
assert.match(app, /new Chart\(\$\('mon-chart'\)/);
assert.match(app, /async function refreshMonitorPreview\(\)/);
assert.match(app, /bridge\.headlessCommand\('headless-snapshot', runId\)/);
const livePaint = app.slice(app.indexOf('function paintMonitorLive'), app.indexOf('function paintMonitorProgress'));
assert.doesNotMatch(livePaint, /snapshot|headlessCommand/,
  'ordinary monitor polling must not render, read, or write a preview');
assert.match(app, /bridge\.headlessPrune\(S\._monRun, percent\)/);
assert.match(main, /headlessPrune\(String runId, int percent\)/);
assert.match(service, /getIntExtra\("prunePercent", 25\)/);
for (const cmd of ['headless-prune', 'headless-grow']) {
  assert.ok(main.includes(cmd), `${cmd} must cross the Activity bridge`);
  assert.ok(service.includes(cmd), `${cmd} must dispatch in the service`);
}
assert.match(headless, /globalThis\.__headlessGrow/);
assert.match(headless, /globalThis\.__headlessPrune/);

assert.match(app, /adjacentPoseCandidates\(S\.photos, S\.scene\.cams, i\)/);
assert.match(align, /ransacE\(/);
assert.match(align, /selectPose\(/);
assert.match(app, /Automatic alignment found/);
assert.match(app, /Manual frame alignment/);
assert.match(app, /checkpointRun\('align'\)/);
assert.match(session, /async updateCameraPose\(ci, R, t\)/);
assert.match(session, /this\.recon\?\.cams\?\.find/);

console.log('background monitor controls + durable selected-frame alignment: passed');
