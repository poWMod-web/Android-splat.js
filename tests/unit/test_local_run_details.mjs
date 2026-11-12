import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');
const html = readFileSync(resolve(root, 'app/index.html'), 'utf8');
const session = readFileSync(resolve(root, 'src/session.js'), 'utf8');
const sessionIo = readFileSync(resolve(root, 'app/js/session_io.js'), 'utf8');

const captureStart = app.indexOf('function captureRunSolveDetails()');
const captureEnd = app.indexOf('function persistRunDetails(', captureStart);
assert.ok(captureStart >= 0 && captureEnd > captureStart, 'local solver details should have a snapshot helper');
const solveSnapshot = app.slice(captureStart, captureEnd);
assert.match(solveSnapshot, /solveStats:/, 'solve counters and solve time must be persisted');
assert.match(solveSnapshot, /const features =|\bfeatures,/, 'landmark coordinates and counts must be persisted');
assert.match(solveSnapshot, /lastPair:/, 'a representative matching sample must be persisted');
assert.match(solveSnapshot, /chart: compactRows\(history/, 'score history must be retained in local run diagnostics');
assert.match(solveSnapshot, /perf:\s*\{[\s\S]*frames:/, 'timing samples must be retained');
assert.match(solveSnapshot, /settings:/, 'the training settings used by timing reports must be retained');

assert.match(app, /detailSolve: captureRunSolveDetails\(\)[\s\S]*detailProgress: captureRunProgressDetails\(\)[\s\S]*detailGpu: gpuFacts\(false\)/,
  'new local runs should save solver, progress/timing, and GPU details');
assert.match(app, /async function checkpointRun\([\s\S]*detailSolve: captureRunSolveDetails\(\)[\s\S]*detailProgress,[\s\S]*detailGpu: gpuFacts\(false\)/,
  'pause/finish checkpoints should refresh the saved Details payload');
assert.match(app, /function restoreRunDetails\([\s\S]*session\.setLossHistory\(history\)[\s\S]*session\.perf\.frames = frameRows\.map/,
  'reopening a local model should hydrate its score history and timing samples');
assert.match(app, /restoreRunDetails\(S\._localRun, ses\)/,
  'the local Details payload should be restored when the saved model opens');
assert.match(app, /restoreRunDetails\(lr, ses\)[\s\S]*S\._carryTrainElapsed = true/,
  'continuing a local run should carry forward its history and cumulative elapsed time');
const continueStart = app.indexOf('async function continueLocalRun(');
const continueEnd = app.indexOf('\nasync function ', continueStart + 1);
assert.ok(continueStart >= 0 && continueEnd > continueStart, 'the local-run continuation function should exist');
assert.match(app.slice(continueStart, continueEnd), /S\.photos = files\.map\([\s\S]*?buildStrip\(\);/,
  'continuing a saved run should rebuild the bottom image-selection strip from its restored photos');
assert.match(app, /await persistRunDetails\(runId, \{ full: true \}\)/,
  'leaving the model viewer should flush its current Details snapshot before navigation');
assert.match(app, /if \(!S\.restored && !S\._localRun\)\s*\{[\s\S]*more\.textContent = 'Train'/,
  'the generic continuation button must not render alongside a local-run Train action');
assert.match(app, /if \(S\._localRun && S\._localRun\.ownSrc\)[\s\S]*train\.textContent = 'Train'[\s\S]*else if \(rj && rj\.source/,
  'local model viewing should expose exactly the appropriate local or source-photo Train action');
assert.match(app, /if \(viewOnly\) \{[\s\S]*S\.restored = \{ hasState: true, stats: \(recon && recon\.stats\) \|\| \{\}, reconJson: recon \}[\s\S]*\$\('btn-new'\)\.hidden = false[\s\S]*enterDone\(\)/,
  'opening a local checkpoint for viewing should enter restored mode and show the top Back button');
assert.match(app, /if \(S\._localRun && S\._viewerOpen\) \{[\s\S]*closeViewerToHome\(\)/,
  'the top Back button should close a local viewer to the app home');

assert.doesNotMatch(html, /id="btn-install"|>Install app</,
  'the top-bar Install app button should be removed');
assert.match(html, /id="btn-training-visuals"[^>]*hidden[^>]*>Show visuals/,
  'the training-only visualization toggle should start hidden');
assert.doesNotMatch(app, /installSlot|initPWAInstall/,
  'removing install actions must not leave a reserved header slot or install prompt wiring');
const drawStart = app.indexOf('function draw() {');
const drawEnd = app.indexOf('/** a photograph filling the stage', drawStart);
const draw = app.slice(drawStart, drawEnd);
assert.match(draw, /S\.state === 'train' && \(!S\.trainVisuals \|\| !\$\('details'\)\.hidden\)[\s\S]*ctx\.clearRect[\s\S]*return;/,
  'training visualization should skip the app presentation path when off');
assert.ok(draw.indexOf('S.session.view.renderNow()') > draw.indexOf('S.state === \'train\' && !S.trainVisuals'),
  'the optional app viewport should be gated by the training visualization switch');
assert.match(app, /S\.session\.start\(\)/,
  'the trainer itself must continue running independently of app visualization');
assert.match(session, /setEnabled\(enabled\)[\s\S]*this\.enabled = enabled/,
  'the session viewport should expose a presentation-only enable switch');
assert.match(session, /stepOnce\(\)[\s\S]*this\.view\._tick/,
  'trainer optimization steps should remain in the frame loop when viewport rendering is disabled');
assert.match(session, /if \(!this\.enabled \|\| !this\.ctx \|\| !this\.camera\) return/,
  'disabled session view should skip its standalone GPU render');
assert.match(app, /dchart\.draw\(\)/,
  'the Details score graph remains rendered and refreshed during training');
assert.match(app, /const capturedPoints = S\.preset\?\.capture\?\.points \|\| \[\][\s\S]*const savedCloud = S\._localRun\?\.recon\?\.cloud \|\| S\.restored\?\.reconJson\?\.cloud[\s\S]*buildDisplayCloud\(capturedPoints, savedCloud, \(recon && recon\.points\) \|\| \[\]\)/,
  'the Cameras view should preserve the captured or restored local-run cloud instead of relying only on reconstruction points');
assert.match(sessionIo, /capture-time snapshot[\s\S]*potentially-mutated optimizer\/reconstruction state/,
  'saved model details should preserve the frozen camera-view point cloud');
assert.match(sessionIo, /if \(S\.scene\?\.xyz\?\.length\)/,
  'the saved scene cloud should be the source for model Details after training');

console.log('Local model Details persistence, continuation, and training visualization gate: passed');
