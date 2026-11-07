import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');
const html = readFileSync(resolve(root, 'app/index.html'), 'utf8');
const session = readFileSync(resolve(root, 'src/session.js'), 'utf8');

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
assert.match(app, /function restoreRunDetails\([\s\S]*session\.lossHistory = history\.map[\s\S]*session\.perf\.frames = frameRows\.map/,
  'reopening a local model should hydrate its score history and timing samples');
assert.match(app, /restoreRunDetails\(S\._localRun, ses\)/,
  'the local Details payload should be restored when the saved model opens');
assert.match(app, /restoreRunDetails\(lr, ses\)[\s\S]*S\._carryTrainElapsed = true/,
  'continuing a local run should carry forward its history and cumulative elapsed time');
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

assert.match(html, /<span id="install-slot"><button[^>]*id="btn-install"/,
  'the PWA install action should be isolated in its own top-bar slot');
assert.match(html, /id="btn-training-visuals"[^>]*hidden[^>]*>Show visuals/,
  'the training-only visualization toggle should start hidden');
assert.match(app, /if \(installSlot\) installSlot\.hidden = training/,
  'the install slot should be replaced only during training');
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

console.log('Local model Details persistence, continuation, and training visualization gate: passed');
