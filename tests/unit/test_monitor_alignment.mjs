import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browserApp = readFileSync(new URL('../../browser/app/js/app.js', import.meta.url), 'utf8');
const headless = readFileSync(new URL('../../app/js/train-headless.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const service = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');
const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const sessionIo = readFileSync(new URL('../../app/js/session_io.js', import.meta.url), 'utf8');
const align = readFileSync(new URL('../../app/js/align.js', import.meta.url), 'utf8');
const css = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');

assert.match(headless, /function decimateHistory\(rows, max = 800\)[\s\S]*\(rows\.length - 1\) \/ \(max - 1\)/,
  'background progress should bound payloads by sampling the whole run');
assert.match(headless, /scoreHistory: decimateHistory\(ses\.lossHistory \|\| \[\]\)/);
assert.match(headless, /splatHistory: decimateHistory\(splatHist\)/);
assert.doesNotMatch(headless, /scoreHistory:[^\n]*slice\(-400\)/,
  'the Activity side of the curve must not disappear from the monitor after handoff');
assert.match(headless, /events: chartEvents\.slice\(-300\)/,
  'background progress carries prune/grow marks to the monitor');
assert.match(app, /c\.events = \(Array\.isArray\(progress\.events\)/,
  'the monitor graph renders durable service marks');
assert.match(headless, /e\.kind === 'prune-continue'/);
assert.match(headless, /e\.kind === 'grow-now'/);
assert.match(app, /new Chart\(\$\('mon-chart'\)/);
assert.match(app, /async function refreshMonitorPreview\(\)/);
assert.match(app, /bridge\.headlessCommand\('headless-snapshot', runId\)/);
const livePaint = app.slice(app.indexOf('function paintMonitorLive'), app.indexOf('function paintMonitorProgress'));
assert.doesNotMatch(livePaint, /snapshot|headlessCommand/,
  'ordinary monitor polling must not render, read, or write a preview');
assert.match(app, /bridge\.headlessPrune\(S\._monRun, percent\)/);
assert.match(main, /headlessPrune\(String runId, int percent\)/);
assert.match(service, /getIntExtra\("prunePercent", 25\)/);
assert.match(app, /bridge\.headlessGpu\(S\._monRun, value\)/,
  'monitor GPU slider must update the live service trainer');
assert.match(main, /headlessGpu\(String runId, int percent\)/);
assert.match(service, /getIntExtra\("gpuPercent", 100\)/);
assert.match(headless, /globalThis\.__headlessGpu = \(percent\)/);
assert.match(headless, /ses\.opts\.gpuUsage = v \/ 100/);
assert.match(headless, /gpu: Math\.round\(\(ses\.opts\.gpuUsage \?\? 1\) \* 100\)/,
  'progress reports the service trainer’s authoritative live duty');
for (const cmd of ['headless-prune', 'headless-grow', 'headless-save']) {
  assert.ok(main.includes(cmd), `${cmd} must cross the Activity bridge`);
  assert.ok(service.includes(cmd), `${cmd} must dispatch in the service`);
}
assert.match(app, /bridge\.headlessCommand\('headless-save', runId\)/);
assert.match(headless, /globalThis\.__headlessSave = async \(\) =>/);
assert.match(headless, /await ses\.pauseAndDrain\(\);[\s\S]*await ses\.finish\(\)/,
  'Stop & save crosses a safe GPU boundary before finalizing');
assert.match(headless, /globalThis\.__headlessGrow/);
assert.match(headless, /globalThis\.__headlessPrune/);
assert.match(headless, /resume = await ses\.pauseAndDrain\(\);[\s\S]*const state = await packState\(ses\)/,
  'service checkpoints wait for the active GPU frame before mapping buffers');
assert.match(session, /async pauseAndDrain\(\)/);
assert.match(session, /const wasTraining = await this\.pauseAndDrain\(\);[\s\S]*return this\._locked/,
  'manual prune drains its frame before taking the readback lock');
assert.match(session, /const work = this\._frameLoop\(\);[\s\S]*this\._frameWork = work/,
  'the pause barrier tracks the complete in-flight frame');

assert.match(app, /adjacentPoseCandidates\(S\.photos, S\.scene\.cams, i\)/);
assert.match(align, /ransacE\(/);
assert.match(align, /selectPose\(/);
assert.match(align, /const neighbours = \[-2, -1, 1, 2\]/,
  'automatic alignment uses two pictures on either side');
assert.match(app, /Balanced · \$\{good\.length\} neighbours/,
  'consistent adjacent poses are offered as a consensus placement');
assert.match(app, /undoSelectedAlignment/);
for (const [name, source] of [['Android', app], ['Browser', browserApp]]) {
  assert.match(source, /function inAppConfirm\(title, message[\s\S]*dialog\.showModal\(\)/,
    `${name}: alignment confirmation should use a branded in-app dialog`);
  assert.match(source, /inAppConfirm\(`Align frame \$\{i \+ 1\}`[\s\S]*'Find placements'/,
    `${name}: the alignment dialog should have a meaningful frame title`);
  assert.doesNotMatch(source, /confirm\(`(?:Undo the alignment|Apply “|Find automatic placements)/,
    `${name}: alignment must not use an origin-titled browser confirmation`);
}
assert.match(css, /\.app-confirm[\s\S]*\.app-confirm \.dlg-head h2/,
  'the branded alignment confirmation should have dedicated modal styling');
assert.doesNotMatch(app, /Manual frame alignment|manualAlignSelected/,
  'the first alignment version is automatic-only');
assert.match(app, /align\.id = 'c-align'/,
  'Align belongs to the selected-frame controls');
assert.doesNotMatch(app, /strip-align|Align selected/,
  'the filmstrip must not show Align before an image is selected');
assert.match(css, /#monitor \{ width: 100%; max-width: none; height: calc\(100dvh - 52px\); box-sizing: border-box; \}/,
  'the background monitor covers the full viewport below the app bar');
assert.match(app, /applyAlignedPose[\s\S]*persistReconEdit\(\)/,
  'alignment should persist camera metadata without rewriting the splat state');
assert.match(session, /async updateCameraPose\(ci, R, t\)/);
assert.match(session, /this\.recon\?\.cams\?\.find/);
assert.match(app, /No suitable alignment for frame/);
assert.match(app, /remove\.textContent = 'Remove image'/);
assert.match(app, /S\.session\.excludeCameraImage\(cam\.ci\)/,
  'failed automatic alignment should offer removal from the trainable camera set');
assert.match(session, /async excludeCameraImage\(ci\)[\s\S]*rc\.excluded = true[\s\S]*_commitSplit\(excluded\)/,
  'image removal should exclude the live trainer and mark the reconstruction');
assert.match(app, /state: rc\?\.excluded \? 'removed'/,
  'restored removed images should remain visibly excluded');
assert.match(sessionIo, /\.\.\.\(c\.excluded \? \{ excluded: true \} : \{\}\)/,
  'checkpoints should persist explicit image removal');

console.log('background monitor controls + durable selected-frame alignment: passed');
