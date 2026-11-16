import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const app = readFileSync(resolve(root, 'app/js/app.js'), 'utf8');

// the overlay closer: every modal family plus both sheets and chart hovers
const helperStart = app.indexOf('function closeOverlays()');
assert.ok(helperStart >= 0, 'there should be one overlay closer for screen transitions');
const helper = app.slice(helperStart, app.indexOf('\n}\n', helperStart));
assert.match(helper, /\.upcard/, 'modal cards should be removed');
assert.match(helper, /\.xr-dialog/, 'capture dialogs should be closed');
assert.match(helper, /if \(d\.open\) d\.close\(\); else d\.remove\(\);/,
  'dialogs should close properly so their promises resolve');
assert.match(helper, /\$\('details'\)\.hidden = true/, 'the Details sheet (with its score graph) should close');
assert.match(helper, /\$\('about'\)\.hidden = true/, 'the About sheet should close');
assert.match(helper, /chart\?\.clearHover\(\);/, 'chart hovers should be cleared');

// home: no Back button, no leftovers — every home landing funnels through here
const homeStart = app.indexOf('function showHome()');
const home = app.slice(homeStart, app.indexOf('\n}\n', homeStart));
assert.match(home, /closeOverlays\(\);/, 'landing home should dismiss overlays');
assert.match(home, /\$\('btn-new'\)\.hidden = true/, 'home should hide the Back button');
assert.match(home, /dock\(''\);/, 'landing home should clear the dock graph');

// every setup screen shows Back (v2 intake, classic detail, viewer Train path)
const setupLine = /\$\('btn-new'\)\.hidden = false;   \/\/ the setup card is not home/g;
assert.equal((app.match(setupLine) || []).length, 3,
  'all three setup screens should show the Back button');
const openFn = app.slice(app.indexOf('async function open(preset'), app.indexOf('const BEATS_BASE'));
assert.match(openFn, /\$\('btn-new'\)\.hidden = false/, 'the v2 setup card should show Back');
assert.match(openFn, /closeOverlays\(\);/, 'the v2 setup card should start without overlays');
const detailFn = app.slice(app.indexOf('function showDetail('), app.indexOf("addEventListener('popstate'"));
assert.match(detailFn, /\$\('btn-new'\)\.hidden = false/, 'the classic setup card should show Back');
const trainShare = app.slice(app.indexOf('function trainFromShare()'), app.indexOf('bmp(S.photos[0].url);'));
assert.match(trainShare, /\$\('btn-new'\)\.hidden = false/, 'the Train-from-viewer setup should show Back');
assert.match(trainShare, /closeOverlays\(\);/, 'leaving the viewer for setup should close its sheets');

// going back to a trained model or home closes overlays — every landing calls it
const calls = (app.match(/\n\s*closeOverlays\(\);/g) || []).length;
assert.equal(calls, 11, `all eleven transition sites, including monitor/recovery/adopt, should call the closer (found ${calls})`);
for (const [name, from, to] of [
  ['closeDetailCard', 'function closeDetailCard()', 'async function closeViewerToHome()'],
  ['closeViewerToHome', 'async function closeViewerToHome()', 'async function fetchShareRecon('],
  ['restoreShared', 'async function restoreShared(', 'function setStartStyle('],
]) {
  const slice = app.slice(app.indexOf(from), app.indexOf(to));
  assert.match(slice, /closeOverlays\(\);/, `${name} should dismiss overlays`);
}
const restoreHead = app.slice(app.indexOf('async function restoreSession('),
  app.indexOf("flash('Loading the model"));
assert.match(restoreHead, /closeOverlays\(\);/, 'landing on a local model should dismiss overlays');
assert.match(restoreHead, /\$\('detail'\)\.hidden = true/, 'landing on a model should hide the setup card');

// the phone Back gesture: snapshot the pushed layers, dismiss, then walk as before
const popStart = app.indexOf("addEventListener('popstate'");
const pop = app.slice(popStart, app.indexOf('\n});\n', popStart));
assert.match(pop, /const aboutOpen =/, 'popstate should snapshot the About layer before closing');
assert.match(pop, /const viewerOpen = S\._viewerOpen;/, 'popstate should snapshot the viewer layer before closing');
assert.match(pop, /const detailOpen =/, 'popstate should snapshot the detail layer before closing');
assert.match(pop, /closeOverlays\(\);/, 'phone Back should dismiss modals and score graphs');
assert.match(pop, /if \(aboutOpen\) return;/, 'phone Back should still close About first');



const dockStart = app.indexOf('function dock(kind)');
const dockFn = app.slice(dockStart, app.indexOf('function syncScoreScaleControls', dockStart));
assert.match(dockFn, /#stage > \.chartwrap, #stage > \.chartwrap-expanded[\s\S]*forEach\(\(el\) => el\.remove\(\)\)/,
  'every dock transition must destroy a score graph reparented into stage');
console.log('back-button visibility + overlay cleanup on navigation: passed');

const doneStart = app.indexOf('function enterDone()');
const done = app.slice(doneStart, app.indexOf('/** Avatar mode', doneStart));
assert.match(done, /S\.state = 'done';[\s\S]*\$\('btn-new'\)\.hidden = false/,
  'every model-view path must expose Back as an enterDone invariant');
