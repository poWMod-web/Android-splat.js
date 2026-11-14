import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const xrUi = readFileSync(new URL('../../app/js/xr-ui.js', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseStyles = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');
const manifest = readFileSync(new URL('../../android/app/src/main/AndroidManifest.xml', import.meta.url), 'utf8');

assert.doesNotMatch(html, /btn-xr-example|Try a scene with known cameras|Synthetic Corner/i,
  'the synthetic room example must not appear on the home screen');
assert.doesNotMatch(app, /onExample|btn-xr-example|fetchGallery|HOME_BENCHMARK_IDS|creationTile\(/,
  'the synthetic example and remote community/benchmark gallery must not be active in the app');
const wall = app.match(/async function mountWall\(\)[\s\S]*?(?=\/\*\* From the viewer into the studio)/)?.[0] || '';
assert.match(wall, /listHomeLibrary\(\)[\s\S]*nativeCaptureTiles\(catalog\.captures\)[\s\S]*androidRunTiles\(catalog\.runs\)[\s\S]*localRunTiles\(\)/,
  'Android should merge canonical captures with native and main-branch model checkpoints');
assert.match(wall, /typeof native\.listHomeLibrary === 'function'[\s\S]*listHomeLibrary\(\)[\s\S]*nativeCaptureTiles\(catalog\.captures\)[\s\S]*androidRunTiles\(catalog\.runs\)[\s\S]*else \{[\s\S]*lastCaptureTile\(\)[\s\S]*localRunTiles\(\)/,
  'Android captures stay native while model checkpoints retain main-branch compatibility');
assert.doesNotMatch(wall, /Promise\.all\(/,
  'catalog tile groups should paint independently');
assert.doesNotMatch(wall, /fetch\(|import\(['"]\.\/share\.js/,
  'mounting saved work must not issue network requests');
assert.match(wall, /heading = document\.createElement\('button'\)[\s\S]*wall-refresh[\s\S]*heading\.addEventListener\('click', \(\) => mountWall\(\)\)/,
  'the saved-work heading should explicitly refresh the native and IndexedDB catalogs');
assert.match(app, /visibilitychange[\s\S]*S\.state === 'ready'[\s\S]*mountWall\(\)/,
  'returning to a visible Home screen should automatically refresh saved models');
assert.match(app, /nativeNewer[\s\S]*Keep main's full model record[\s\S]*status: nativeRow\.status[\s\S]*splats: nativeRow\.splats/,
  'service-owned Android progress must overlay stale IndexedDB card metadata');
assert.match(app, /openNativeFresh[\s\S]*mirrorHeadlessRun\(r\.id, nativeRow\)/,
  'opening a newer native checkpoint should refresh the model record before continuing');
assert.doesNotMatch(xrUi, /btn-xr-example|onExample/,
  'XR UI must not retain a handler for the removed demo tile');
assert.doesNotMatch(html, /fonts\.googleapis\.com|fonts\.gstatic\.com/,
  'page rendering must not fetch external fonts');
assert.match(baseStyles, /--ui:\s+system-ui/,
  'the app should use locally installed system fonts');
assert.doesNotMatch(manifest, /android\.permission\.INTERNET/,
  'the Android app must not request Internet permission');
assert.doesNotMatch(html, /id="btn-install"|id="xr-install-app"|Install app|Install on Android/,
  'all install actions should be removed from the home UI');
assert.doesNotMatch(html, /id="acct"|>Login</,
  'the standalone login/account control should be removed');
assert.doesNotMatch(app, /function wireAccount\(|function renderAccount\(|handleOAuthCallback|hasToken\(|focusSignIn\(|sendToArrival\(|arrival-oauth|shareCreation\(|adminDashboard|setSharePrivacy\(|renameShare\(|deleteShare\(/,
  'account sign-in, admin, and cloud-publishing controls should be removed from the app');
const avatarPublish = readFileSync(new URL('../../app/avatar/stages/publish.js', import.meta.url), 'utf8');
assert.doesNotMatch(avatarPublish, /getToken|hasToken|uploadFile|arrival-signin|Use as my avatar/,
  'avatar packaging should download locally without an account sign-in flow');
assert.match(avatarPublish, /Download package/,
  'the avatar completion stage should retain its local package download');
assert.match(html, /https:\/\/github\.com\/poWMod-web\/Android-splat\.js/,
  'the home Source link should use the Android-splat.js repository');
assert.match(styles, /#btn-xr-help, #btn-training-visuals, #xr-device-guide\s*\{[\s\S]*padding: 4px 9px;[\s\S]*font-size: 11px;/,
  'Device guide and Show visuals should use the compact utility-button size');

console.log('Local-only home, removed demo/gallery, offline Android manifest, compact controls, and removed login/install actions: passed');
