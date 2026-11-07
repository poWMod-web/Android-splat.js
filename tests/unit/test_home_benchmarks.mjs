import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseStyles = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');
const data = readFileSync(new URL('../../app/js/data.js', import.meta.url), 'utf8');

assert.match(app, /const HOME_BENCHMARK_IDS = new Set\(\[[\s\S]*preset\.id === 'synthetic'[\s\S]*'42485456_7518', \/\/ Camping/,
  'the home benchmark allowlist should contain Synthetic Corner and Camping only');
assert.match(app, /const presets = rest\.filter\(\(x\) => HOME_BENCHMARK_IDS\.has\(String\(x\.id\)\)\)/,
  'the home Benchmarks tab should filter all other benchmark tiles');
assert.match(html, /id="btn-xr-example"[\s\S]*Synthetic Corner|known cameras/i,
  'the Synthetic Corner example should remain on the home screen');
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
assert.match(data, /https:\/\/github\.com\/poWMod-web\/Android-splat\.js/,
  'the Source action should point at the Android-splat.js repository');
assert.match(styles, /#btn-xr-help, #btn-training-visuals, #xr-device-guide\s*\{[\s\S]*padding: 4px 9px;[\s\S]*font-size: 11px;/,
  'Device guide and Show visuals should use the compact utility-button size');
assert.match(html, /https:\/\/github\.com\/poWMod-web\/Android-splat\.js/,
  'the home Source link should use the requested repository');

console.log('Home benchmark allowlist, compact controls, source link, and removed login/install actions: passed');
