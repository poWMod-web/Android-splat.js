import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// one UI, two shells: the marker (generated browser entry) or ?shell=browser
assert.match(app, /const SHELL = new URLSearchParams\(location\.search\)\.get\('shell'\) \|\| globalThis\.__SPLAT_SHELL \|\| 'app';/,
  'the shell should resolve from the query, the generated marker, or default to app');
assert.match(app, /const WALL_FIRST = SHELL === 'browser' \|\|/,
  'the browser profile should always be wall-first');

// the browser profile hides AR capture, the avatar route and backend links;
// photo/pose review stays (it is local)
assert.match(app, /for \(const id of \['btn-ar', 'btn-record', 'btn-xr-help', 'xr-open-tab', 'xr-ar-state', 'xr-device-guide', 'av-choice'\]\)/,
  'the browser profile should hide AR and camera actions, guides and the avatar choice');
assert.match(app, /document\.querySelector\('\.xr-flow'\)\?\.remove\(\);/,
  'the browser profile should drop the AR marketing strip');
assert.match(app, /if \(SHELL === 'browser'\) S\.avatarWanted = false;/,
  'the browser profile should never run the avatar route');

// local-only: no shared-space restore, no space relink, no service worker
assert.match(app, /if \(mp\.get\('space'\) && SHELL !== 'browser'\) restoreShared\(mp\.get\('space'\)\);/,
  'the browser profile should not restore arrival spaces');
assert.match(app, /if \(SHELL !== 'browser' && S\._localRun && S\._localRun\.spaceId\) S\.share = /,
  'the browser profile should not relink share identities');
assert.match(app, /if \(setOrPreset && setOrPreset\.spaceId && SHELL !== 'browser'\) \{/,
  'the browser profile should not resolve shared-model links on the detail card');
assert.match(app, /if \(SHELL !== 'browser' && 'serviceWorker' in navigator\) \{/,
  'the browser profile should not install the app service worker');

console.log('browser shell profile gates: passed');
