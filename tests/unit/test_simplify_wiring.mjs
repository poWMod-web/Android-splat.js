import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browser = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');
const browserHtml = readFileSync(new URL('../../browser/index.html', import.meta.url), 'utf8');

// ---- the session exposes a simplified export --------------------------------
assert.match(session, /import \{ simplifyGaussians \} from '\.\/gs\/simplify\.js';/,
  'the session loads the simplifier');
assert.match(session, /exportSimplifiedPlyBlob\(ratio, \{ onStatus = \(\) => \{\}, onProgress = null \} = \{\}\) \{/,
  'the simplified export takes a ratio plus progress hooks');
assert.match(session, /\{ data, n, sh, shK, dc: this\.trainer\.dcMode \},/,
  'the simplifier gets the readback plus the trainer color mode');
assert.match(session, /return this\._bakedPly\(simp\.data, simp\.n, simp\.sh, simp\.shK\);/,
  'the simplified model leaves through the same bake + filter as a full export');
assert.match(session, /return this\._bakedPly\(data, n, sh, shK\);/,
  'the full export shares that tail (no duplicated filter)');

// ---- the full app offers it in the share sheet ------------------------------
assert.match(app, /act: 'simplify', label: 'Download simplified \.ply'/,
  'the share sheet lists the simplified download');
assert.match(app, /if \(!S\.restored\) acts\.push\(\{ act: 'simplify'/,
  'view-only restores hide it (no live trainer to read back)');
assert.match(app, /data-ratio="0\.5">Keep ½<\/button>/,
  'the picker card offers half size');
assert.match(app, /S\.session\.exportSimplifiedPlyBlob\(ratio, \{/,
  'the picked ratio drives the export');

// ---- the browser mirrors it --------------------------------------------------
assert.match(browserHtml, /<button id="btn-simplify" class="btn" disabled>Simplify ½ ↓<\/button>/,
  'the button sits next to Download .ply');
assert.match(browser, /simplify: \$\('btn-simplify'\),/,
  'the button is wired into the ui map');
assert.match(browser, /s\.exportSimplifiedPlyBlob\(0\.5, \{/,
  'the browser simplifies to half');
assert.match(browser, /ui\.simplify\.disabled = false;\n  ui\.sess\.disabled = false;/,
  'finishing a run enables it');
assert.match(browser, /ui\.simplify\.disabled = true;   \/\/ needs the live trainer — continue first/,
  'a bare restore keeps it disabled');

console.log('simplify wiring (session + app + browser): passed');
