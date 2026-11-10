import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const xr = readFileSync(new URL('../../app/js/xr-ui.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const nativeCap = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');

// Imported frames are fetched as Blobs, so once a native capture is accepted
// its on-disk copies are redundant — and deleting models never touches them.
// The success path must drop them instead of leaving ~200 JPEGs behind.
assert.match(xr, /if \(!reviewed\) return;\s*await onCapture\(reviewed\);\s*\/\/[\s\S]*?try \{ nativeBridge\.resetCaptureSession\(\); \} catch/,
  'an accepted native import should wipe the redundant on-disk capture');
assert.match(main, /@JavascriptInterface public void resetCaptureSession\(\) \{[\s\S]*?NativeCaptureActivity\.resetCaptureSession\(MainActivity\.this\);/,
  'the bridge should expose the native capture wipe');
assert.match(nativeCap, /static void resetCaptureSession\(Context context\) \{[\s\S]*?deleteTree\(new File\(context\.getFilesDir\(\), "captures\/current"\)\);/,
  'the wipe should delete the capture directory when no capture is live');
assert.match(nativeCap, /if \(active == this\) active = null;/,
  'finishing the capture activity should release it so the wipe takes the delete path');

console.log('native capture cleanup after import: passed');
