import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const xr = readFileSync(new URL('../../app/js/xr-ui.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const nativeCap = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');

// Browser-only imports can clean their native copies after IndexedDB accepts
// them. Headless-capable Android keeps captures/current as an independent
// recovery source, and can rehydrate it when IndexedDB is absent.
assert.match(xr, /if \(!reviewed\) return;\s*await onCapture\(reviewed\);\s*\/\/[\s\S]*?try \{ nativeBridge\.resetCaptureSession\(\); \} catch/,
  'non-headless imports may wipe their redundant on-disk capture');
assert.match(xr, /export async function importNativeCapture\(nativeBridge[\s\S]*exportDataset\(\)[\s\S]*payload\.frames\.map/,
  'a completed scan should be rehydratable from the Android data folder');
assert.match(main, /@JavascriptInterface public void resetCaptureSession\(\) \{[\s\S]*?NativeCaptureActivity\.resetCaptureSession\(MainActivity\.this\);/,
  'the bridge should expose the native capture wipe');
assert.match(nativeCap, /static void resetCaptureSession\(Context context\) \{[\s\S]*?deleteTree\(new File\(context\.getFilesDir\(\), "captures\/current"\)\);/,
  'the wipe should delete the capture directory when no capture is live');
assert.match(nativeCap, /if \(active == this\) active = null;/,
  'finishing the capture activity should release it so the wipe takes the delete path');

console.log('native capture cleanup after import: passed');
