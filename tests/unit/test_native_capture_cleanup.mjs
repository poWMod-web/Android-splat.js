import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const xr = readFileSync(new URL('../../app/js/xr-ui.js', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const nativeCap = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');

// A completed native scan is an immutable, ID-addressed archive. New scans
// and training handoffs must reference it rather than wiping or copying it.
assert.match(nativeCap, /captureId = "capture_" \+ System\.currentTimeMillis\(\);[\s\S]*new File\(new File\(getFilesDir\(\), "captures"\), captureId\)/,
  'each capture should get a stable native archive directory');
const onCreate = nativeCap.slice(nativeCap.indexOf('onCreate(Bundle'), nativeCap.indexOf('private void applySafeAreaInsets'));
assert.doesNotMatch(onCreate, /deleteTree\(/,
  'opening a new capture must not delete an older scan archive');
assert.match(nativeCap, /if \(!captureFinished && outputDir != null\) deleteTree\(outputDir\)/,
  'only an incomplete capture should be cleaned up automatically');
assert.match(nativeCap, /putExtra\("captureId", captureId\)/,
  'the completed archive identity should be returned to the WebView');
assert.match(main, /splatxr-native-capture[\s\S]*detail:\{captureId:/,
  'the native result event should carry the archive identity');
assert.match(xr, /export async function importNativeCapture\(nativeBridge[\s\S]*captureId = null[\s\S]*exportDataset\(captureId \|\| ''\)[\s\S]*nativeCaptureId: payload\.captureId/,
  'native import should preserve its archive reference');
assert.match(main, /public String listCaptureArchives\(\)[\s\S]*previewUri/,
  'Home should be able to enumerate all valid native scan archives');
assert.match(main, /Native scan archives are the image ground truth[\s\S]*public String exportDataset\(String requestedId\)/,
  'the bridge should export references for a requested archive');
assert.match(main, /public String deleteCaptureArchive\(String captureId\)[\s\S]*manifest\.json[\s\S]*\"referenced\"/,
  'explicit scan deletion should refuse archives referenced by durable runs');
assert.match(main, /public boolean deleteRunArchive\(String runId\)[\s\S]*readActive/,
  'explicit model deletion should protect a live run and remove its durable directory');

console.log('native capture archive lifecycle: passed');
