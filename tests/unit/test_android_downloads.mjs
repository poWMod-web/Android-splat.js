import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { saveBlob } from '../../app/js/download.js';

const appRoot = new URL('../../app/', import.meta.url);
const sog = readFileSync(new URL('js/sog.js', appRoot), 'utf8');
const app = readFileSync(new URL('js/app.js', appRoot), 'utf8');
const xrUi = readFileSync(new URL('js/xr-ui.js', appRoot), 'utf8');
const downloader = readFileSync(new URL('js/download.js', appRoot), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');

assert.match(sog, /import\(['"]\.\.\/models\/splat-transform\.bundle\.mjs['"]\)/,
  'SOG compression must import the bundle from its packaged models directory');
assert.ok(existsSync(new URL('models/splat-transform.bundle.mjs', appRoot)),
  'the SOG bundle must remain among the Android-packaged app assets');
assert.match(app, /await downloadBlob\(S\.plyBlob[\s\S]*?'ply'\)/,
  'PLY export must wait for the native save to complete before reporting success');
assert.match(xrUi, /return saveBlob\(zipStore\(entries\),[\s\S]*?'application\/zip'\)/,
  'capture ZIP export must use the same native save flow');
assert.match(main, /Intent\.ACTION_CREATE_DOCUMENT/,
  'native Android exports should open the system file destination picker');
assert.match(main, /writeFileChunk\(String base64\)/,
  'the WebView bridge should accept bounded chunks for large export files');
assert.match(downloader, /splatxr-native-file-save/);
assert.match(main, /splatxr-native-file-save/,
  'native save completion and JavaScript export handling must use the same event');

const windowEvents = new EventTarget();
globalThis.window = windowEvents;
const send = (detail) => {
  const event = new Event('splatxr-native-file-save');
  event.detail = detail;
  windowEvents.dispatchEvent(event);
};
let request;
const received = [];
globalThis.AndroidARBridge = {
  isAvailable: () => true,
  beginFileSave: (filename, mimeType) => {
    request = { filename, mimeType };
    queueMicrotask(() => send({ type: 'ready' }));
  },
  writeFileChunk: (encoded) => {
    received.push(Buffer.from(encoded, 'base64'));
    return true;
  },
  finishFileSave: () => queueMicrotask(() => send({ type: 'complete' })),
  cancelFileSave() {},
};

const bytes = Uint8Array.from({ length: 540_017 }, (_, i) => i % 251);
assert.equal(await saveBlob(new Blob([bytes]), 'capture photos.zip', 'application/zip'), true);
assert.deepEqual(request, { filename: 'capture photos.zip', mimeType: 'application/zip' });
assert.deepEqual(Buffer.concat(received), Buffer.from(bytes));

AndroidARBridge.beginFileSave = () => queueMicrotask(() => send({ type: 'cancelled' }));
assert.equal(await saveBlob(new Blob(['cancel']), 'cancel.zip', 'application/zip'), false,
  'cancelling the Android picker must not report a successful download');

delete globalThis.AndroidARBridge;
delete globalThis.window;
console.log('SOG bundle path and streamed Android file-save bridge: passed');
