const SAVE_EVENT = 'splatxr-native-file-save';
const CHUNK_SIZE = 256 * 1024;
let nativeSavePending = false;

function base64Of(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function nativeSave(blob, filename, mimeType, bridge) {
  if (nativeSavePending) return Promise.reject(new Error('Another file save is already in progress.'));
  nativeSavePending = true;
  return new Promise((resolve, reject) => {
    let settled = false;
    let transferring = false;
    const cleanup = () => {
      window.removeEventListener(SAVE_EVENT, onSaveEvent);
      nativeSavePending = false;
    };
    const fail = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      try { bridge.cancelFileSave?.(); } catch { /* native transfer already closed */ }
      reject(error instanceof Error ? error : new Error(String(error)));
    };
    const onSaveEvent = (event) => {
      const detail = event.detail || {};
      if (detail.type === 'ready') {
        if (transferring || settled) return;
        transferring = true;
        (async () => {
          for (let offset = 0; offset < blob.size; offset += CHUNK_SIZE) {
            const bytes = new Uint8Array(await blob.slice(offset, offset + CHUNK_SIZE).arrayBuffer());
            if (!bridge.writeFileChunk(base64Of(bytes))) {
              throw new Error('Android could not write the export file. Check available storage and try again.');
            }
          }
          bridge.finishFileSave();
        })().catch(fail);
      } else if (detail.type === 'complete') {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(true);
      } else if (detail.type === 'cancelled') {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(false);
      } else if (detail.type === 'error') {
        fail(new Error(detail.message || 'Android could not save the export.'));
      }
    };

    window.addEventListener(SAVE_EVENT, onSaveEvent);
    try {
      bridge.beginFileSave(filename, mimeType);
    } catch (error) {
      fail(error);
    }
  });
}

/** Save with Android's system document picker in the native app, or the
 *  browser download mechanism everywhere else. Resolves false if the user
 *  cancels the Android picker. */
export async function saveBlob(blob, filename, mimeType = 'application/octet-stream') {
  const bridge = globalThis.AndroidARBridge;
  if (bridge && typeof bridge.isAvailable === 'function' && bridge.isAvailable()) {
    if (typeof bridge.beginFileSave !== 'function'
      || typeof bridge.writeFileChunk !== 'function'
      || typeof bridge.finishFileSave !== 'function') {
      throw new Error('This app version cannot save files. Install the updated SplatXR app and try again.');
    }
    return nativeSave(blob, filename, mimeType, bridge);
  }

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
  return true;
}
