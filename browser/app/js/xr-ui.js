import { probeXR, captureXR } from './xr-capture.js';
import { validateDataset, filterDataset, cameraForFile, measuredSeeds, levelingNote } from './xr-dataset.js';
import { zipStore } from './zip.js';
import { saveBlob } from './download.js';

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function modal(html, className = '') {
  const dialog = document.createElement('dialog');
  dialog.className = `xr-dialog ${className}`;
  dialog.innerHTML = html;
  document.body.appendChild(dialog);
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => dialog.remove(), { once: true });
  dialog.showModal();
  return dialog;
}

export function showXRHelp(reason = '') {
  const native = !!(globalThis.AndroidARBridge && globalThis.AndroidARBridge.isAvailable?.());
  const guidance = native ? `
    <h3>Native ARCore capture</h3>
    <p>Use an ARCore-compatible Android phone with Google Play Services for AR installed and updated. Allow camera permission when prompted. If tracking does not start, close other camera apps and relaunch the capture screen.</p>
    <p>Move slowly around a stationary subject and capture at least three overlapping views (20–200 recommended). Manual shutter is the default; optionally enable distance-based auto-capture and edit its 2–200 cm spacing (10 cm default). The native coverage cloud adds observations only when a keyframe is captured; saved points trend from red to green as viewpoint diversity grows. CPU image resolution depends on the device's supported ARCore camera configuration.</p>
    <p>If Android WebView does not expose a WebGPU adapter, finish capture and export the dataset for training on a compatible desktop.</p>
  ` : `
    <h3>For pose-tagged AR photos</h3>
    <p>Use a supported Android device with Google Play Services for AR, in a Chromium browser exposing WebXR and <b>Raw Camera Access</b>. Browser/device support varies; immersive AR alone is not enough. Camera access is checked when you start the session.</p>
    <p>Open this app over HTTPS in a full browser tab and allow camera and spatial-tracking permissions. Embedded previews may block those permissions.</p>
    <p>The splat.js renderer assumes zero projection skew. Skewed XR cameras or imported calibrations are rejected explicitly rather than trained with a mismatched camera model.</p>
    <div class="xr-help-actions"><a class="btn btn-accent" href="${esc(location.href)}" target="_blank" rel="noopener">Open a capture tab ↗</a></div>
    <h3>Measured seeds, when available</h3>
    <p>Optional WebXR depth and hit tests provide colored 3D seed positions in meters. ARCore's native feature-point cloud is not directly exposed to the browser. If measured seeds are unavailable, the app labels the library's random-volume initialization clearly.</p>
  `;
  const dialog = modal(`
    <button class="xr-dialog-close" aria-label="Close device guide">×</button>
    <p class="eyebrow">DEVICE GUIDE</p><h2>${native ? 'ARCore tracks every view.' : 'A camera that knows where it is.'}</h2>
    <p>${native
      ? 'The Android app captures camera images, poses, intrinsics and RGB seed points natively. Known poses skip the camera solve—not splat training.'
      : 'AR capture pairs a photo with the exact WebXR camera pose from the same XR frame. It skips landmark matching and the camera solve—not splat training.'}</p>
    ${reason ? `<div class="xr-notice">${esc(reason)}</div>` : ''}
    ${guidance}
    <div class="xr-help-actions"><button class="btn btn-outline" data-close>Got it</button></div>
    <p class="xr-credit">Built on <a href="https://github.com/arrival-space/splat.js" target="_blank" rel="noopener">arrival-space/splat.js</a></p>
  `, 'xr-help');
  dialog.querySelector('.xr-dialog-close').onclick = () => dialog.close();
  dialog.querySelector('[data-close]').onclick = () => dialog.close();
}

export async function exportCapture(dataset) {
  validateDataset(dataset);
  const encode = (value) => new TextEncoder().encode(JSON.stringify(value, null, 2));
  const entries = [];
  for (const file of dataset.files) entries.push({ name: file.name, data: new Uint8Array(await file.arrayBuffer()) });
  if (dataset.transforms) entries.push({ name: dataset.transforms.frames ? 'transforms.json' : 'cameras.json', data: encode(dataset.transforms) });
  entries.push({ name: 'seed_points.json', data: encode({ points: dataset.points || [], pointsByFrame: dataset.pointsByFrame || null }) });
  entries.push({ name: 'capture.json', data: encode({ ...dataset.metadata, frames: dataset.files.length, coordinateConvention: 'WebXR local world, meters; transforms camera-to-world OpenGL; seeds in the same world', measuredSeeds: measuredSeeds(dataset).length }) });
  entries.push({ name: 'README.txt', data: new TextEncoder().encode('SplatXR capture dataset\n\nUnzip this archive. In SplatXR, choose all photos and the camera JSON together, plus seed_points.json and optionally capture.json. Known camera poses bypass the camera solve. Do not crop, rotate or mirror photos independently of their intrinsics. Measured seed points are optional; an orbit is needed for random-volume initialization if none are present. All capture files stay on your device unless you choose to share them.\n') });
  return saveBlob(zipStore(entries), `splatxr-capture-${Date.now()}.zip`, 'application/zip');
}

export function reviewCapture(dataset) {
  validateDataset(dataset);
  return new Promise((resolve) => {
    let selected = new Set(dataset.files.map((f) => f.name));
    let active = dataset.files[0].name;
    const urls = new Map(dataset.files.map((f) => [f.name, URL.createObjectURL(f)]));
    let result = null;
    const dialog = modal(`
      <button class="xr-dialog-close" aria-label="Close frame review">×</button>
      <p class="eyebrow">${dataset.transforms ? 'KNOWN-POSE CAPTURE' : 'PHOTO CAPTURE'}</p>
      <h2>Every view, accounted for.</h2>
      <p>Inspect your frames before training. Remove a blurred view or reset the whole set. Images and camera poses stay matched by filename.</p>
      ${dataset.metadata?.interrupted ? `<div class="xr-notice">Capture interrupted: ${esc(dataset.metadata.reason || 'AR session ended')}. These saved frames remain valid in one reference space. Start a new capture to take more; coordinate epochs are never mixed.</div>` : ''}
      <div class="xr-review-preview"><img alt="Selected captured frame"><div class="xr-review-pose"><span class="eyebrow">FRAME DATA</span><b data-name></b><pre></pre></div></div>
      <div class="xr-review-strip" aria-label="Captured frames"></div>
      <div class="xr-review-toolbar"><span data-count></span><button class="linkish" data-remove>Remove this frame</button><button class="linkish" data-restore>Restore all</button><button class="linkish" data-reset>Reset capture</button></div>
      <div class="xr-notice" data-seeds></div>
      <p class="xr-review-status" role="status" aria-live="polite"></p>
      <div class="xr-help-actions"><button class="btn btn-outline" data-export>Export capture ↓</button><button class="btn btn-outline" data-cancel>Cancel</button><button class="btn btn-accent btn-cta" data-use>Use these frames →</button></div>
    `, 'xr-review');
    const status = dialog.querySelector('.xr-review-status');
    const repaint = () => {
      const retained = filterDataset(dataset, selected);
      dialog.querySelector('[data-count]').textContent = `${selected.size} / ${dataset.files.length} frames selected`;
      dialog.querySelector('[data-use]').disabled = selected.size < 3;
      dialog.querySelector('[data-export]').disabled = selected.size < 3;
      dialog.querySelector('[data-remove]').disabled = !active || !selected.has(active);
      const img = dialog.querySelector('.xr-review-preview img');
      img.src = urls.get(active) || ''; img.hidden = !active;
      dialog.querySelector('[data-name]').textContent = active || 'No frame selected';
      const camera = active ? cameraForFile(dataset, active) : null;
      dialog.querySelector('pre').textContent = camera ? JSON.stringify(camera, null, 2) : 'No tracked pose.\nThe original camera solve will be used.';
      const seedCount = measuredSeeds(retained).length;
      const levelNote = levelingNote(dataset.metadata?.leveling);
      dialog.querySelector('[data-seeds]').textContent = (!dataset.transforms
        ? 'Standard photos → original camera solve → Gaussian seed → WebGPU training.'
        : `${selected.size} known cameras · camera solve skipped · ${seedCount ? `${seedCount.toLocaleString()} measured seed points (${dataset.metadata?.seedSource || 'provided'})` : 'no sufficient measured seeds; random-volume initialization (orbit required)'}. ${selected.size < 20 ? 'At least 20 overlapping views are recommended.' : 'Ready for direct pose input.'}`) + (levelNote ? ` ${levelNote}` : '');
      const strip = dialog.querySelector('.xr-review-strip'); strip.replaceChildren();
      for (const file of dataset.files) {
        const b = document.createElement('button'); b.type = 'button';
        b.className = `xr-review-thumb${file.name === active ? ' on' : ''}${selected.has(file.name) ? '' : ' removed'}`;
        b.title = `${file.name}${selected.has(file.name) ? '' : ' (removed)'}`;
        b.setAttribute('aria-label', `Inspect ${file.name}`);
        const th = new Image(); th.src = urls.get(file.name); th.alt = '';
        const label = document.createElement('span'); label.textContent = file.name.replace(/\.[^.]+$/, '');
        b.append(th, label); b.onclick = () => { active = file.name; repaint(); }; strip.append(b);
      }
    };
    dialog.querySelector('[data-remove]').onclick = () => { selected.delete(active); repaint(); };
    dialog.querySelector('[data-restore]').onclick = () => { selected = new Set(dataset.files.map((f) => f.name)); active = dataset.files[0].name; repaint(); };
    dialog.querySelector('[data-reset]').onclick = () => { selected.clear(); active = null; status.textContent = 'Selection reset. Restore all to recover it, or close and start a new capture.'; repaint(); };
    dialog.querySelector('[data-use]').onclick = () => {
      try { result = validateDataset(filterDataset(dataset, selected)); dialog.close(); }
      catch (e) { status.textContent = e.message; }
    };
    dialog.querySelector('[data-export]').onclick = async () => {
      try {
        const saved = await exportCapture(filterDataset(dataset, selected));
        status.textContent = saved
          ? 'Capture export saved. Unzip and select the photos and camera JSON together to import on another device.'
          : 'Export cancelled.';
      } catch (e) { status.textContent = `Export failed: ${e.message}`; }
    };
    dialog.querySelector('[data-cancel]').onclick = () => dialog.close();
    dialog.querySelector('.xr-dialog-close').onclick = () => dialog.close();
    dialog.addEventListener('close', () => { for (const url of urls.values()) URL.revokeObjectURL(url); resolve(result); }, { once: true });
    repaint();
  });
}

export function initXRUI({ onCapture, flash, setBusy }) {
  const button = $('btn-ar');
  const nativeBridge = globalThis.AndroidARBridge;
  const nativeAvailable = !!(nativeBridge && typeof nativeBridge.isAvailable === 'function' && nativeBridge.isAvailable());
  let capability = nativeAvailable
    ? { supported: true, reason: 'Native ARCore capture is available in this Android app.' }
    : { supported: false, reason: 'Checking WebXR support…' };
  $('btn-xr-help').onclick = () => showXRHelp(capability.reason);
  $('xr-device-guide').onclick = () => showXRHelp(capability.reason);
  $('xr-open-tab').href = location.href;
  let embedded = true;
  try { embedded = window.self !== window.top; } catch { /* cross-origin embedding */ }
  $('xr-open-tab').hidden = !embedded;

  const importNativeCapture = async () => {
    const payload = JSON.parse(nativeBridge.exportDataset());
    if (!payload.ready || payload.frameCount < 3) throw new Error(payload.error || 'Capture at least three frames before importing.');
    const metadataUri = 'https://appassets.androidplatform.net/captures/current/capture.json';
    const responses = await Promise.all([
      fetch(payload.transformsUri), fetch(payload.pointsUri), fetch(metadataUri),
    ]);
    if (responses.some((response) => !response.ok)) throw new Error('Native capture files could not be read from app storage.');
    const [transforms, pointDocument, metadata] = await Promise.all(responses.map((response) => response.json()));
    const files = await Promise.all(payload.frames.map(async (frame) => {
      const response = await fetch(frame.uri);
      if (!response.ok) throw new Error(`Could not read ${frame.name} from the native capture.`);
      return new File([await response.blob()], frame.name, { type: 'image/jpeg' });
    }));
    return validateDataset({ kind: 'ar', files, transforms, points: pointDocument.points || [],
      pointsByFrame: pointDocument.pointsByFrame || null,
      metadata: { ...metadata, ...pointDocument.metadata, nativeARCore: true } });
  };

  button.onclick = async () => {
    if (!capability.supported) { showXRHelp(capability.reason); return; }
    button.disabled = true;
    try {
      if (nativeAvailable) { nativeBridge.startCapture(); return; }
      // the browser capture resolves before any review modal exists: Start
      // queues behind the counter (the app's setBusy)
      setBusy?.(true);
      let dataset = null;
      try { dataset = await captureXR(); }
      finally { setBusy?.(false); }
      if (!dataset) return;
      const reviewed = await reviewCapture(dataset);
      if (reviewed) await onCapture(reviewed);
    } catch (e) {
      console.error('[SplatXR]', e);
      showXRHelp(e.message || 'AR capture could not start. Try the standard camera or choose photos.');
    } finally { button.disabled = false; }
  };

  if (nativeAvailable) {
    $('xr-ar-state').textContent = 'Native ARCore ready';
    $('xr-ar-state').dataset.ready = '1';
    button.querySelector('span').textContent = 'Capture with ARCore';
    $('xr-capability-note').textContent = 'Native ARCore capture pairs camera images with tracked poses and an RGB seed cloud.';
    window.addEventListener('splatxr-native-capture', async () => {
      // the import (and the review after it) land through open(): Start
      // queues behind the counter (the app's setBusy)
      setBusy?.(true);
      try {
        const dataset = await importNativeCapture();
        const reviewed = await reviewCapture(dataset);
        if (!reviewed) return;
        await onCapture(reviewed);
        // the frames now live in the app as Blobs (IndexedDB) — the native
        // copies are redundant, so drop them instead of leaving up to 200
        // JPEGs in app storage until the next capture. Except on
        // headless-capable bridges: the migrate-on-swipe retain copies them
        // into the run dir at Start (zero bytes over the bridge) and the
        // staging path resets the session once the copy matches.
        if (typeof nativeBridge.writeRunJson !== 'function') {
          try { nativeBridge.resetCaptureSession(); } catch { /* best-effort */ }
        }
      } catch (e) {
        console.error('[SplatXR] Native capture import failed:', e);
        showXRHelp(e.message || 'Native ARCore capture could not be imported.');
      } finally { setBusy?.(false); }
    });
  }

  probeXR().then((value) => {
    if (nativeAvailable) return;
    capability = value;
    $('xr-ar-state').textContent = value.supported ? 'AR session available' : 'AR not available here';
    $('xr-ar-state').dataset.ready = value.supported ? '1' : '0';
    button.querySelector('span').textContent = value.supported ? 'Capture with AR' : 'AR device guide';
    $('xr-capability-note').textContent = value.supported
      ? 'Immersive AR detected. Pose-tagged photos still require camera-access permission and Raw Camera Access support.'
      : value.reason;
  }).catch((e) => {
    if (nativeAvailable) return;
    capability.reason = e.message;
    $('xr-ar-state').textContent = 'AR unavailable';
    $('xr-capability-note').textContent = e.message;
  });
  const gpuState = $('xr-gpu-state');
  if (!navigator.gpu) { gpuState.textContent = 'WebGPU unavailable'; return; }
  navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).then((a) => {
    gpuState.textContent = a ? 'WebGPU ready' : 'WebGPU unavailable'; gpuState.dataset.ready = a ? '1' : '0';
  }).catch(() => { gpuState.textContent = 'WebGPU unavailable'; });
}
