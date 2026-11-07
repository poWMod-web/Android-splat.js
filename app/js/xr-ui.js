import { probeXR, captureXR } from './xr-capture.js';
import { validateDataset, filterDataset, cameraForFile, measuredSeeds } from './xr-dataset.js';
import { zipStore } from './zip.js';

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
  const dialog = modal(`
    <button class="xr-dialog-close" aria-label="Close device guide">×</button>
    <p class="eyebrow">DEVICE GUIDE</p><h2>A camera that knows where it is.</h2>
    <p>AR capture pairs a photo with the exact WebXR camera pose from the same XR frame. It skips the landmark matching and camera solve—not splat training.</p>
    ${reason ? `<div class="xr-notice">${esc(reason)}</div>` : ''}
    <h3>For pose-tagged AR photos</h3>
    <p>Use a supported Android device with Google Play Services for AR, in a Chromium browser exposing WebXR and <b>Raw Camera Access</b>. Browser/device support varies; immersive AR alone is not enough. Camera access is checked when you start the session.</p>
    <p>Open this app over HTTPS in a full browser tab and allow camera and spatial-tracking permissions. Embedded previews may block those permissions.</p>
    <p>The splat.js renderer assumes zero projection skew. Skewed XR cameras or imported calibrations are rejected explicitly rather than trained with a mismatched camera model.</p>
    <div class="xr-help-actions"><a class="btn btn-accent" href="${esc(location.href)}" target="_blank" rel="noopener">Open a capture tab ↗</a><button class="btn btn-outline" data-close>Got it</button></div>
    <h3>Measured seeds, when available</h3>
    <p>Optional WebXR depth and hit tests provide colored 3D seed positions in meters. ARCore's native feature-point cloud is not directly exposed. If measured seeds are unavailable, the app labels the library's random-volume initialization clearly. Walk an orbit around the subject so the cameras converge.</p>
    <h3>Capture on a phone. Train on a GPU.</h3>
    <p>AR capture does not require WebGPU. You can export the photos, camera transforms and seeds, then import them on a WebGPU-capable desktop. Unsupported devices can use regular photos or standard camera capture and the original splat.js solve.</p>
    <p class="xr-credit">Built on <a href="https://github.com/arrival-space/splat.js" target="_blank" rel="noopener">arrival-space/splat.js</a> · <a href="https://immersive-web.github.io/raw-camera-access/" target="_blank" rel="noopener">WebXR camera specification</a></p>
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
  const url = URL.createObjectURL(zipStore(entries));
  const a = document.createElement('a'); a.href = url; a.download = `splatxr-capture-${Date.now()}.zip`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
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
      dialog.querySelector('[data-seeds]').textContent = !dataset.transforms
        ? 'Standard photos → original camera solve → Gaussian seed → WebGPU training.'
        : `${selected.size} known cameras · camera solve skipped · ${seedCount ? `${seedCount.toLocaleString()} measured seed points (${dataset.metadata?.seedSource || 'provided'})` : 'no sufficient measured seeds; random-volume initialization (orbit required)'}. ${selected.size < 20 ? 'At least 20 overlapping views are recommended.' : 'Ready for direct pose input.'}`;
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
      try { await exportCapture(filterDataset(dataset, selected)); status.textContent = 'Capture exported. Unzip and select the photos and camera JSON together to import on another device.'; }
      catch (e) { status.textContent = `Export failed: ${e.message}`; }
    };
    dialog.querySelector('[data-cancel]').onclick = () => dialog.close();
    dialog.querySelector('.xr-dialog-close').onclick = () => dialog.close();
    dialog.addEventListener('close', () => { for (const url of urls.values()) URL.revokeObjectURL(url); resolve(result); }, { once: true });
    repaint();
  });
}

export function initXRUI({ onCapture, onExample, flash }) {
  const button = $('btn-ar');
  let capability = { supported: false, reason: 'Checking WebXR support…' };
  $('btn-xr-help').onclick = () => showXRHelp(capability.reason);
  $('xr-device-guide').onclick = () => showXRHelp(capability.reason);
  $('btn-xr-example').onclick = async () => {
    try { await onExample(); } catch (e) { flash(`Example unavailable: ${e.message}`, 8000); }
  };
  $('xr-open-tab').href = location.href;
  let embedded = true;
  try { embedded = window.self !== window.top; } catch { /* cross-origin embedding */ }
  $('xr-open-tab').hidden = !embedded;
  button.onclick = async () => {
    if (!capability.supported) { showXRHelp(capability.reason); return; }
    button.disabled = true;
    try {
      const dataset = await captureXR();
      if (!dataset) return;
      const reviewed = await reviewCapture(dataset);
      if (reviewed) await onCapture(reviewed);
    } catch (e) {
      console.error('[SplatXR]', e);
      showXRHelp(e.message || 'AR capture could not start. Try the standard camera or choose photos.');
    } finally { button.disabled = false; }
  };
  probeXR().then((value) => {
    capability = value;
    $('xr-ar-state').textContent = value.supported ? 'AR session available' : 'AR not available here';
    $('xr-ar-state').dataset.ready = value.supported ? '1' : '0';
    button.querySelector('span').textContent = value.supported ? 'Capture with AR' : 'AR device guide';
    $('xr-capability-note').textContent = value.supported
      ? 'Immersive AR detected. Pose-tagged photos still require camera-access permission and Raw Camera Access support.'
      : value.reason;
  }).catch((e) => { capability.reason = e.message; $('xr-ar-state').textContent = 'AR unavailable'; $('xr-capability-note').textContent = e.message; });
  const gpuState = $('xr-gpu-state');
  if (!navigator.gpu) { gpuState.textContent = 'WebGPU unavailable'; return; }
  navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).then((a) => {
    gpuState.textContent = a ? 'WebGPU ready' : 'WebGPU unavailable'; gpuState.dataset.ready = a ? '1' : '0';
  }).catch(() => { gpuState.textContent = 'WebGPU unavailable'; });
}
