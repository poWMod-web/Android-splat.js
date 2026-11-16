// xr-capture.js — ARCore/WebXR synchronized photo capture for SplatXR.
//
// The browser-owned XRView.camera texture and its XRView pose are consumed in
// one XR animation frame.  This intentionally does not use a separate media
// stream: media frames do not carry the same tracking timestamp or camera pose.

import {
  matrixRows,
  projectionIntrinsics,
  sampleDepthMeters,
  unprojectDepthSample,
  transformPoint,
  projectCameraPoint,
} from './xr-math.js';
import { DistanceCaptureTrigger, DEFAULT_CAPTURE_SPACING_M } from './xr-distance.js';

const REQUIRED_FEATURES = ['camera-access', 'local', 'dom-overlay'];
const OPTIONAL_FEATURES = ['hit-test', 'depth-sensing'];
const MAX_FRAMES = 200;
const MAX_POINTS = 30_000;
const MIN_FINISH_FRAMES = 3;
const MIN_CAPTURE_SPACING_CM = 2;
const MAX_CAPTURE_SPACING_CM = 200;
const CAPTURE_SPACING_KEY = 'splatxr-auto-capture-spacing-cm-v1';
const DEPTH_GRID_X = 16;
const DEPTH_GRID_Y = 12;
const VOXEL_SIZE_M = 0.02;

/**
 * Probe the broad immersive-AR capability without requesting a session.
 *
 * isSessionSupported('immersive-ar') cannot prove that camera-access will be
 * granted: raw camera access is an experimental, permission-policy-gated
 * feature and can still be rejected by requestSession().
 */
export async function probeXR() {
  if (typeof globalThis === 'undefined' || !globalThis.navigator) {
    return { supported: false, reason: 'WebXR is only available in a browser.' };
  }
  if (globalThis.isSecureContext === false) {
    return {
      supported: false,
      reason: 'WebXR camera-access requires HTTPS (or localhost) in a secure context.',
    };
  }
  if (!globalThis.navigator.xr || typeof globalThis.navigator.xr.isSessionSupported !== 'function') {
    return { supported: false, reason: 'This browser does not expose WebXR.' };
  }
  if (typeof globalThis.XRWebGLBinding !== 'function' || typeof globalThis.XRWebGLLayer !== 'function') {
    return {
      supported: false,
      reason: 'This browser lacks the WebXR raw-camera WebGL binding; camera-access is experimental.',
    };
  }
  if (!globalThis.document?.createElement) {
    return { supported: false, reason: 'A DOM document is required for the WebXR overlay.' };
  }
  const testCanvas = globalThis.document.createElement('canvas');
  const testGl = testCanvas.getContext?.('webgl2', { xrCompatible: true });
  if (!testGl) {
    return { supported: false, reason: 'WebXR photo capture requires a WebGL2 context.' };
  }
  try {
    const supported = await globalThis.navigator.xr.isSessionSupported('immersive-ar');
    if (!supported) {
      return {
        supported: false,
        reason: 'immersive-ar is not supported. camera-access is experimental and is not guaranteed even when AR is supported.',
      };
    }
    return {
      supported: true,
      reason: 'immersive-ar is available; camera-access still requires HTTPS, permissions, and a compatible top-level or permitted iframe context.',
    };
  } catch (error) {
    return {
      supported: false,
      reason: `WebXR support check failed: ${error?.message || String(error)}`,
    };
  } finally {
    testGl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}

/**
 * Start a synchronized WebXR photo capture.
 *
 * The method is designed to be called directly from a user-activation handler.
 * It returns null for cancellation, unsupported sessions, and interrupted
 * sessions that ended before three complete photos were encoded.
 */
export async function captureXR() {
  const nav = globalThis.navigator;
  const doc = globalThis.document;
  if (!nav?.xr || !doc?.createElement || globalThis.isSecureContext === false) {
    throw new Error('WebXR capture requires a compatible browser and a secure HTTPS context. Use the standard camera or choose photos instead.');
  }

  const canvas = doc.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    antialias: false,
    depth: false,
    preserveDrawingBuffer: false,
    xrCompatible: true,
  });
  if (!gl || typeof globalThis.XRWebGLLayer !== 'function' || typeof globalThis.XRWebGLBinding !== 'function') {
    throw new Error('This browser does not expose WebGL2 raw-camera WebXR bindings. Camera-access is experimental; use ordinary photos instead.');
  }

    const spacingCm = loadCaptureSpacingCm(doc);
    const overlay = createOverlay(doc, spacingCm);
  doc.body?.appendChild(overlay);
  let blitter = null;
  let session = null;
  let baseLayer = null;

  try {
    // Creating the context and shader synchronously before requestSession keeps
    // requestSession in the caller's user-activation task.
    blitter = createBlitter(gl);
    const sessionInit = {
      requiredFeatures: REQUIRED_FEATURES.slice(),
      optionalFeatures: OPTIONAL_FEATURES.slice(),
      domOverlay: { root: overlay },
      // These are consulted only if depth-sensing is granted.  Failure to
      // provide CPU depth must never make the photo capture unavailable.
      depthSensing: {
        usagePreference: ['cpu-optimized'],
        dataFormatPreference: ['float32', 'luminance-alpha'],
        matchDepthView: true,
      },
    };
    session = await nav.xr.requestSession('immersive-ar', sessionInit);
    baseLayer = new globalThis.XRWebGLLayer(session, gl, {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
    });
    session.updateRenderState({ baseLayer });

    const glBinding = new globalThis.XRWebGLBinding(session, gl);
    const referenceSpace = await session.requestReferenceSpace('local');
    let viewerSpace = null;
    let hitTestSource = null;
    try {
      viewerSpace = await session.requestReferenceSpace('viewer');
      if (typeof session.requestHitTestSource === 'function') {
        hitTestSource = await session.requestHitTestSource({ space: viewerSpace });
      }
    } catch {
      // hit-test is optional; the capture remains valid without measured hits.
      hitTestSource = null;
    }

    return await runCapture({
      doc,
      gl,
      canvas,
      overlay,
      session,
      baseLayer,
      glBinding,
      blitter,
      referenceSpace,
      hitTestSource,
      spacingCm,
    });
  } catch (error) {
    // requestSession commonly rejects here when camera-access is unsupported,
    // the document is insecure, or an iframe lacks the camera/xr-spatial-
    // tracking permissions policy. Surface that exact failure to the UI.
    console.warn('[SplatXR] WebXR camera-access capture unavailable:', error);
    try {
      await session?.end();
    } catch {
      // Already-ended sessions are harmless.
    }
    destroyCaptureResources({ gl, blitter, overlay, canvas });
    throw new Error(`AR capture could not start: ${error?.message || String(error)}. Required: camera-access, local tracking and DOM Overlay. Allow camera/spatial-tracking permissions in a full HTTPS browser tab, or use standard photos.`);
  }
}

function createOverlay(doc, spacingCm = DEFAULT_CAPTURE_SPACING_M * 100) {
  const root = doc.createElement('div');
  root.className = 'xr-overlay';
  root.setAttribute('role', 'dialog');
  root.setAttribute('aria-label', 'WebXR photo capture');
  root.innerHTML = `
    <div class="xr-topbar">
      <div class="xr-status" data-role="status" aria-live="polite">Establishing AR tracking…</div>
      <button class="xr-cancel" type="button" data-action="cancel" aria-label="Cancel AR capture">Cancel</button>
    </div>
    <div class="xr-telemetry" aria-live="polite">
      <span class="xr-count" data-role="count">0 photos</span>
      <span class="xr-pose" data-role="pose">Tracking: waiting</span>
      <span class="xr-seed" data-role="seed">Seeds: none measured</span>
    </div>
    <div class="xr-guide" data-role="guide">
      Manual capture is ready. Press Shutter for each photo, or check Auto-capture and press Start to take one every ${spacingCm} cm traveled.
      Move slowly and aim for 20 or more overlapping views. Rotating in place does not trigger automatic photos.
    </div>
    <div class="xr-controls">
      <label class="xr-auto">
        <span class="xr-auto-toggle"><input type="checkbox" data-action="auto"> Auto-capture</span>
        <span class="xr-auto-distance"><input type="number" data-action="distance" min="${MIN_CAPTURE_SPACING_CM}" max="${MAX_CAPTURE_SPACING_CM}" step="1" value="${spacingCm}" inputmode="numeric" aria-label="Auto-capture travel distance in centimeters"> cm</span>
      </label>
      <button class="xr-shutter" type="button" data-action="shutter" aria-label="Take photo">Shutter</button>
      <button class="xr-finish" type="button" data-action="finish" disabled>Finish (0/3)</button>
    </div>`;
  return root;
}

async function runCapture({
  doc,
  gl,
  canvas,
  overlay,
  session,
  baseLayer,
  glBinding,
  blitter,
  referenceSpace,
  hitTestSource,
  spacingCm,
}) {
  return new Promise((resolve, reject) => {
    const status = overlay.querySelector('[data-role="status"]');
    const count = overlay.querySelector('[data-role="count"]');
    const poseLabel = overlay.querySelector('[data-role="pose"]');
    const seedLabel = overlay.querySelector('[data-role="seed"]');
    const guide = overlay.querySelector('[data-role="guide"]');
    const autoInput = overlay.querySelector('[data-action="auto"]');
    const distanceInput = overlay.querySelector('[data-action="distance"]');
    const shutter = overlay.querySelector('[data-action="shutter"]');
    const finish = overlay.querySelector('[data-action="finish"]');
    const cancel = overlay.querySelector('[data-action="cancel"]');

    const state = {
      files: [],
      frames: [],
      points: [],
      pointsByFrame: {},
      frameSources: {},
      pending: new Set(),
      encodeChain: Promise.resolve(),
      voxelKeys: new Set(),
      nextFileNumber: 1,
      auto: false,
      autoRunning: false,
      autoFirstPending: false,
      manualRequests: 0,
      spacingCm,
      distanceTrigger: new DistanceCaptureTrigger(spacingCm / 100),
      stopped: false,
      finishRequested: false,
      cancelRequested: false,
      interrupted: false,
      stopReason: '',
      rafHandle: null,
      finalizePromise: null,
      stopPromise: null,
      resetListener: null,
      endListener: null,
      depthFrames: 0,
      hitFrames: 0,
      noneFrames: 0,
      failedEncodes: 0,
    };

    const setStatus = (text) => {
      if (status) status.textContent = text;
    };
    const visibleCount = () => state.frames.length + state.pending.size;
    const updateUi = () => {
      const n = visibleCount();
      if (count) count.textContent = `${n} photo${n === 1 ? '' : 's'}`;
      if (finish) {
        finish.disabled = state.stopped || n < MIN_FINISH_FRAMES;
        finish.textContent = n >= MIN_FINISH_FRAMES ? `Finish (${n})` : `Finish (${n}/3)`;
      }
      if (shutter) shutter.disabled = state.stopped || n >= MAX_FRAMES;
      if (seedLabel) {
        const measured = state.points.length;
        seedLabel.textContent = measured
          ? `Seeds: ${measured} measured`
          : 'Seeds: none measured (random fallback)';
      }
    };
    const setPoseLabel = (text) => {
      if (poseLabel) poseLabel.textContent = `Tracking: ${text}`;
    };

    // the shutter doubles as the auto run's Start/Stop while auto mode is on
    const updateShutter = () => {
      if (!shutter) return;
      const label = !state.auto ? 'Shutter' : state.autoRunning ? 'Stop' : 'Start';
      shutter.textContent = label;
      shutter.setAttribute('aria-label',
        !state.auto ? 'Take photo' : state.autoRunning ? 'Stop auto-capture' : 'Start auto-capture');
    };
    overlay.addEventListener('beforexrselect', (event) => event.preventDefault());
    autoInput?.addEventListener('change', () => {
      state.auto = !!autoInput.checked;
      state.autoRunning = false;   // checking arms the mode; the shutter starts it
      state.autoFirstPending = false;
      state.distanceTrigger.reset();
      updateShutter();
      setStatus(state.auto ? `Auto-capture armed: press Start, then move slowly for one photo every ${state.spacingCm} cm.` : 'Manual capture enabled. Press Shutter to take a photo.');
    });
    distanceInput?.addEventListener('change', () => {
      distanceInput.setCustomValidity('');
      const value = Number(distanceInput.value);
      if (!Number.isInteger(value) || value < MIN_CAPTURE_SPACING_CM || value > MAX_CAPTURE_SPACING_CM) {
        distanceInput.setCustomValidity(`Enter a whole distance from ${MIN_CAPTURE_SPACING_CM} to ${MAX_CAPTURE_SPACING_CM} cm.`);
        distanceInput.reportValidity();
        distanceInput.value = String(state.spacingCm);
        return;
      }
      distanceInput.setCustomValidity('');
      state.spacingCm = value;
      state.distanceTrigger.setSpacingMeters(value / 100);
      storeCaptureSpacingCm(doc, value);
      if (guide) guide.textContent = `Manual capture is ready. Press Shutter for each photo, or check Auto-capture and press Start to take one every ${value} cm traveled. Move slowly and aim for 20 or more overlapping views. Rotating in place does not trigger automatic photos.`;
      setStatus(state.auto ? `Auto-capture updated: one photo every ${value} cm traveled.` : `Auto-capture distance set to ${value} cm. Auto-capture is still off.`);
    });
    shutter?.addEventListener('click', () => {
      if (state.stopped || visibleCount() >= MAX_FRAMES) return;
      if (state.auto) {
        // auto mode: the shutter starts/stops the automatic run
        state.autoRunning = !state.autoRunning;
        state.autoFirstPending = state.autoRunning;
        state.distanceTrigger.reset();
        updateShutter();
        setStatus(state.autoRunning
          ? `Auto-capture started: taking the first photo now, then one every ${state.spacingCm} cm traveled. Move slowly.`
          : 'Auto-capture stopped. Press Start to resume, or uncheck Auto-capture for manual shots.');
        return;
      }
      state.manualRequests += 1;
      setStatus('Shutter armed; waiting for the next tracked camera frame…');
    });
    finish?.addEventListener('click', () => {
      if (visibleCount() < MIN_FINISH_FRAMES) {
        setStatus('Take at least 3 tracked photos before finishing; 20 or more overlapping photos are recommended.');
        return;
      }
      state.finishRequested = true;
      void stopAndResolve({ interrupted: false, reason: 'finish' });
    });
    cancel?.addEventListener('click', () => {
      state.cancelRequested = true;
      void stopAndResolve({ cancel: true, interrupted: false, reason: 'cancel' });
    });

    state.resetListener = () => {
      if (state.stopped) return;
      // A reset starts a new local-space epoch.  Do not transform old frames
      // into the new epoch or append new poses to them.
      state.interrupted = true;
      setStatus('Tracking reset; capture stopped so camera epochs are not mixed.');
      if (guide) guide.textContent = 'The local AR origin changed. Existing photos keep their old epoch; start a new capture for more photos.';
      void stopAndResolve({ interrupted: true, reason: 'reference-space-reset' });
    };
    referenceSpace.addEventListener?.('reset', state.resetListener);

    state.endListener = () => {
      if (!state.stopped) {
        state.stopped = true;
        state.interrupted = !state.finishRequested;
        state.stopReason = state.stopReason || 'session-ended';
      }
      void finalize({
        cancel: state.cancelRequested,
        interrupted: state.interrupted,
        reason: state.stopReason || 'session-ended',
      });
    };
    session.addEventListener('end', state.endListener, { once: true });

    const onXRFrame = (time, frame) => {
      if (state.stopped) return;
      state.rafHandle = session.requestAnimationFrame(onXRFrame);
      try {
        gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.disable(gl.SCISSOR_TEST);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const viewerPose = frame.getViewerPose(referenceSpace);
        if (!viewerPose) {
          state.distanceTrigger.breakContinuity();
          setPoseLabel('waiting for tracking');
          setStatus('Waiting for non-emulated AR tracking…');
          return;
        }
        if (viewerPose.emulatedPosition) {
          state.distanceTrigger.breakContinuity();
          setPoseLabel('position emulated — photo rejected');
          setStatus('AR position is emulated; move until tracking is actively established.');
          return;
        }
        const viewIndex = viewerPose.views.findIndex((candidate) => candidate.camera);
        const view = viewerPose.views[viewIndex >= 0 ? viewIndex : 0];
        if (!view?.camera) {
          state.distanceTrigger.breakContinuity();
          setPoseLabel('pose tracked; camera image unavailable');
          setStatus('The XR pose is tracked, but camera-access has not supplied an image for this frame.');
          return;
        }

        const transform = view.transform;
        const position = [transform.position.x, transform.position.y, transform.position.z];
        const orientation = [
          transform.orientation.x,
          transform.orientation.y,
          transform.orientation.z,
          transform.orientation.w,
        ];
        setPoseLabel(`active · ${position.map((value) => value.toFixed(2)).join(', ')} m`);

        const manual = state.manualRequests > 0;
        // Start captures the first synchronized tracked frame immediately;
        // distance spacing begins only after that frame is actually queued.
        const firstAuto = state.auto && state.autoRunning && state.autoFirstPending;
        const auto = state.auto && state.autoRunning &&
          (firstAuto || state.distanceTrigger.sample(position));
        if (!manual && !auto) return;
        // PNG encoding can lag a mobile GPU; do not accumulate full-size raw frames.
        if (state.pending.size >= 2) { setStatus('Saving photos… hold steady, then keep moving.'); return; }
        if (visibleCount() >= MAX_FRAMES) {
          setStatus('Frame limit reached; finish the capture.');
          return;
        }
        if (manual) state.manualRequests -= 1;

        const snapshot = readSynchronizedFrame({
          frame,
          view,
          viewIndex: viewIndex >= 0 ? viewIndex : 0,
          time,
          position,
          orientation,
          gl,
          glBinding,
          blitter,
          hitTestSource,
          referenceSpace,
        });
        if (!snapshot) {
          setStatus('This frame had no readable camera texture; it was not captured.');
          return;
        }
        if (state.auto && state.autoRunning) {
          state.autoFirstPending = false;
          state.distanceTrigger.captured(position);
        }
        scheduleEncode(snapshot);
        updateUi();
        setStatus(manual ? 'Photo queued.' : 'Auto-captured a tracked frame. Keep moving slowly.');
      } catch (error) {
        // A lost pose or a transient opaque-texture error invalidates this
        // frame only.  Never cache or replay it after tracking is lost.
        setStatus(`Frame skipped: ${error?.message || 'temporary XR camera error'}`);
        if (error?.code === 'XR_UNSUPPORTED_SKEW') {
          state.interrupted = true;
          void stopAndResolve({ interrupted: true, reason: 'unsupported-projection-skew' });
        }
      }
    };

    function scheduleEncode(snapshot) {
      const fileName = `xr_${String(state.nextFileNumber++).padStart(4, '0')}.png`;
      const job = state.encodeChain.then(async () => {
        const blob = await encodePng(doc, snapshot.pixels, snapshot.width, snapshot.height);
        const file = new File([blob], fileName, { type: 'image/png' });
        const frame = { ...snapshot.frame, file_path: fileName };
        state.files.push(file);
        state.frames.push(frame);
        state.pointsByFrame[fileName] = snapshot.points;
        state.frameSources[fileName] = snapshot.source;
        addGlobalPoints(snapshot.points);
        if (snapshot.source === 'depth') state.depthFrames += 1;
        else if (snapshot.source === 'hit-test') state.hitFrames += 1;
        else state.noneFrames += 1;
        updateUi();
      }).catch((error) => {
        state.failedEncodes += 1;
        setStatus(`Photo encoding failed; the frame was discarded (${error?.message || 'unknown error'}).`);
        updateUi();
      });
      state.encodeChain = job;
      state.pending.add(job);
      const finished = () => { state.pending.delete(job); updateUi(); };
      job.then(finished, finished);
    }

    function addGlobalPoints(points) {
      for (const point of points) {
        if (state.points.length >= MAX_POINTS) break;
        const key = point.X.map((value) => Math.floor(value / VOXEL_SIZE_M)).join(':');
        if (state.voxelKeys.has(key)) continue;
        state.voxelKeys.add(key);
        state.points.push(point);
      }
    }

    async function stopAndResolve({ cancel = false, interrupted = false, reason = 'stop' } = {}) {
      if (state.stopPromise) return state.stopPromise;
      state.stopped = true;
      state.cancelRequested ||= cancel;
      state.interrupted ||= interrupted;
      state.stopReason = reason;
      if (state.rafHandle != null && typeof session.cancelAnimationFrame === 'function') {
        try { session.cancelAnimationFrame(state.rafHandle); } catch { /* session may already be ending */ }
      }
      state.stopPromise = (async () => {
        try {
          await session.end();
        } catch {
          // The end event or an already-ended session still gets finalized.
        }
        await finalize({
          cancel: state.cancelRequested,
          interrupted: state.interrupted,
          reason: state.stopReason,
        });
      })();
      return state.stopPromise;
    }

    async function finalize({ cancel = false, interrupted = false, reason = 'session-ended' } = {}) {
      if (state.finalizePromise) return state.finalizePromise;
      state.finalizePromise = (async () => {
        state.stopped = true;
        state.interrupted ||= interrupted;
        state.stopReason = state.stopReason || reason;
        await Promise.all([...state.pending]);
        cleanup();
        if (cancel || state.cancelRequested) {
          resolve(null);
          return;
        }
        if (state.frames.length < MIN_FINISH_FRAMES) {
          reject(new Error(`Capture ended with ${state.frames.length} encoded photos (${state.stopReason}). At least 3 are needed; start a new capture and aim for 20 overlapping views.`));
          return;
        }
        const seedSource = seedSourceForSources(state.frameSources);
        resolve({
          kind: 'ar',
          files: state.files.slice(),
          transforms: buildTransforms(state.frames),
          points: state.points.slice(0, MAX_POINTS),
          pointsByFrame: { ...state.pointsByFrame },
          metadata: {
            referenceSpace: 'local',
            referenceSpaceEpoch: 0,
            interrupted: !!state.interrupted,
            reason: state.stopReason,
            cameraAccess: 'camera-access',
            poseSource: 'XRView.transform from the same XRFrame as each image',
            imageOrigin: 'top-left',
            frameCount: state.frames.length,
            maxFrames: MAX_FRAMES,
            failedEncodes: state.failedEncodes,
            seedSource,
            pointCount: state.points.length,
            pointsByFrame: { ...state.pointsByFrame },
            frameSources: { ...state.frameSources },
            depthFrames: state.depthFrames,
            hitTestFrames: state.hitFrames,
            unmeasuredFrames: state.noneFrames,
            randomSeedFallback: state.points.length < 8,
            seedNote: state.points.length < 8
              ? 'Fewer than 8 measured points; keep the library random-volume fallback enabled.'
              : 'Measured points are available for known-camera seeding.',
          },
        });
      })();
      return state.finalizePromise;
    }

    function cleanup() {
      referenceSpace.removeEventListener?.('reset', state.resetListener);
      session.removeEventListener?.('end', state.endListener);
      if (hitTestSource?.cancel) {
        try { hitTestSource.cancel(); } catch { /* already cancelled */ }
      }
      destroyCaptureResources({ gl, blitter, overlay, canvas });
    }

    updateUi();
    setStatus('Move slowly until position tracking is active.');
    state.rafHandle = session.requestAnimationFrame(onXRFrame);
  });
}

function readSynchronizedFrame({
  frame,
  view,
  viewIndex,
  time,
  position,
  orientation,
  gl,
  glBinding,
  blitter,
  hitTestSource,
  referenceSpace,
}) {
  const camera = view.camera;
  const width = Number(camera.width);
  const height = Number(camera.height);
  if (!(width > 0) || !(height > 0)) return null;

  // getCameraImage(view.camera) is called while `frame` is the active XR
  // animation frame.  The pose, projection, and texture therefore share the
  // same tracking instant.
  const cameraTexture = glBinding.getCameraImage(camera);
  if (!cameraTexture) return null;
  const pixels = blitter.copy(cameraTexture, width, height);
  const projectionMatrix = Array.from(view.projectionMatrix);
  const intrinsics = projectionIntrinsics(projectionMatrix, width, height, { origin: 'top-left' });
  if (Math.abs(intrinsics.skew) > 1e-4 || Math.abs(projectionMatrix[4]) > 1e-7) {
    const error = new Error('This XR camera has nonzero projection skew, unsupported by the splat.js renderer. Use standard photo capture or a zero-skew camera instead.');
    error.code = 'XR_UNSUPPORTED_SKEW';
    throw error;
  }
  const cameraToWorld = matrixRows(view.transform.matrix);
  const measurements = collectMeasurements({
    frame,
    view,
    projectionMatrix,
    cameraToWorld,
    pixels,
    width,
    height,
    hitTestSource,
    referenceSpace,
  });

  return {
    pixels,
    width,
    height,
    points: measurements.points,
    source: measurements.source,
    frame: {
      // NeRF/OpenGL camera-to-world, nested row-major.  view.transform.matrix
      // is column-major and is copied rather than retained from a live XRView.
      transform_matrix: cameraToWorld,
      fl_x: intrinsics.fl_x,
      fl_y: intrinsics.fl_y,
      cx: intrinsics.cx,
      cy: intrinsics.cy,
      skew: intrinsics.skew,
      w: width,
      h: height,
      timestamp: Number(time),
      position: position.slice(),
      orientation: orientation.slice(),
      projection_matrix: projectionMatrix,
      view_index: viewIndex,
      image_origin: 'top-left',
    },
  };
}

function collectMeasurements({
  frame,
  view,
  projectionMatrix,
  cameraToWorld,
  pixels,
  width,
  height,
  hitTestSource,
  referenceSpace,
}) {
  let depthInfo = null;
  if (typeof frame.getDepthInformation === 'function') {
    try {
      depthInfo = frame.getDepthInformation(view);
    } catch {
      depthInfo = null;
    }
  }

  const points = [];
  if (depthInfo) {
    for (let gy = 0; gy < DEPTH_GRID_Y; gy++) {
      const y = (gy + 0.5) / DEPTH_GRID_Y;
      for (let gx = 0; gx < DEPTH_GRID_X; gx++) {
        const x = (gx + 0.5) / DEPTH_GRID_X;
        const depthMeters = sampleDepthMeters(depthInfo, x, y);
        if (!(depthMeters > 0)) continue;
        const world = unprojectDepthSample({
          projectionMatrix,
          width,
          height,
          x,
          y,
          depthMeters,
          cameraToWorld,
        });
        if (!isFinitePoint(world)) continue;
        points.push({ X: world, rgb: sampleRgb(pixels, width, height, x, y) });
      }
    }
  }
  if (points.length) return { points, source: 'depth' };

  // Hit-test is only a fallback measurement.  It is never used to invent a
  // cloud: a point is emitted only when the runtime supplies an actual pose.
  if (hitTestSource && typeof frame.getHitTestResults === 'function') {
    try {
      const hit = frame.getHitTestResults(hitTestSource)?.[0];
      const hitPose = hit?.getPose?.(referenceSpace);
      const p = hitPose?.transform?.position;
      if (p && hitPose.emulatedPosition !== true) {
        const world = [Number(p.x), Number(p.y), Number(p.z)];
        if (isFinitePoint(world)) {
          const cameraPoint = transformPoint(view.transform.inverse.matrix, world);
          const uv = projectCameraPoint(projectionMatrix, cameraPoint);
          if (cameraPoint[2] < 0 && uv.x >= 0 && uv.x <= 1 && uv.y >= 0 && uv.y <= 1) {
            return { points: [{ X: world, rgb: sampleRgb(pixels, width, height, uv.x, uv.y) }], source: 'hit-test' };
          }
        }
      }
    } catch {
      // Optional hit-test data is allowed to disappear between frames.
    }
  }
  return { points: [], source: 'none' };
}

function sampleRgb(pixels, width, height, normalizedX, normalizedY) {
  const x = Math.max(0, Math.min(width - 1, Math.round(normalizedX * (width - 1))));
  const y = Math.max(0, Math.min(height - 1, Math.round(normalizedY * (height - 1))));
  const i = (y * width + x) * 4;
  return [pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255];
}

function seedSourceForSources(frameSources) {
  const hasDepth = stateSourceCount('depth');
  const hasHit = stateSourceCount('hit-test');
  if (hasDepth && hasHit) return 'mixed';
  if (hasDepth) return 'depth';
  if (hasHit) return 'hit-test';
  return 'none';

  function stateSourceCount(source) {
    return Object.values(frameSources || {}).some((value) => value === source);
  }
}

function buildTransforms(frames) {
  const first = frames[0] || {};
  return {
    camera_model: 'PINHOLE',
    coordinate_system: 'NeRF/OpenGL camera-to-world in local XR space',
    image_origin: 'top-left',
    fl_x: first.fl_x,
    fl_y: first.fl_y,
    cx: first.cx,
    cy: first.cy,
    w: first.w,
    h: first.h,
    frames: frames.map((frame) => ({ ...frame })),
  };
}

function createBlitter(gl) {
  const vertexSource = `#version 300 es
    in vec2 aPosition;
    in vec2 aUv;
    out vec2 vUv;
    void main() {
      gl_Position = vec4(aPosition, 0.0, 1.0);
      vUv = aUv;
    }`;
  const fragmentSource = `#version 300 es
    precision highp float;
    uniform sampler2D uCamera;
    in vec2 vUv;
    out vec4 outColor;
    void main() {
      outColor = texture(uCamera, vUv);
    }`;
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || 'camera copy shader link failed');
  }

  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  // Texture coordinates intentionally use the camera texture's native
  // bottom-left origin.  readPixels is then flipped row-wise below, producing
  // a top-left PNG without a mirror or crop.
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 0, 0,
     1, -1, 1, 0,
    -1,  1, 0, 1,
     1,  1, 1, 1,
  ]), gl.STATIC_DRAW);
  const positionLocation = gl.getAttribLocation(program, 'aPosition');
  const uvLocation = gl.getAttribLocation(program, 'aUv');
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(uvLocation);
  gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 16, 8);
  gl.bindVertexArray(null);

  const samplerLocation = gl.getUniformLocation(program, 'uCamera');
  const framebuffer = gl.createFramebuffer();
  let targetTexture = null;
  let targetWidth = 0;
  let targetHeight = 0;

  const ensureTarget = (width, height) => {
    if (targetTexture && targetWidth === width && targetHeight === height) return;
    if (targetTexture) gl.deleteTexture(targetTexture);
    targetTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, targetTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('camera readback framebuffer is incomplete');
    }
    targetWidth = width;
    targetHeight = height;
  };

  return {
    copy(cameraTexture, width, height) {
      ensureTarget(width, height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.viewport(0, 0, width, height);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.SCISSOR_TEST);
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, cameraTexture);
      gl.uniform1i(samplerLocation, 0);
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      const raw = new Uint8Array(width * height * 4);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
      const error = gl.getError();
      gl.bindVertexArray(null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      if (error !== gl.NO_ERROR) throw new Error(`camera readback WebGL error 0x${error.toString(16)}`);
      flipRowsInPlace(raw, width, height);
      return raw;
    },
    destroy() {
      if (targetTexture) gl.deleteTexture(targetTexture);
      gl.deleteFramebuffer(framebuffer);
      gl.deleteBuffer(vbo);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      targetTexture = null;
    },
  };
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'camera copy shader compile failed';
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function flipRowsInPlace(bytes, width, height) {
  const rowBytes = width * 4;
  const row = new Uint8Array(rowBytes);
  for (let top = 0, bottom = height - 1; top < bottom; top++, bottom--) {
    const topOffset = top * rowBytes;
    const bottomOffset = bottom * rowBytes;
    row.set(bytes.subarray(topOffset, topOffset + rowBytes));
    bytes.copyWithin(topOffset, bottomOffset, bottomOffset + rowBytes);
    bytes.set(row, bottomOffset);
  }
}

async function encodePng(doc, pixels, width, height) {
  const imageData = typeof globalThis.ImageData === 'function'
    ? new globalThis.ImageData(new Uint8ClampedArray(pixels), width, height)
    : null;
  if (typeof globalThis.OffscreenCanvas === 'function') {
    const canvas = new globalThis.OffscreenCanvas(width, height);
    const context = canvas.getContext('2d', { alpha: true });
    if (!context) throw new Error('2D canvas unavailable for PNG encoding');
    context.putImageData(imageData || makeImageData(context, pixels, width, height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }
  const canvas = doc.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { alpha: true });
  if (!context) throw new Error('2D canvas unavailable for PNG encoding');
  context.putImageData(imageData || makeImageData(context, pixels, width, height), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('PNG encoding returned no blob')), 'image/png');
  });
}

function makeImageData(context, pixels, width, height) {
  const data = context.createImageData(width, height);
  data.data.set(pixels);
  return data;
}

function destroyCaptureResources({ gl, blitter, overlay, canvas }) {
  try { blitter?.destroy(); } catch { /* best-effort cleanup */ }
  try {
    gl?.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl?.useProgram(null);
  } catch { /* context may already be lost */ }
  try { overlay?.remove(); } catch { /* already removed */ }
  try {
    if (canvas) {
      canvas.width = 1;
      canvas.height = 1;
    }
  } catch { /* best-effort cleanup */ }
  try { gl?.getExtension('WEBGL_lose_context')?.loseContext(); } catch { /* optional extension */ }
}

function isFinitePoint(point) {
  return Array.isArray(point) && point.length >= 3 && point.every((value) => Number.isFinite(value));
}

function loadCaptureSpacingCm(doc) {
  try {
    const value = Number(doc.defaultView?.localStorage?.getItem(CAPTURE_SPACING_KEY));
    if (Number.isInteger(value) && value >= MIN_CAPTURE_SPACING_CM && value <= MAX_CAPTURE_SPACING_CM) return value;
  } catch { /* Storage can be unavailable in private or restricted contexts. */ }
  return Math.round(DEFAULT_CAPTURE_SPACING_M * 100);
}

function storeCaptureSpacingCm(doc, value) {
  try { doc.defaultView?.localStorage?.setItem(CAPTURE_SPACING_KEY, String(value)); }
  catch { /* A spacing preference is optional; capture itself must continue. */ }
}
