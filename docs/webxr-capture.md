# SplatXR WebXR photo capture

`app/js/xr-capture.js` exports `probeXR()` and `captureXR()`. The module is a raw-camera WebXR capture path for ARCore-backed browsers. It intentionally does **not** use `getUserMedia`: each PNG is copied from `XRView.camera` by `XRWebGLBinding.getCameraImage(view.camera)` in the same `XRFrame` used for `XRFrame.getViewerPose(localReferenceSpace)`, so the image and pose share a tracking instant.

Raw camera access is still experimental. `probeXR()` can confirm broad `immersive-ar` support, but it cannot promise that a later required `camera-access` request will be granted. A capture must run from a user-activation handler in a secure HTTPS context (localhost is suitable for development). A top-level tab is the least restricted deployment. An iframe also needs the browser's WebXR and camera permissions policy, including `xr-spatial-tracking` and `camera`, and browser/device support for the raw-camera module.

## Public exports

```js
import { captureXR, probeXR } from './js/xr-capture.js';

const capability = await probeXR();
// { supported: boolean, reason: string }

const result = await captureXR();
// null for cancel; setup failures or interruption before 3 encoded photos throw for UI guidance
// otherwise { kind, files, transforms, points, pointsByFrame, metadata }
```

`captureXR()` requests an `immersive-ar` session with these **required** features:

```js
requiredFeatures: ['camera-access', 'local', 'dom-overlay']
```

`hit-test` and `depth-sensing` are requested only as optional features. The `depthSensing` dictionary sets `usagePreference: ['cpu-optimized']`, `dataFormatPreference: ['float32', 'luminance-alpha']` and `matchDepthView: true`; missing optional depth never blocks photo capture. `captureXR()` creates a WebGL2 context with `{ xrCompatible: true }`, an `XRWebGLLayer`, and an `XRWebGLBinding` before reading the camera texture.

## Overlay hooks

The module creates and removes its own DOM overlay. Every class starts with `xr-` so the application can style it without depending on IDs or changing the capture logic. The parent application supplies CSS; the module supplies accessible text and controls.

- `.xr-overlay`: required `domOverlay.root`, full-screen capture shell.
- `.xr-topbar`: status/cancel row.
- `.xr-status`: live tracking, capture, and failure guidance.
- `.xr-cancel`: cancel button; cancellation resolves `null`.
- `.xr-telemetry`: telemetry row.
- `.xr-count`: encoded plus pending photo count.
- `.xr-pose`: current pose/tracking status and local-space position.
- `.xr-seed`: measured seed count or the explicit random-fallback label.
- `.xr-guide`: movement guidance and local-reference reset explanation.
- `.xr-controls`: bottom control row.
- `.xr-auto`: auto-capture label and checkbox.
- `.xr-shutter`: manual shutter; it arms the next valid tracked XR frame.
- `.xr-finish`: finish button, enabled at three or more pending/encoded photos.

The module also cancels `beforexrselect` inside the overlay so a DOM button does not create a duplicate XR select action. There is no app timer and no camera stream to stop.

AR capture starts with auto-capture disabled. Checking the box arms auto-capture (fresh position baseline); the shutter then starts/stops the run — while started it requests one image after each 10 cm of cumulative tracked-camera travel by default. The overlay distance field accepts whole centimeters from 2–200; changing the threshold resets the accumulated path and saves the preference locally for later sessions. Auto-capture does not use time intervals, and rotation in place does not trigger photos. `app/js/xr-distance.js` accumulates the measured path rather than net displacement, filters sub-2 mm jitter, and does not count unobserved jumps across tracking gaps. Successful manual or automatic photos reset the distance since the last capture. Manual shutter requests are consumed only by a valid frame. The UI recommends 20 or more overlapping photos, while finish accepts three or more and caps a session at 200 frames.

## Image and camera conventions

The camera texture is sampled into an application-owned `RGBA8` framebuffer. The module then calls `readPixels()` and flips rows in place because WebGL readback is bottom-up. The encoded PNG is top-left origin, keeps the complete `view.camera.width × view.camera.height` aspect ratio, and is never cropped or mirrored. The opaque user-agent camera texture remains read-only and is not deleted by the app.

Projection intrinsics are calculated from the full camera texture dimensions, not from the `XRWebGLLayer` viewport. For an asymmetric WebXR projection matrix, the module records `fl_x`, `fl_y`, `cx`, `cy`, and the source projection matrix. `cy` is converted to the top-left PNG convention. The pure helpers calculate skew, but splat.js's renderer assumes zero skew: nonzero-skew XR views stop capture, and imported skewed camera definitions are rejected explicitly. This avoids silently training with a mismatched projection. The helpers are tested with portrait dimensions and non-zero asymmetric principal-point terms.

Each transform frame has this shape:

```js
{
  file_path: 'xr_0001.png',       // exact File.name
  transform_matrix: [             // nested row-major NeRF/OpenGL camera-to-world
    [m00, m01, m02, m03],
    [m10, m11, m12, m13],
    [m20, m21, m22, m23],
    [m30, m31, m32, m33]
  ],
  fl_x, fl_y, cx, cy, w, h,
  timestamp,
  position: [x, y, z],
  orientation: [x, y, z, w],
  projection_matrix: [/* original WebXR column-major 16 values */],
  view_index,
  image_origin: 'top-left'
}
```

`transforms` is directly compatible with the repository's `parseCameras()` NeRF/Blender path. It includes `camera_model: 'PINHOLE'`, local XR-space coordinate metadata, full-resolution top-level intrinsics from the first frame, and `frames[]`. The frame transform is copied from `view.transform.matrix` by converting WebXR column-major values to nested row-major rows; no camera solve is needed before `useKnownCameras()`.

## Measured seed points

If CPU depth is available, the module samples a bounded 16×12 normalized-view grid. It calls `depthInfo.getDepthInMeters(x, y)` with the WebXR-defined normalized view coordinates (origin top-left), unprojects each valid camera-plane depth with the same view projection matrix, transforms it through that frame's camera-to-world pose, and samples RGB from the corresponding PNG pixel. Invalid depth values are discarded. Actual points are voxel-filtered at approximately 2 cm and capped at 30,000 points.

If a frame has no valid CPU depth, an optional hit-test result may contribute one point from its actual tracked surface pose. The measured world point is projected back through the same camera to sample its color accurately, including asymmetric principal points. No geometry is invented. If neither measurement is available, that frame contributes no point. A result includes:

```js
points: [{ X: [x, y, z], rgb: [r, g, b] }],
pointsByFrame: {
  'xr_0001.png': [{ X: [x, y, z], rgb: [r, g, b] }],
  'xr_0002.png': []
}
```

`pointsByFrame` uses exact filenames, allowing the parent to filter seeds when dropping a photo. It is repeated as `metadata.pointsByFrame` and exported explicitly in `seed_points.json`; import restores either source for round-trip ownership. Owned points are validated, and review filtering recomputes the bounded voxel cloud using only retained frames. `metadata.seedSource` is one of `depth`, `hit-test`, `mixed`, or `none`; `metadata.frameSources` gives the per-file source. When fewer than eight measured points survive, `metadata.randomSeedFallback` is `true`. A `none` source is an honest empty seed cloud, not a synthetic cloud.

## Tracking and interruption rules

Only viewer poses with `pose.emulatedPosition === false` are captured. A null pose, an emulated pose, or a frame without `view.camera` is skipped and never cached for later reuse. `getCameraImage`, texture sampling, the pose transform, projection matrix, optional depth, and optional hit test all belong to the same active XR animation frame.

The module listens for `local` reference-space `reset`. A reset begins a new coordinate epoch, so capture stops rather than combining old local-space points with new local-space poses. A session interruption/end with at least three encoded photos returns those captures with `metadata.interrupted: true`; fewer images produce an explicit error, while cancel returns `null`. A normal finish sets `metadata.interrupted: false`. The module waits for the bounded PNG encode queue before resolving and removes the overlay, XR animation frame, hit-test source, framebuffer, textures, shaders, and WebGL buffers during cleanup.

## Research references

- [WebXR Raw Camera Access Module](https://immersive-web.github.io/raw-camera-access/) — required `camera-access`, `XRView.camera`, aligned camera dimensions/frustum, and `XRWebGLBinding.getCameraImage()` lifetime rules.
- [WebXR camera-access sample](https://chromium.googlesource.com/chromium/src/+/7c5ac3c0f95b97cf12be95a5c1c0a8ff163246d8/third_party/webxr_test_pages/webxr-samples/proposals/camera-access-barebones.html) — projection-to-intrinsics equations and camera-texture readback behavior.
- [WebXR Depth Sensing Module](https://www.w3.org/TR/webxr-depth-sensing-1/) — normalized view coordinates, `getDepthInMeters()`, raw-depth scale, and camera-plane distance semantics.
- [WebXR Device API](https://www.w3.org/TR/webxr/) — local reference spaces, `emulatedPosition`, and reset behavior.
- [WebXR DOM Overlays Module](https://www.w3.org/TR/webxr-dom-overlays-1/) — required overlay root and `dom-overlay` session configuration.
