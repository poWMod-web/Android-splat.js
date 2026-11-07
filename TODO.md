# SplatXR implementation outcomes

## 1. Recreate the JavaScript splat.js browser workflow
- [x] Recreate the existing splat.js browser application and its core interaction flow using the `arrival-space/splat.js` project.
- [x] Preserve the existing in-browser JavaScript 3D Gaussian Splatting training functionality, including progress feedback, controls, and trained scene preview.

## 2. WebXR AR capture and known-pose training input
- [x] Detect WebXR `immersive-ar` support and expose AR capture only on compatible browsers with the necessary raw-camera bindings. ARCore-backed Android Chromium is the primary target; required `camera-access` and `dom-overlay` are checked at session start, not falsely promised by the broad AR probe.
- [x] Start and manage an immersive WebXR AR session with live camera-aligned capture guidance and session controls.
- [x] Capture a sequence of camera images and associate every image with its corresponding WebXR camera pose, including position and orientation in a consistent reference space.
- [x] Build a training-ready capture dataset that preserves image-to-pose relationships and converts WebXR transforms into the coordinate conventions and input format expected by splat.js.
- [x] Feed known camera poses directly into the splat.js training pipeline so camera positioning estimation or solving can be skipped.

## 3. Unsupported-device fallback, capture review, and resilience
- [x] Provide a browser-compatible fallback for unsupported devices using standard image selection or camera capture and the original splat.js camera-positioning workflow.
- [x] Handle AR permissions, unsupported capabilities, interrupted sessions, insufficient captures, and capture/training errors with clear status messaging.
- [x] Allow users to inspect the captured frame set and remove or reset captures before beginning training.

## 4. Distance-driven AR auto-capture
- [x] Start an AR capture with auto-capture disabled.
- [x] When auto-capture is enabled, trigger photos based on distance traveled by the tracked camera. Do not use a time interval or rotation-only triggers.
- [x] Make the auto-capture travel distance editable in the AR overlay, defaulting to 10 cm, and apply the entered spacing to the distance-based trigger.

## 5. Custom training settings
- [x] Allow a custom user-entered number of training cycles, in addition to existing preset values.
- [x] Keep LOD Off by default; when enabled, allow the user to choose a custom number of intermediate LOD detail snapshots, while including the final full-detail model and respecting splat-budget limits.

## 6. Android install and project export
- [x] Provide an installable, app-like Android experience with the existing web app, preserving browser WebGPU/WebXR capabilities where the user's Android browser and device support them.
- [x] Export the project as a ZIP archive the user can download.

## Implementation evidence and material limits

The app preserves the upstream JavaScript engine and viewer. Known-camera input calls `useKnownCameras()` instead of `solve()`, with fixed pose/pixel-aspect training options. Captures and seed ownership persist in IndexedDB and survive ZIP export/import. The read-only integration review found seed-ownership round-trip and unsupported-skew issues; both were fixed with regression coverage. Nonzero camera skew is now explicitly rejected because the upstream renderer assumes zero skew.

The unit suite passes 11 tests, skips one upstream optional SIFT dependency/reference-data test, and has zero failures. New cases verify manual-first capture, editable 10 cm default distance logic, tracking-gap handling, custom cycles, LOD-off defaults and snapshot budgets. Standalone XR matrix, asymmetric/portrait intrinsics, normalized depth and unprojection tests pass. New application modules pass Node syntax checks. Static build succeeds; manifest, icons and route declaration validate. The project is exported to a downloadable ZIP without `.git`, `node_modules`, or temporary configuration. The Android package runbook and helper intentionally stop before signing; a verified TWA/APK still requires the owner’s deployment URL, Android package ID, release certificate and Digital Asset Links.

Physical ARCore tracking, pixel orientation and depth capture have **not** been validated on Android hardware in this sandbox. WebGPU training requires a real adapter; the sandbox screenshot browser has no training adapter. These are disclosed in the device guide and README. Depth/hit-test seeds are optional; without at least eight measured points, random-volume initialization is labeled and an orbit is required. No native ARCore sparse feature cloud is claimed.
