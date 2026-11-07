# SplatXR — Capture & Train

A browser-based fork of [arrival-space/splat.js](https://github.com/arrival-space/splat.js), based on commit `a84620c`. The original MIT-licensed JavaScript/WebGPU 3D Gaussian Splatting engine, app flow, live progress, viewer and PLY export are retained. SplatXR adds pose-synchronized WebXR photo capture and direct known-camera training.

## Run and build

Run `node serve.mjs 3000` to serve the app. `node build.mjs` assembles a deployable static `dist/` directory without a bundler or production package install. Serve the output over HTTPS. Browser capture and training happen on the user's device; the development server has no photo-upload or compute endpoints.

## Capture and train

Use **Capture with AR** on a supported device exposing immersive AR, Raw Camera Access and DOM Overlay. The browser is required to grant `camera-access` and a `local` reference space. WebXR tracking alone is not enough to read camera images. Use a standalone tab rather than an embedded Preview if camera/spatial-tracking permissions are blocked.

AR capture always starts in manual mode. Press Shutter for individual views, or explicitly enable auto-capture to take a photo after each 10 cm of tracked camera travel by default. Edit the distance in the AR overlay from 2 to 200 cm; the valid value persists on that device between capture sessions. Auto-capture has no time interval and does not trigger from rotation in place. Tracking gaps establish a new baseline rather than counting an unknown position jump.

Capture overlapping images while moving around a stationary subject. Each saved photo includes its same-XR-frame OpenGL camera-to-world transform and intrinsics from the view projection scaled to the raw camera image dimensions. Optional CPU depth and hit tests supply actual world-space seed positions; no native ARCore point cloud is claimed. At least three images are needed, and 20–200 are recommended. Inspect and remove frames in **Review frames & poses** before training.

Known-camera datasets go through `session.load()`, `session.useKnownCameras(parseCameras(...), { points })`, `session.seed()` and `session.start()`. The `session.solve()` call is skipped completely. Camera pose and pixel-aspect optimization stay off for known cameras. Fewer than eight measured points trigger an explicitly labeled random-volume initialization; the cameras must converge around the subject for that fallback. The original SfM pipeline remains available for ordinary images and standard camera capture.

The splat.js renderer assumes zero projection skew. Capture stops and imported known-camera input is rejected if a nonzero-skew calibration is detected; asymmetric principal points are supported. Seed ownership is preserved through export/import so removed frames also remove their measured seeds.

The bundled **Synthetic Corner** example demonstrates direct known-camera training without AR hardware. Its twelve rendered views are supplied by the upstream repository, not live captures.

## Custom training settings

Open the gear next to Start training. **Cycles** accepts a custom whole number from 1 to 1,000,000; suggested values remain available through the browser's numeric input list. Blank keeps the original 20,000-cycle default. Numeric validation runs before training, and accepted values persist across visits. Explicit developer URL/performance overrides still take precedence.

LOD is **Off by default**. To enable it, choose **On**, select a splat budget of at least 1,000,000, then edit **LOD snapshots**. Zero uses the automatic halving schedule; a positive count requests that many saved intermediate snapshots, with the final full-detail model always included. Counts are constrained so no level goes below 250,000 splats. For example, a 1,000,000 budget supports 1–2 intermediate snapshots, and a 4,000,000 budget supports 1–4. Short runs may not grow through every requested level; LOD exports include actual completed snapshots, not promised or fabricated ones.

## Install on Android

The app includes a standalone PWA manifest and **Install app** controls. On a compatible Android device, open the hosted HTTPS page in full Chrome and use the install prompt or Chrome menu → Install app / Add to Home screen. This runs the JavaScript app in the browser's runtime; it does not unlock WebGPU on unsupported GPUs or grant unavailable WebXR modules. The current service worker intentionally does not provide offline training.

See [Android installation guide](docs/android-install.md) and [Android packaging runbook](android/README.md). The `android/` directory now contains a native arm64-v8a ARCore/WebView app project. GitHub Actions builds an installable debug APK artifact; a Play Store/release APK still needs owner-managed release signing credentials and physical-device validation. The browser PWA/WebXR path remains available separately.

## Transfer a capture

**Export capture** saves a ZIP containing the original photos, `transforms.json` or `cameras.json`, `seed_points.json`, and metadata. Unzip it on the training device and select the photos and camera JSON together through **Choose photos / import**. Seed points are optional. Image basenames must be unique. Cropping, rotating or mirroring images independently of the saved calibration invalidates their poses/intrinsics.

Captures, their transforms, seed points and local runs are kept in the existing origin-local IndexedDB. Storage is best effort; export important work. Original Arrival community/viewer integrations remain from upstream and may require that service; no account is needed for capture, local training or export. Photos stay local unless a user explicitly uses upstream sharing functionality.

## Platform limits

The browser version's WebXR route uses the device's available AR backend; browser capability checks decide whether immersive AR is offered. The separate Android APK contains a native ARCore capture route with CPU camera images, tracked poses, an accumulated coverage cloud, and RGB seed export. ARCore depth support and WebXR Raw Camera Access vary independently. Unsupported devices retain file and standard-camera input. Native ARCore Cloud Anchors and the Geospatial API are not implemented.

WebGPU training requires an actual browser GPU adapter. AR capture can be used without WebGPU, followed by dataset export and desktop training. A real ARCore-compatible Android phone is required to validate physical tracking, camera orientation and depth on the native APK; those hardware capabilities cannot be exercised in this sandbox.

## Project structure

`app/` retains the upstream interface, viewer, capture fallback and export modules. `app/js/xr-capture.js` manages WebXR lifecycle and synchronous camera readback. `app/js/xr-math.js` handles camera/depth transforms. `app/js/xr-ui.js` provides capability guidance and dataset review/export. `app/js/xr-dataset.js` validates and filters image-camera matches. `src/` is the upstream JavaScript training engine with a small opt-out for panorama slicing of calibrated XR images. `data/synthetic/` is the bundled known-camera dataset. `tests/unit/` contains upstream and extension tests.

## License and credit

The foundation is **splat.js by Arrival.Space**, licensed under MIT; its `LICENSE` and `README.md` are preserved. Support the maintainers at https://github.com/arrival-space/splat.js. WebXR capture follows the [Raw Camera Access specification](https://immersive-web.github.io/raw-camera-access/) and [Depth Sensing specification](https://www.w3.org/TR/webxr-depth-sensing-1/).
