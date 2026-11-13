# SplatXR Android (arm64-v8a)

Part of **Android-Splat.js** — a fork of [arrival-space/splat.js](https://github.com/arrival-space/splat.js)
encapsulated in a native Android app so captures train on ARCore-tracked camera poses
(see the [project overview](../README.md)).

This Android project wraps the existing `app/` WebGPU interface in a local WebView and adds an optional native ARCore capture route. It is separate from the PWA/TWA install path: the native route uses ARCore directly, rather than relying on the browser's experimental WebXR Raw Camera Access extension.

## Build the APK

Requirements: JDK 17, Android SDK Platform 35, Build Tools 35.0.0, and Gradle 8.9. From this directory:

```sh
gradle --no-daemon assembleDebug
```

The arm64-only debug APK is written to:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install on a compatible arm64 Android phone with Google Play Services for AR:

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Debug builds use Android's automatically generated debug signing key and are for testing only; they are not Play Store release packages. The GitHub Actions workflow `.github/workflows/android-apk.yml` builds the same APK and uploads it as the `splatxr-arm64-debug-apk` artifact.

## Native capture flow

1. The WebView loads the checked-in HTML/CSS/JS app from APK assets using `WebViewAssetLoader`; it does not need a server.
2. **Capture with ARCore** opens a native camera surface. Allow camera permission and install/update Google Play Services for AR if requested.
3. Move around a stationary subject and capture at least three overlapping keyframes (20–200 recommended). Manual shutter is the default; optionally check Auto-capture and edit the 2–200 cm spacing (10 cm default) — the center button then starts/stops the automatic run (it reads Start/Stop), so you control when the distance trigger is live. The preview accumulates a voxelized ARCore feature-point cloud only when a keyframe is successfully captured; it is colored red-to-green by angular diversity. The saved cloud remains visible as you move between shots.
4. **Finish & train** writes JPEG keyframes, `transforms.json`, `seed_points.json`, a binary little-endian RGB `seed.ply`, and capture metadata under private app storage. The export is leveled to the largest ARCore horizontal plane seen while tracking (its normal becomes world +Y with the cameras above it; `capture.json` records the method under `leveling`, falling back to the cameras' average up when no floor or table was in view — except for portrait captures, whose sensor-up lies sideways to gravity: those keep the gravity-aligned AR world rather than baking in a 90° roll). The web app loads those files through same-origin local asset URLs and then uses its existing known-camera reconstruction path, bypassing SfM.
5. Review the frames and train if the device's WebView exposes a usable WebGPU adapter. Otherwise the captured dataset remains exportable from the web interface for desktop training.

The native capture overlay mirrors the WebXR capture layout (telemetry, framing reticle, guide, distance-auto toggle, shutter and finish control). Capture is locked to portrait orientation, and both the capture UI and main WebView are inset below Android's status/navigation bars (including display cutouts), so system bars do not cover the camera view or app canvas. Coverage points use a gradual red-to-orange (40°), yellow (80°), and green (120°) viewpoint-diversity ramp. Web app assets are copied from the repository's top-level `app/` and `src/` directories into the APK.

The camera configuration is selected from the device's supported ARCore configurations by largest CPU-image size. Each pose and its image/intrinsics are sampled from the same ARCore frame. Seed colors are sampled from that frame's CPU image after projecting finite ARCore world feature points into the image. Depth mode is enabled when ARCore reports it as supported; the current seed source is the ARCore feature-point cloud, not a dense depth reconstruction.

## Current scope and hardware checks

- Only `arm64-v8a` is packaged. `minSdk` is 24 and `targetSdk` is 35.
- A real ARCore-compatible Android device is required to validate camera configuration, display orientation, tracking, GPU coverage rendering and image capture. Those device tests cannot be run in this repository sandbox.
- CPU camera image resolution depends on the ARCore device configuration. Selecting the largest supported image size does not guarantee a particular megapixel count.
- The native coverage colors summarize the largest angular separation from tracked camera observations, capped at 90 degrees. They are feedback for capture coverage, not a photogrammetric quality metric.
- The APK bundles the current app and training code, but WebGPU availability and memory limits are determined by Android System WebView and the device GPU.
- While a model trains, an immediate, ongoing foreground-service notification shows the live score, splat count and cycle progress. When the Activity backgrounds, it writes one exact checkpoint, releases its WebGPU device, and the isolated `:training` service resumes the work in its own headless WebView; reopening the app shows the file-backed monitor without stopping or adopting the service trainer; ownership returns only after **Continue here** is pressed. The score curve and annotations cross the handoff with the model. Each checkpoint is mirrored to IndexedDB, complete Android run directories are rediscovered as Home tiles, and the latest native ARCore scan remains in `captures/current` as an independent source-image recovery copy. Its Pause/Resume action pauses training while keeping the service up (the CPU wake lock is released when paused); finishing leaves one dismissible summary. The notification permission is requested when training starts.
- The PWA/TWA path remains available for deployed HTTPS sites. It does not provide native ARCore capture by itself.

## Release packaging

The debug artifact is sufficient for device testing. A distributable release APK still needs an owner-managed release signing key, versioning, and a hardware-tested release build. Do not reuse or publish the debug keystore as a release credential. TWA/Digital Asset Links configuration is only relevant if a deployed HTTPS site is packaged as a separate TWA.
