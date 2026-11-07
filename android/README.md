# SplatXR Android (arm64-v8a)

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
3. Move around a stationary subject and capture at least three overlapping keyframes (20–200 recommended). Manual shutter is the default; optionally enable distance-based auto-capture and edit the 2–200 cm spacing (10 cm default). The preview accumulates a voxelized ARCore feature-point cloud only when a keyframe is successfully captured; it is colored red-to-green by angular diversity. The saved cloud remains visible as you move between shots.
4. **Finish & train** writes JPEG keyframes, `transforms.json`, `seed_points.json`, a binary little-endian RGB `seed.ply`, and capture metadata under private app storage. The web app loads those files through same-origin local asset URLs and then uses its existing known-camera reconstruction path, bypassing SfM.
5. Review the frames and train if the device's WebView exposes a usable WebGPU adapter. Otherwise the captured dataset remains exportable from the web interface for desktop training.

The camera configuration is selected from the device's supported ARCore configurations by largest CPU-image size. Each pose and its image/intrinsics are sampled from the same ARCore frame. Seed colors are sampled from that frame's CPU image after projecting ARCore world feature points into the image. Depth mode is enabled when ARCore reports it as supported; the current seed source is the ARCore feature-point cloud, not a dense depth reconstruction.

## Current scope and hardware checks

- Only `arm64-v8a` is packaged. `minSdk` is 24 and `targetSdk` is 35.
- A real ARCore-compatible Android device is required to validate camera configuration, display orientation, tracking, GPU coverage rendering and image capture. Those device tests cannot be run in this repository sandbox.
- CPU camera image resolution depends on the ARCore device configuration. Selecting the largest supported image size does not guarantee a particular megapixel count.
- The native coverage colors summarize the largest angular separation from tracked camera observations, capped at 90 degrees. They are feedback for capture coverage, not a photogrammetric quality metric.
- The APK bundles the current app and training code, but WebGPU availability and memory limits are determined by Android System WebView and the device GPU.
- The PWA/TWA path remains available for deployed HTTPS sites. It does not provide native ARCore capture by itself.

## Release packaging

The debug artifact is sufficient for device testing. A distributable release APK still needs an owner-managed release signing key, versioning, and a hardware-tested release build. Do not reuse or publish the debug keystore as a release credential. TWA/Digital Asset Links configuration is only relevant if a deployed HTTPS site is packaged as a separate TWA.
