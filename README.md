# Android-Splat.js

**A fork of [arrival-space/splat.js](https://github.com/arrival-space/splat.js),
encapsulated in a native Android app to train on ARCore-tracked camera poses.**

Upstream Splat.js is an MIT-licensed JavaScript engine that reconstructs camera
poses with structure-from-motion (SfM) and trains a 3D Gaussian splat with
WebGPU — all in the browser. This fork keeps that engine and wraps it in an
Android app (display name **SplatXR**) whose native ARCore capture activity
records each photo together with its tracked pose and intrinsics. Those known
poses feed the trainer directly, so the SfM solve is skipped and reconstruction
starts from measured ARCore geometry instead of matched features.

Everything runs on the device: capture, training, and export need no account,
no photo upload, and no compute server. The result is a standard 3DGS `.ply`
that opens in any splat viewer.

## How it fits together

```text
Android native app (ARCore)              Local WebView (bundled web app)
┌──────────────────────────────┐          ┌────────────────────────────────┐
│ ARCore camera + tracking     │          │ Splat.js app (HTML/CSS/JS)     │
│ same-frame image, pose,      │  local   │ known-camera import skips SfM, │
│ intrinsics, feature points   ├── files ─>│ then seeds and trains on WebGPU│
│ coverage + camera overlays   │          │ exports .ply / session / photos│
└──────────────────────────────┘          └────────────────────────────────┘
```

1. **Capture (native).** The ARCore activity tracks the phone while you walk
   around a stationary subject. Each keyframe saves its JPEG, pose, and
   intrinsics from the same ARCore frame, plus sampled RGB seed points —
   written as `transforms.json`, `seed_points.json`, `seed.ply`, and
   `capture.json` in app-private storage.
2. **Import (bridge).** The files are exposed to the bundled WebView through
   `WebViewAssetLoader` (same-origin local URLs, no server, no upload) and
   loaded through the engine's known-camera path.
3. **Train (WebGPU).** Seeding and optimization run in the WebView via the
   shared `src/` engine. Poses stay fixed; only the splats train.
4. **Export.** Download the standard `.ply`, a resumable `session.zip`, the
   training `photos.zip`, or a transferable capture `.zip` for desktop
   training.

Without ARCore poses the app falls back gracefully: ordinary photos and video
go through the engine's own SfM solve, supported browsers keep the WebXR
capture route, and any capture can be exported for training elsewhere.

## What the app does

- **ARCore capture.** Manual shutter by default, optional distance-based
  auto-capture (2–200 cm spacing, 10 cm default), a red-to-green coverage
  cloud with camera viewpoints to guide the path, and ground leveling from
  the largest horizontal plane seen. At least 3 overlapping views; 20–200
  recommended. Requires Google Play Services for AR.
- **Known-pose training.** `load()` → `useKnownCameras(parseCameras(...))` →
  `seed()` → `start()` — the solve is bypassed entirely. Measured seed
  points initialize the model; with fewer than 8 the run falls back to a
  labeled random-volume seed.
- **Photo & video intake.** Loose photos, a video (sharp frames picked
  automatically, with an average-spacing control), or an exported `.zip`
  train through the same pipeline, solved from scratch when no poses ship
  with them.
- **Background training.** A foreground-service notification shows the live
  score, splat count, and cycle progress, with **Pause/Resume** actions that
  keep the service up while releasing the CPU wake lock when paused.
- **Review & library.** Captured frames are reviewable (inspect, remove,
  reset) before training; finished runs persist in an on-device library and
  reopen straight into the viewer.
- **Browser trainer (`browser/`).** A companion build of the same engine as
  a plain web page: photos, video, or an exported `.zip` in, `.ply` out —
  no Android code at all. See [browser/README.md](browser/README.md).

## Repository structure

```text
android/      native app: ARCore capture activity, WebView wrapper,
              training notification service. See android/README.md
app/          the web app bundled into the APK (UI, viewer, capture
              fallback, dataset review/export) — also runs standalone
src/          shared engine: Session pipeline, SfM solver, WebGPU 3DGS
              trainer, video/frame I/O, PLY export. No UI, no globals
browser/      browser-only trainer on the same engine (no device code)
tests/unit/   node unit tests (maths, pipeline, app behaviors)
tests/quality/ headless-Chrome accuracy gates (pose rows, Truck/Camping)
tests/e2e/    Playwright suite for app flows (train, pause/resume, restore)
docs/         install guides, capture notes, lab log, benchmark charts
scripts/      repo tooling (charts, data helpers)
Releases/     prebuilt APKs
serve.mjs     dev server · build.mjs assembles a static dist/ (no bundler)
```

## Run it

**Android APK** (ARCore capture + on-device training) — needs JDK 17,
Android SDK Platform 35, Build Tools 35.0.0, Gradle 8.9, and an
ARCore-compatible arm64 phone:

```sh
cd android
gradle --no-daemon assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The build copies the current `app/` and `src/` trees into APK assets, so
the WebView trains with this checkout's code. Details:
[android/README.md](android/README.md) ·
[Android installation guide](docs/android-install.md).
Or grab a prebuilt APK from `Releases/`.

**Web app** (photos/video/WebXR, desktop training):

```sh
node serve.mjs 8734
# http://localhost:8734/app/
```

**Browser-only trainer:**

```sh
python3 -m http.server 8000   # from the repo root
# http://localhost:8000/browser/
```

Requirements: a WebGPU browser (current Chrome/Edge/Safari, iPhones
included) for training; AR capture runs without WebGPU and its dataset can
be exported for desktop training instead.

## Tests

- `npm test` — node unit suite.
- `npm run test:quality` — public-API accuracy gates in headless Chrome
  (skips when the untracked photo datasets are absent).
- `npm run test:e2e` — Playwright app flows with real WebGPU training
  (needs a GPU box, not CI).

## Transfer a capture

**Export capture** saves a ZIP with the original photos, `transforms.json`
(or `cameras.json`), `seed_points.json`, and metadata. Unzip it on the
training device and select the photos plus the camera JSON together —
filenames pair each photo with its pose, so never rename, crop, rotate, or
mirror images independently of their calibration. Seed points are optional.

## License and credit

The engine is **splat.js by Arrival.Space**, MIT-licensed; its `LICENSE` is
preserved — support the maintainers at
https://github.com/arrival-space/splat.js.
Capture in this fork follows ARCore-tracked known poses rather than the
upstream photo-only workflow; the browser's WebXR route follows the
[Raw Camera Access](https://immersive-web.github.io/raw-camera-access/) and
[Depth Sensing](https://www.w3.org/TR/webxr-depth-sensing-1/) specifications.
