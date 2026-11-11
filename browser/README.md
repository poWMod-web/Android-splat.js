# Splat.js · browser trainer

Part of **Android-Splat.js** — a fork of
[arrival-space/splat.js](https://github.com/arrival-space/splat.js) (see the
[project overview](../README.md)). This folder is the browser-only build:
drop in photos, a video, or an exported `.zip` and train a splat model in
the tab — no app install, no AR capture hardware, nothing uploaded
anywhere. The result is a standard 3DGS `.ply` that opens in any splat
viewer.

It is self-contained: five native files plus vendored, byte-identical
copies of the engine (`src/`) and the app modules it reuses (`app/js/`,
`app/models`, `app/vendor`). No build step, no dependencies, no imports
that leave this folder — copy it anywhere and serve it.

The mirrors are synced from the repo by `tools/sync-browser.sh` (run it
from the repo root after changing any mirrored source);
`tests/unit/test_browser_vendor.mjs` fails on drift.

## Run it

ES modules and WebGPU both refuse `file://`, so serve **this folder**
over http:

```sh
# from the browser/ folder
python3 -m http.server 8000
# → http://localhost:8000/
```

Requires a WebGPU browser: recent Chrome / Edge, or Safari on macOS 15+
/ iOS 18+. `localhost` counts as a secure context, so WebGPU works there.
Without WebGPU the page says so and disables training.

## Inputs

| What you drop | What happens |
|---|---|
| 2+ overlapping photos | Camera solve → seed → train. |
| Photos + `transforms.json` / `cameras.json` | Known camera poses: the solve is skipped. A `seed_points.json` alongside is used as the seed cloud (8+ points). |
| One video (mp4/mov/webm/…) | Sharp frames are picked automatically (stable, unblurry frames win) and continue as photos. |
| One `.zip` | Unpacked in the tab (stored **and** deflated entries), then handled like the loose equivalent. |

Notes:

- A trained-**session** export (`recon.json` / `state.bin` / `model.sog`)
  opens as a finished model in the viewer — and with its checkpoint it
  can continue training (the source photos come from the recon's URLs,
  or you drop the matching frames when the export carries names only).
  Loose `.ply` / `.sog` files open the same way; with a `recon.json`
  alongside they also know the camera path.
- Filenames pair photos with poses (`frame_0001.jpg` ↔ its entry in
  `transforms.json`); keep them as exported.
- The **video frame spacing** option sets the preferred *average* gap
  between picked frames (Auto = the extractor's own pacing). Motion
  still adapts the pacing and the sharpest frame still wins each window.

## Options

The same training knobs as the full app's start card (same values, same
session wiring — a config here trains exactly as it would there):

- **Quality** — one knob for the rows below: Draft is quick and light,
  Showcase trains long at high detail. Editing any row shows Custom.
- **Training resolution** — largest dimension the photos are downscaled
  to. Auto picks from image size, count and GPU memory; Full trains at
  the pictures' native size as far as the memory budget allows.
- **Solve resolution** — dimension the solver finds its landmarks at.
  Higher = sharper poses, slower solve.
- **Camera solve** — Quick / Standard / Precise. Only used when the input
  has no known poses.
- **Working buffer** — render grid relative to the training resolution
  (2× removes edge ringing, ~1.7× slower; 0.5× quarters raster work).
- **Spherical harmonics** — view-dependent colour detail, 0 (flat) to 3.
- **Splats** — budget the model grows toward. Auto sizes from the cycle
  budget and device.
- **LOD levels** — with a 1M+ splat budget, the model pauses at halving
  detail levels for a polish-and-snapshot before growing on; each
  snapshot downloads as its own `.ply` next to the full model.
- **MCMC optimizer** — on by default (measured ahead of the classic
  schedule on most scene/budget combinations); Off restores it.
- **Iterations** — 8k draft to 60k full. Schedules (learning-rate decay,
  growth stop) scale to the horizon.
- **GPU usage** — share of wall time training runs the GPU. Below 100%
  the loop idles between full batches — the run just takes
  proportionally longer. Live: dragging it mid-run needs no restart.
- **Video frame spacing** — preferred average gap, Auto by default.

## During training

A live preview orbits the model as it trains (drag to orbit, scroll to
zoom), with the score curve beside it and a Details sheet (Score / Perf
/ Cameras / GPU) below. **Pause** halts the loop without losing
anything, **Finish early** keeps the model as it stands, and **Download
.ply** saves the standard 3DGS file once training ends. The log details
every stage.

## Yours + last capture (on this device)

Every run gets a record the moment training starts: the **Yours**
shelf lists finished and interrupted runs with a thumbnail, and a
finished tile reopens its model, curve, and Details memory — ready to
continue with its photos. Staged photos/video/zips are also remembered
as the **Last capture** tile, so a refresh never erases a walked set.
All of it stays in this origin's IndexedDB (nothing leaves the
machine); the record shape matches the full app's `store.js`.

## What's not here (by design)

- No AR capture, no camera UI, no device bridges — intake is files only.
- No SOG export — the shelf keeps the standard `.ply` (plus per-level
  files when LOD is on).
- No video review filmstrip: extraction keeps its automatic picks
  (the spacing option is the one control).

## Files

Native (this app's own code):

- `index.html` — layout and intake / training / preview sections.
- `app.js` — intake (photos / video / zip), the session pipeline, the
  orbit preview, PLY download. Imports the engine from `./src/index.js`.
- `zip.js` — DOM-free zip reader for user-supplied archives.
- `styles.css` — standalone styling.
- `README.md` — this file.

Vendored (byte-synced mirrors — never edit here, edit the source and
re-run `tools/sync-browser.sh`):

- `src/` — the engine, whole (`createSession`, solver, trainer, …).
- `app/js/` — the reused app modules: `chart`, `data` (presets),
  `develop` (photo/model compare), `download` (file saving),
  `img` (bitmap cache), `scene-cloud`, `session_io` (session
  save/load), `sog` (SOG codec loader), `store` (on-device run
  library), `xr-dataset` (capture validation), `zip` (zip writer).
- `app/models/splat-transform.bundle.mjs` + `app/vendor/st-worker.mjs`,
  `app/vendor/webp.wasm` — the on-demand SOG codec (~5.7 MB, fetched
  only when a SOG export or import needs it).

Pinned by `tests/unit/test_browser_webapp.mjs` (structure, in-folder
imports, no device/capture references in native code) and
`tests/unit/test_browser_vendor.mjs` (mirror byte-identity).
