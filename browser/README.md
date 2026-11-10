# Splat.js · browser trainer

Part of **Android-Splat.js** — a fork of
[arrival-space/splat.js](https://github.com/arrival-space/splat.js) (see the
[project overview](../README.md)). This folder is the browser-only build:
drop in photos, a video, or an exported `.zip` and train a splat model in
the tab — no app install, no AR capture hardware, nothing uploaded
anywhere. The result is a standard 3DGS `.ply` that opens in any splat
viewer.

It is intentionally small: five files, no build step, no dependencies
beyond the shared engine in `../src` (imported as relative ES modules,
single source of truth with the full app).

## Run it

ES modules and WebGPU both refuse `file://`, so serve the **repo root**
over http and open `/browser/`:

```sh
# from the repository root
python3 -m http.server 8000
# → http://localhost:8000/browser/
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
  carries the finished model but no source images — the page detects it
  and explains there is nothing to train from. Reopen those in the full
  app instead.
- Filenames pair photos with poses (`frame_0001.jpg` ↔ its entry in
  `transforms.json`); keep them as exported.
- The **video frame spacing** option sets the preferred *average* gap
  between picked frames (Auto = the extractor's own pacing). Motion
  still adapts the pacing and the sharpest frame still wins each window.

## Options

- **Camera solve** — Quick / Standard / Precise. Only used when the input
  has no known poses.
- **Iterations** — 8k draft to 60k full. Schedules (learning-rate decay,
  growth stop) scale to the horizon.
- **Video frame spacing** — preferred average gap, Auto by default.

## During training

A live preview orbits the model as it trains (drag to orbit, scroll to
zoom). **Pause** halts the loop without losing anything, **Finish early**
keeps the model as it stands, and **Download .ply** saves the standard
3DGS file once training ends. The log details every stage.

## What's not here (by design)

- No AR capture, no camera UI, no device bridges — intake is files only.
- No session save/resume, no run library, no SOG/LOD export — train and
  download the `.ply`.
- No video review filmstrip: extraction keeps its automatic picks
  (the spacing option is the one control).

## Files

- `index.html` — layout and intake / training / preview sections.
- `app.js` — intake (photos / video / zip), the session pipeline, the
  orbit preview, PLY download. Imports the engine from `../src/index.js`.
- `styles.css` — standalone styling.
- `README.md` — this file.

Pinned by `tests/unit/test_browser_webapp.mjs` (structure, engine-only
imports, no device/capture references).
