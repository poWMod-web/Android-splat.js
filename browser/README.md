# Splat.js · browser shell

Part of **Android-Splat.js** — a fork of
[arrival-space/splat.js](https://github.com/arrival-space/splat.js) (see the
[project overview](../README.md)). This folder is the browser-only build:
the **same UI** as the full app, booted with the browser profile —
local-only, no AR capture, no avatar route, no share backend. Drop in
photos, a video, or an exported `.zip` and train a splat model in the
tab. The result is a standard 3DGS `.ply` that opens in any splat viewer.

It is self-contained: a generated `index.html`, vendored shared app
files and assets, and the engine (`src/`). Keeping the app subtree one
level under the root preserves every relative import untouched
(`../avatar`, `../../src`, …) — no imports leave this folder.

Shared files are synced by `tools/sync-browser.sh`. The Browser project
owns `app/js/app.js`, `pcview.js`, and `headless.js`; the sync preserves
rather than overwrites them, so Android viewer/background changes cannot
silently enter Browser. The Android service driver `train-headless.js`
is not shipped. `tests/unit/test_browser_vendor.mjs` enforces this
boundary and byte-identity for the remaining shared files.

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

## The browser profile

Same screens, same flows, same training knobs as the app's start
card and dock — a config here trains exactly as it would there. The
profile (`SHELL` in `app/js/app.js`, set by the generated entry's
marker or `?shell=browser`) hides only what doesn't fit a plain tab:

- No AR capture, no camera mode, no guides — intake is files only
  (photos, video, capture `.zip`). Reviewing picked photo+pose
  sets stays.
- No avatar route (its ML models don't ship in this folder).
- No share backend: no arrival-space restore, no share links, no
  service worker. Runs, captures, and models stay in this origin's
  IndexedDB — nothing leaves the machine.

The wall library, setup card, live training dock, Details sheet, and
exports retain the shared product flow. Model viewing and background
integration are Browser-owned so they can stay aligned with upstream
browser behavior instead of Android service behavior.

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

## Files

Generated or mirrored (edit the source and re-run
`tools/sync-browser.sh`):

- `index.html` — generated from `app/index.html`: install bits out,
  shell marker in.
- Shared files under `app/js/`, plus `app/css/` and `app/avatar/`.
- `src/` — the engine, whole (`createSession`, solver, trainer, …).

Browser-owned (edit deliberately in this folder; sync preserves them):

- `app/js/app.js`, `app/js/pcview.js`, `app/js/headless.js`.
- `train-headless.js` is Android-only and intentionally absent.
- `app/models/splat-transform.bundle.mjs` + `app/vendor/st-worker.mjs`
  — the on-demand SOG codec (fetched only when a SOG export or import
  needs it).
- `app/vendor/playcanvas.min.mjs`, `app/vendor/webp.wasm` — the
  lite-viewer renderer and the thumbnail encoder.

Native (this shell's own words):

- `README.md` — this file.

Pinned by `tests/unit/test_browser_webapp.mjs` (structure),
`tests/unit/test_browser_vendor.mjs` (shared-file identity and protected
boundaries), and `tests/unit/test_browser_shell.mjs` (profile gates).
