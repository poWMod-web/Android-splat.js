# SfM benchmark: stock COLMAP vs the Splat.js solver on Truck

Prepared 2026-09-09, not yet run. The question: the in-browser solve got
slower with the 09-04 desktop defaults (8000 SIFT features from the upsampled
first octave, pixel-aspect bundle adjustment) — how does it compare with stock
COLMAP on the same photographs, in time per phase and in pose accuracy?

## Inputs

- Images: `data/downloads/extracted/tandt/truck/images` — the 251 Tanks &
  Temples Truck release photographs, 979 × 546 px, the set the README table
  uses.
- Accuracy reference: the release COLMAP model
  `data/downloads/extracted/tandt/truck/sparse/0` (`images.bin`). Both solvers
  are scored against it with `tests/compare_colmap.mjs` (Umeyama-aligned ATE as
  a fraction of path length), so a fresh COLMAP run is judged the same way
  as ours.
- Machine: RTX 5080 desktop, GPU idle (close the app tab first).

## COLMAP side

Two builds are in `tools/`: 3.11.1 and 4.1.1, both **without CUDA** (CPU SIFT,
CPU matching). That is the honest "what this box can run" number; the
number people usually mean by stock COLMAP is the CUDA build. To get it,
download `COLMAP-3.11.1-windows-cuda.zip` from the COLMAP GitHub releases,
unzip anywhere, and pass its `bin\colmap.exe` with `-Gpu 1`.

```
powershell -ExecutionPolicy Bypass -File scratch\sfm_bench\run_colmap_truck.ps1
powershell -ExecutionPolicy Bypass -File scratch\sfm_bench\run_colmap_truck.ps1 -Colmap <cuda>\bin\colmap.exe -Gpu 1
```

The script times `feature_extractor` (8000 features, first octave −1, one
shared SIMPLE_RADIAL camera — the same budget as our defaults),
`exhaustive_matcher` (dense graph, like ours) and `mapper` (defaults), converts
the model to text, scores it against the release model and writes
`scratch/sfm_bench/colmap_truck/timings.json` plus `colmap.log`. Expect the
CPU exhaustive matcher to be the slow step (31,375 pairs).

## Splat.js side

Three headless cells, one per solve tier (`SOLVE_TIERS` in
`src/sfm/sfm.js`, bench `?solve=`), fresh solves (no `gtrecon`), each
followed by 30k training so the held-out PSNR shows what the poses buy; the
solver's full log and the solved cameras are posted:

```
node scratch/ab_cells.mjs scratch/sfm_bench/cells_sfm_bench.json
```

- `quick` — 3900 features, base octave, no aspect term (the pre-09-04
  defaults; Draft preset, phones).
- `standard` — 8000 features, base octave, no aspect term (the new desktop
  Standard preset, provisional until this benchmark says otherwise).
- `precise` — 8000 features from the upsampled octave + the aspect term
  (the 09-04 desktop defaults; High and Showcase presets; the README numbers).

The solver now logs a duration for every phase (commit after 22ba55d):
SIFT extraction, the feature-poor rescue pass, GPU matching, matching + pair
geometry (RANSAC over every pair), each focal-search candidate, and the final
registration + bundle adjustment. Kill leftover headless Chromes first
(`headless-rig-discipline`), and keep the desktop off the GPU while the cells
run.

## Collect

```
node scratch/sfm_bench/collect.mjs
```

prints one markdown table: features / matching / registration / total /
registered cameras / ATE for the two Splat.js cells and the COLMAP run.

## What to look for

- **Where the browser minutes go.** The app shows "Finding landmarks" through
  the rescue pass and the whole GPU matching (no progress event), then
  "Matching photos" during the per-pair RANSAC, then "Solving positions" for
  five focal-search registrations plus the final BA run. The log durations
  say which of those the 12 minutes belong to.
- **Feature budget vs time.** The previous-defaults cell isolates the cost of
  8000 features at octave −1 (roughly 2× the descriptors to match, more
  RANSAC work per pair, more observations in BA).
- **COLMAP's split.** Feature extraction and matching are the parts a CUDA
  build makes cheap; the mapper is CPU in every build, so its time is the
  like-for-like comparison with our registration + BA.
- **Accuracy.** Both ATEs against the release model, and registered counts.
  Ours read 0.00 % on this set in the README; a fresh COLMAP run should read
  near zero too (it is the same software as the reference).

## Results (2026-09-09, RTX 5080, GPU idle)

Splat.js, fresh solves at the three tiers plus two probes, each followed by
30 k training (seed 1, eval8):

| solver | SIFT | matching + pair geometry | focal search | final registration + BA | solve total | registered | ATE vs release | 30 k test PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quick — 3900 feats, octave 0 | 13 s | 27 s (GPU 2 s) | 32 s | 144 s (40 interim BAs = 97 s) | **3.6 min** | 251/251 | 0.00 % | 25.51 dB |
| 8000 feats, octave 0 (the provisional standard) | 13 s | 24 s | 32 s | 142 s | 3.5 min | 251/251 | 0.00 % | 25.42 dB |
| precise — 8000 feats, octave −1, aspect | 49 s | 86 s (GPU 23 s) | 82 s | 430 s (40 interim BAs = 270 s, final BAs 22 s) | **10.8 min** | 251/251 | 0.00 % | **25.86 dB** |
| precise without the aspect term | 50 s | 86 s | 81 s | 418 s | 10.6 min | 251/251 | 0.00 % | 25.72 dB |
| 5000 feats, octave −1, aspect | 50 s | 85 s | 94 s | 437 s | 11.1 min | 251/251 | 0.00 % | 25.84 dB |

What the table says:

- **The quality is in the upsampled octave, not the feature count.** 8000
  features at the base octave score *below* 3900 (the base octave cannot
  supply 8000 good keypoints at 960 px; the extra ones are weak). 5000 at
  octave −1 scores like 8000 and costs the same. The aspect term is worth
  0.14 dB at no time cost.
- **The time is in the final registration pass.** The upsampled octave
  yields 96 k points / 514 k observations instead of 25 k / 124 k, and the
  interim global bundle adjustment fires every six registrations regardless
  of size — forty of them, 270 s of the 430 s. The candidate fix is a
  geometric cadence (COLMAP adjusts globally every ~10 % more images):
  the same small early adjustments that stop drift, far fewer big ones.
- **Registration and pose accuracy** are identical across tiers on Truck
  (251/251, 0.00 % ATE against the release model): the tiers differ in
  sub-pixel landmark precision, which the trained model feels, not in
  which cameras get placed.

Conclusion for the presets: *quick* stays as defined; the *standard* tier
should be the precise recipe made cheaper (BA cadence, RANSAC in workers),
not a smaller feature budget — "8000 at octave 0" is withdrawn.

### Making the precise recipe cheaper (same day)

| change | solve | 30 k test PSNR | verdict |
|---|---:|---:|---|
| baseline precise | 10.8 min | 25.86 | — |
| geometric interim-BA cadence (×1.15) + worker RANSAC | 6.9 min | 21.91 | **dived** (rms 22.7 px before the last BA, ATE 1.77 %) |
| fixed cadence + worker RANSAC | 8.6 min | 21.51 | still bent (ATE 1.36 %) — init pair 6+199 at 3.7° parallax |
| inline RANSAC on the new structure (A/B) | 8.7 min | 25.82 | structure fine; the seeds changed the init pair |
| **init fix** (parallax bonus to 0.2 rad, prefer ≥ 5°) + worker RANSAC, interim BA on 25 k points | 7.8 min | 25.93 | init back to 8+198 at 12.3° |
| same, interim BA on all points | 10.1 min | 25.88 | subsampling is free |
| same, **interim BA on 10 k points** | **7.0 min** | **26.03** | shipped default |

Where the 7.0 minutes go now: SIFT 49 s, GPU matching 23 s, pair geometry
42 s (workers), focal search ~80 s, final registration 236 s (of which
interim BAs 78 s, final BAs ~22 s, registrations + LK + track extension the
rest). Next levers, in order: the focal search (four full registrations at
the upsampled octave; an EXIF focal prior skips it on phone captures), and
the registration loop itself.

**Preset decision:** Draft → *quick* (3.6 min, 25.5 dB); Standard, High and
Showcase → *standard* = the precise recipe at 7.0 min (26.0 dB). The
"8000 features at octave 0" middle tier is withdrawn (worse than quick);
'precise' remains an alias of standard.

### COLMAP, same photographs, same reference

| solver | features | matching | mapper / registration + BA | total | registered | ATE vs release |
|---|---:|---:|---:|---:|---:|---:|
| COLMAP 4.1.1, CPU build (no CUDA), 8000 feats, octave −1, exhaustive | 6 s | 501 s | 147 s | **10.9 min** | 251/251 | 0.00 % |
| COLMAP 3.11.1, CPU build | 6 s | crashes (access violation, 0xC0000005) a few blocks into the exhaustive match, twice | — | — | — | — |
| Splat.js Standard (this deploy) | 49 s | 65 s (GPU matching 23 s + worker RANSAC 42 s) | 316 s (focal search ~80 s + final 236 s) | **7.0 min** | 251/251 | 0.00 % |
| Splat.js Quick | 13 s | 27 s | 176 s | 3.6 min | 251/251 | 0.00 % |

Same accuracy against the release model (both 0.00 % ATE, all 251 cameras).
On this machine the browser solve is faster than the CPU COLMAP build: the
GPU matcher does in 23 s what COLMAP's CPU matcher does in 501 s, while
COLMAP's mapper (147 s) beats our registration + BA (316 s) — the focal
search accounts for most of that gap, since COLMAP takes the focal from
EXIF / its prior and we evaluate four candidates. A CUDA COLMAP build would
bring its matching down to seconds; not measured (needs a download).
