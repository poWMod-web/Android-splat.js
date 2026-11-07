# WebGPU trainer speed — source-level findings to test before any CUDA work

Source review of main at f829a5b (2026-09-08; shaders, trainer, session loop,
lab log). These are code-reading findings, not measurements. Each one is an
experiment with a cell to run; the order at the bottom is the proposed order.

## Ground truth to measure against

Frozen-model speed cells (one minute each, see the lab log 09-06):
`set=truck&iters=1&maxsplats=1050000&seed=1&gtrecon=/scratch/truck_solve_nf8000_oc-1.json&frommodel=/scratch/sig_truck_170000_cap1050000_m60_oc_s1.ply&gputime=200`
and the bicycle twin (`sample_bicycle.ply`, 966 k). Per-kernel baseline with
the 09-07 defaults (truck 1.04 M, 979 px): render 8.2 ms (fwd ≈ 3.3 inside
the fused pass, bwd ≈ 4.9), sort 1.9, chain 1.7, shAdam 1.0, scatter 0.75,
project 0.5, adam 0.4 + invis 0.4, vis passes 0.15. Full-training checks: the
30k truck cell (25.70 ± 0.05 on the frozen new-solve poses), garden 30k
(26.72 ± 0.05), and the hour curve.

**Profiler accounting (fix first).** `profileSteps()` runs a diagnostic
`renderFwd` and the fused forward/backward render and adds both to
`msPerStep`; production runs only the fused render. The 09-06 totals
(18.5 ms in profile mode) therefore carry an extra ~3.4 ms forward; the real
step is ≈ 15 ms. Report `msPerStep` without the diagnostic pass, keep
`renderFwd` as its own line, and remember the diagnostic pass also changes
cache state — normal training throughput (iterations per minute on the
hour cell) stays the final check. The profiler also omits refinement and the
normal scheduling path.

## Findings

### 1. Gradients and statistics computed for features that are off

| Work still performed | Where | Default situation |
|---|---|---|
| Camera rotation, translation, focal and aspect gradients: 8 global atomic adds per visible splat | `makeChainSrc()` | camera / aspect optimisation off |
| Exposure gain/bias gradients: 2 global atomic adds per valid pixel | render loss code | exposure updates off |
| Rendered mass, error mass and growth-gradient statistics: 3 shared atomic adds per contributing pixel–splat pair, then global flushes | render backward | legacy refinement without `errDonors` does not consume them |
| Robust-loss tile residual/count accumulation and a barrier | render loss code | runs when the robust loss is off |

The camera-gradient branch is gated on training mode (always on), not on
whether camera optimisation is enabled, so many splats accumulate into the
same few camera-gradient addresses with no consumer.

**Change:** specialise the shaders for the enabled features — compile out
unused camera/exposure gradients, refinement statistics and robust-loss
voting; keep variants for camera refinement, gradient checking and the
optional refinement policies (real features, they just should cost nothing
when off). This is the opposite of the failed subgroup-aggregation
experiment: not a cheaper way to do the work, no work.

**Measure:** speed cells (render, chain); 30k parity truck + garden; the
gradcheck (`src/gs/gradcheck.js`) for every variant that keeps a gradient.

### 2. Sort only as many entries as a tile has

The shared-memory sort always sorts 2,048 slots (padding initialised up to
`SHSORT`, every stage of a 2,048-element bitonic network), even for tiles
with 100 or 300 entries.

| Entries incl. padding | bitonic barrier stages | compare–exchange ops |
|---|---|---|
| 256 | 36 | 4,608 |
| 512 | 45 | 11,520 |
| 2,048 (current) | 66 | 67,584 |

Algorithmic counts, not predicted speedups: static shared-memory allocation
and occupancy stay the same at first.

**Change:** sort the next power of two above the actual count — a
count-dependent loop bound and an early return below two entries; keep the
depth-plus-ID comparison. **Also collect a tile-count histogram**: tiles above
2,048 go through the global-memory bitonic path, and if those dominate the
sort time the small-tile fix does not touch the main cost and a different
large-segment sorter is the real experiment.

**Measure:** sort line of the speed cells on truck and bicycle (1.9 / 1.3
ms); parity is a given if the comparison is unchanged, but run the 30k cell.

### 3. Move per-splat constants out of the pixel–splat backward loop

The backward computes `lmax` (largest eigenvalue of the projected
covariance, a square root) and `cnorm = 1 / (1 + lmax)` for every
contributing pixel–splat pair; both are constant per splat and view, and
projection already computes the same eigenvalue for the radius.

**Change:** compute the normalisation once in projection and store it; the
chain pass uses the same factor. The 16-float projected record stores three
covariance components mainly to rebuild this normalisation — there may be
room to repack without growing the record. Audit every consumer of `proj`
before repacking (render fwd/bwd, chain, compaction passes, the invisible
Adam pass reading slot 11).

**Measure:** render backward in the speed cells; run the gradcheck (the
fixed-point quantisation is sensitive to a changed floating-point path).

### 4. Remove the optimizer-state round trip in refinement

`_refineLegacy()` reads six buffers into JavaScript, sequentially — params,
Adam m, Adam v, SH, SH m, SH v — at allocated capacity, not active count,
and uploads all six again.

| Allocated capacity | download + upload per refinement (degree-3 SH) |
|---|---|
| 1.05 M | 1.54 GB |
| 4 M | 5.86 GB |

Excludes staging and JS copies; the default refines every 500 iterations
(the bar showcase night showed the refine cost directly: 200k iterations
took 36 min without relocation after 80k and 44–56 min with it).

The gather/plan/apply infrastructure exists but is tied to the alternative
refinement policies the lab log found can lose quality.

**Change:** separate data movement from policy. Keep the legacy selection,
splitting, jitter and moment-reset semantics; download only what the policy
needs (opacities, scales for donors), build the mutation plan on the CPU,
copy/reset SH and optimizer state on the GPU. Preserve ordering when donors
are reused or modified — a naïve parallel apply changes the algorithm.

**Measure:** refine wall time per call (log it), total training time on the
30k and hour cells, and parity (this must be bit-for-bit the same policy;
compare `_refines.txt` counts against the current path on the same seed).

### 5. Bin anisotropic splats more tightly

Projection derives one radius from the largest eigenvalue and bins the
square around that circle; a long thin Gaussian fills tiles where no pixel
passes the ellipse test, and each entry costs scatter, sort and both render
traversals.

**Change (conservative first step):** separate horizontal and vertical
bounds from the projected covariance V and the opacity-aware cutoff e_max:
r_x = sqrt(2 · e_max · V_xx), r_y = sqrt(2 · e_max · V_yy); intersect with
the existing circular clamp, keep the numerical margins, use identical
bounds in counting and scattering, keep the pixel-level rejection tests.
Later: reject tiles that provably miss the ellipse.

**Measure:** entries per frame (log the total from the scan), scatter/sort/
render lines; parity at 30k — the forward's binned-radius cut is mirrored in
the backward ("a splat the forward skipped must not receive gradient"), so
both must change together. Value depends on the anisotropy distribution
(truck p95 ratio ≈ 11, the bar 32).

### 6. Avoidable work in Adam

- Both Adam shaders evaluate `pow(beta1, t)` and `pow(beta2, t)` per
  parameter; the bias corrections are uniform per step — compute once, pass
  in.
- The three position lanes each generate the same Langevin random vector and
  covariance transform and select one component; a per-splat noise
  computation would do it once.
- **Possible data race:** position lanes read opacity/scales/quaternion from
  `params` while other lanes update those values in the same dispatch. A
  separate noise pass with explicitly pre- or post-update state would make
  the semantics defined. Correctness check first, then speed.

**Measure:** adam / shAdam lines; parity at 30k (the noise path changes the
random stream — expect run-to-run noise, compare two seeds).

## Already measured, leave alone

Subgroup-aggregated flush (42 ms), lane-spread partials (14–15 ms), vec4
`proj` loads (flat), zero-skip atomics (flat), K = 19 flush batches (noise) —
all opt-in with numbers in the shader comments. Preview refresh is throttled
and metric readbacks are paced.

## Order

1. Fix profiler accounting (report the fused step only).
2. Compile out unused gradients / statistics (#1).
3. Count-dependent small-tile sort (#2), with the tile histogram.
4. Precompute the conic normalisation (#3).
5. Tighter binning (#5).
6. Legacy refinement semantics with optimizer state kept on the GPU (#4).
7. Adam bias factors / noise pass (#6), race check first.

Run each against the same frozen inputs first, then full training time and
held-out quality. Postpone the CUDA trainer until these are measured.

## Status

- [x] 0 profiler accounting — `msPerStep` counts the production kernels only;
      `renderFwd` stays its own line, `renderBwd = render − renderFwd` is derived
      (2026-09-09).
- [x] 1 feature-specialised shaders — `makeRenderSrc(..., feat)` and
      `makeChainSrc(..., camGrad)`: camera/exposure gradients compile out unless
      `camOpt`/`aspectOpt`/`expComp`/`camGrads`; the refinement statistics
      (slots 10–12, the shared block shrinks 13 → 10 slots) compile out unless
      `engine v2`/`refineV2`/`errDonors`/`statMax`; the robust vote compiles out
      unless `robustLoss`. Lesson: a compiled-out buffer vanishes from the
      `'auto'` pipeline layout, so the kernels keep one statically unreachable
      reference to `gradCam` (the bind groups still pass it).
- [x] 2 count-dependent sort + histogram — the shared path sorts the next power
      of two ≥ count; `profileSteps()` reports `tileHist`.
- [x] 3 conic normalisation precompute — projection stores `1/(1+λmax)` in
      `proj[12]`; render and chain read it (slots 13/14 keep vb/vc, now unused).
- [x] 4 GPU-side refinement data movement — the legacy policy is untouched
      (same rng draws, same values); only the params of live rows come back
      (one download instead of six at capacity) and a `refine-patch` kernel
      writes the touched rows, zeroes their moments and copies the donor's SH
      on the GPU. Pinned-batch parity: 25.845 → 25.845, refinement logs
      identical line for line (2026-09-09). Wall time per refine still to be
      measured on an idle GPU.
- [x] 5 anisotropic binning — opt-in `opts.rectBin` (bench `?rectbin=1`):
      projection stores per-axis half-extents in `proj[13..14]`, scatter walks
      the same rectangle, the render keeps its circular cut. Truck: entries per
      frame 5.07 M → 3.01 M, tiles above 2048 entries 974 → 330; bicycle 2.90 M
      → 1.08 M, none above 2048, step 6.85 → 5.91 ms (shared GPU). Pinned-batch
      parity: the first cut also used the rectangle for the frustum test and
      read 25.816 / 25.879 vs 25.845 / 25.923 (seeds 1, 2) — the rectangle
      decided visibility for edge splats; with the frustum test back on the
      circle it reproduces 25.845 with a byte-identical refinement log.
      **Default ON since 2026-09-09** (`opts.rectBin`); idle-GPU speed cells
      still to be re-measured (daytime numbers were on a shared GPU).
- [x] 6 Adam constants — bias corrections in a new `bc` vec4 of both Adam
      uniforms, computed per step on the CPU. Noise pass / race: not done.

### Measured 2026-09-09 (RTX 5080, frozen models, 200 profiled steps)

| kernel (ms) | truck 1.04 M, before | after | bicycle 966 k, before | after |
|---|---|---|---|---|
| render (fused fwd+bwd) | 8.2 | **6.01** | 4.76 | **3.79** |
| sort | 1.9 | 1.71 | 1.3 | **0.47** |
| chain | 1.7 | 1.55 | — | 0.28 |
| shAdam / adam / adamInvis | 1.0 / 0.4 / 0.4 | 0.97 / 0.35 / 0.33 | — | 0.25 / 0.15 / 0.42 |
| scatter / project | 0.75 / 0.5 | 0.73 / 0.54 | — | 0.94 / 0.37 |
| **step, production kernels** | ≈ 15.1 (18.5 − 3.4 diagnostic fwd) | **12.35** | ≈ 8.9 | **6.85** |

Tile histogram (entries per 16 px tile, one frame): truck 5.07 M entries over
2170 tiles — 974 tiles **above 2048** (max 20 k), 927 in 1025–2048, 255 in
513–1024, 14 ≤ 512. Bicycle 2.9 M entries over 4056 tiles — 3 above 2048,
1021 in 1025–2048, 1123 in 513–1024, 1909 ≤ 512. So on truck half the tiles
take the global-memory bitonic path and #2 only touched the other half; the
sort win there (−0.2 ms) is small, on bicycle (−0.8 ms) large. The next sort
experiment is a large-segment sorter (radix, or a per-tile two-level merge),
and #5 (tighter binning) attacks the 4.9 entries per splat directly.

Correctness: gradcheck passes on all four rigs (params tile/global, pose,
SH3). Parity at 30 k, same seeds back to back on the same day: truck seed 1
old 25.89 / new 25.74, 25.77; seed 2 old 25.78 / new 25.82 — two-seed means
25.83 (old) vs 25.79 (new); garden seed 1 26.88 (new) vs 26.75 (09-08
reference). Inside the ±0.05 run noise — and a deterministic ruler confirms it: with the
per-frame batch pinned (bench `?ipf=15`) runs repeat bit-for-bit (same refine
iterations, 25.845 twice), and on that ruler the old code reads 25.807 vs the
new 25.845 (seed 1). The apparent seed-1 gap on free-running cells was
batch-timing chaos: the refine trigger fires once per frame after an adaptive
batch, so a faster kernel shifts refine iterations by tens of steps and the
trajectories diverge at the ±0.1 level. Use `?ipf=` for every A/B from now on.
