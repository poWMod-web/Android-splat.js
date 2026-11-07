# Parked ideas — with the conditions of their verdicts

Rule (2026-09-08): a negative result is a verdict on an idea *under the
defaults of that day*. Several ideas rejected in August and early September
came back as wins once a co-factor was fixed (relocation churn, the bench's
capacity rule, the pair gate, the export bake). So every parked idea lists
**what it lost against**, **what could have masked it**, and a **retest cell**.
Re-run the retest cells after every default change to the trainer or solver;
they are six-minute cells.

Speed cell base: `set=truck&iters=30000&seed=1&gtrecon=/scratch/truck_solve_nf8000_oc-1.json`
(reference 25.70 ± 0.05 under the 09-07 defaults), garden `set=garden&iters=30000&seed=1` (26.72).

| idea (bench flags) | verdict when parked | conditions of that verdict | masked by | status |
|---|---|---|---|---|
| Needle set `dilate=0.1&aniso=0&minscale=1e-5` | won at 20k, lost at 40k+ (09-01/02) | relocation to the last step; export bake at 0.3 isotropic | churn degenerated thin splats; export made them transparent | **DEFAULT since 09-08**: +0.14/+0.19/+0.15 at 30k/40k/hour on truck; garden/bicycle/playroom/synthetic all up |
| Placement set `?placement=1` (mipComp off, refineV2, growRate 0.1 + needle) | wins 20k, loses 40k on 5/8 sets | relocation to the last step | same churn (relocUntil 0.9·H recovered garden) | retest at 30k/40k under the anneal-bound relocation |
| `regvis=1` (regs on visible rows only) | −0.02 / −0.15 (rung 2) | truck/garden photos | scene type: on the 360 bar it keeps 3.7 M live (−0.37 dB) | scene-dependent; not a default |
| 2 M cap on truck | 26.19 at 114k vs 26.40 at 1.05 M/170k | hour budget, relocation to the end | fewer cycles per hour; churn | retest 2 M at a fixed 200k under the anneal-bound relocation |
| Bigger cap on the bar | +0.04 | `capMult 8` capped the buffer at 1.78 M | the ruler | pass `capmult=20`; the economy (75 % dead) is the real limit |
| Absolute pair gate (`pairinl=100`) | truck 22.67 as a default | fresh solve, incremental path | path fragility | shipped as the < 70 % retry only |
| Finer feature frames (`featres=1600/2400`) | fewer registrations on statue, Lisa | low-texture / soft footage | ratio gate first, then genuine descriptor quality | 960 stays; revisit with a better matcher |
| SSAA training (`trainer.ssaa=2`) | +0.08 truck at 2.3× cost | photos | targets carry optical blur | opt-in; big win on synthetic (+1.5) |
| Robust loss tile vote | negative on truck | no transient-splat decay | interaction with relocation | retest with `opadecay` |
| Subgroup flush / lane spread / vec4 proj loads / zero-skip | slower or flat | 1.04 M truck, RTX 5080 | none found — genuine | leave |
| `relocTaper` | neutral, 16 % dead | 30k | none | leave; the stop at the anneal end covers it |
| Aspect refinement on sparse sets | drifted to 1.045 on the statue | 34 photos, 2.5k features | feature starvation | consider locking below N images; measured ±0.5 on 3 views only |
| Pixel-tolerance scaling with feature frame | neutral | statue | not the limiter | kept (correct in principle) |

## How to use

1. Changed a trainer or solver default? Run the retest cells of every row whose
   "masked by" mentions what you changed, plus the needle and placement rows.
2. Record the new verdict with its conditions here and in the lab log.
3. A model that looks wrong in the viewer at a good PSNR is an export or
   viewer bug until proven otherwise — check `bakeOpacityCompensation` and the
   SOG conversion before touching the trainer.
