// softprune.js — periodic importance culling during training.
//
// Method reference: "Speedy-Splat: Fast 3D Gaussian Splatting with Sparse
// Pixels and Sparse Primitives" (CVPR 2025, https://speedysplat.github.io).
// Clean-room implementation: the pipeline shape (score each splat's render
// contribution, periodically cull the lowest-scoring fraction, keep
// training) follows the published method, but every line below is original
// — nothing is copied from the upstream project (which carries the INRIA
// non-commercial research license).
//
// Importance here is accumulated RENDERED MASS (gradP slot 10, read through
// the refine gather): the total render weight Σ alpha·T each splat
// contributed over the window. It answers Speedy's question — "which
// splats does the image not depend on?" — with statistics the trainer
// already accumulates, instead of Speedy's exact Σ(∂I/∂g)² (which would
// need rasterizer-side gradient plumbing this engine doesn't have).
// Survivors keep their Adam moments: unlike relocation and reseed, a
// soft-prune never resets optimizer state.
//
// This module is DOM- and GPU-free on purpose: the trainer calls into it
// for planning, the unit tests import it directly.

/** Scratch temp for chunked compaction (16 MiB — safely under the 128 MiB
 *  phone storage binding, big enough that a 1M-splat model compacts in a
 *  few dozen dispatches). */
export const COMPACT_TEMP_BYTES = 16 * 1024 * 1024;

/** Rankmap sentinel: culled rows never scatter. */
export const CULLED = 0xffffffff;

/** COMPACT: ranked scatter of one source chunk into the temp buffer. The
 *  caller then copies tmp[0 .. survivors) onto dst[dstBase ..] with a
 *  plain buffer copy and walks to the next chunk. Ranks come from
 *  planSoftPrune; every dstBase is at or below its srcBase (compaction
 *  only moves rows left), so a chunk's destination never overlaps a
 *  later chunk's source. */
export const COMPACT_SRC = /* wgsl */ `
struct CUni { srcBase: u32, count: u32, stride: u32, dstBase: u32 };
@group(0) @binding(0) var<uniform> u: CUni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> rank: array<u32>;
@group(0) @binding(3) var<storage, read_write> tmp: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nw: vec3u) {
  let li = gid.x + gid.y * nw.x * 256u;
  if (li >= u.count) { return; }
  let r = rank[u.srcBase + li];
  if (r == 0xFFFFFFFFu) { return; }
  let di = (r - u.dstBase) * u.stride;
  let si = (u.srcBase + li) * u.stride;
  for (var k = 0u; k < u.stride; k++) { tmp[di + k] = src[si + k]; }
}
`;

/**
 * Plan one soft-prune: keep the `target` highest-mass rows.
 *
 * masses: Float32Array(n) of rendered mass per splat (gather slot 2).
 * ratio: fraction to cull, in (0, 1).
 * last: post-prune count at the previous prune (the growth ratchet —
 *   below the cap a prune never shrinks the model under it, so periodic
 *   culls can only eat regrowth, never reverse the growth curve; at the
 *   cap the ratchet releases and culling recycles freely).
 * cap: the true splat cap (atCap releases the ratchet).
 * logitOp/meanLogS: optional structural tie-break — among equal masses
 *   the bigger, more opaque splat survives (a large rarely-seen splat
 *   outranks a tiny frequent one at equal contribution).
 * Returns null when there is nothing worth doing, else { rank, kept,
 * culled, threshold, floorBound }: rank[i] is the survivor's compacted
 * row or CULLED; floorBound names the binding floor ('ratchet',
 * 'absolute', null). Deterministic: same inputs, same plan.
 */
export function planSoftPrune(masses, n, ratio,
    { minCull = 1024, floor = 8192, last = 0, cap = Infinity, logitOp = null, meanLogS = null,
      votes = null, votesNeeded = 2, skipRatchet = false } = {}) {
  if (!(ratio > 0 && ratio < 1)) throw new Error(`planSoftPrune: ratio must be in (0, 1), got ${ratio}`);
  if (!masses || masses.length < n) throw new Error(`planSoftPrune: masses holds ${masses ? masses.length : 0} rows for n=${n}`);
  let target = n - Math.floor(n * ratio);
  let floorBound = null;
  if (!skipRatchet && n < cap && last > 0 && target < Math.min(last, n)) {
    target = Math.min(last, n);   // the cull eats regrowth, not the curve
    floorBound = 'ratchet';
  }
  if (target < Math.min(n, floor)) {
    target = Math.min(n, floor);
    floorBound = 'absolute';
  }
  if (target >= n || n - target < minCull) return null;   // culls too little to matter
  let max = 0;
  for (let i = 0; i < n; i++) {
    const m = masses[i];
    if (!Number.isFinite(m)) return null;                 // fail safe on broken stats
    if (m > max) max = m;
  }
  // all-zero mass means the stats never accumulated (or a fully occluded
  // model): any threshold would cull ARBITRARY rows — refuse.
  if (max <= 0) return null;
  const sorted = Float32Array.from(masses.subarray(0, n));
  sorted.sort();
  const threshold = sorted[n - target];
  const rank = new Uint32Array(n).fill(CULLED);
  let kept = 0;
  const ties = [];
  for (let i = 0; i < n; i++) {
    if (masses[i] > threshold) rank[i] = kept++;
    else if (masses[i] === threshold) ties.push(i);
  }
  let order = ties;
  if (logitOp && meanLogS && kept < target) {
    // structural tie-break: opacity × volume, descending (stable — full
    // ties keep ascending rows). Non-finite structure sorts last: a
    // splat with broken params is culled before a sound one.
    const key = (i) => {
      const o = 1 / (1 + Math.exp(-logitOp[i]));
      const k = o * Math.exp(meanLogS[i] * 3);
      return Number.isFinite(k) ? k : -Infinity;
    };
    order = ties.slice().sort((a, b) => key(b) - key(a));
  }
  for (const i of order) {
    if (kept >= target) break;
    rank[i] = kept++;
  }
  const candidates = n - kept;
  if (!votes) return { rank, kept, culled: candidates, threshold, floorBound, candidates };
  if (votes.length < n) throw new Error(`planSoftPrune: votes holds ${votes.length} rows for n=${n}`);
  // hysteresis (Mobile-GS votes, on our measured-mass signal): candidates
  // vote, the merely unlucky reset — only consecutive offenders die, so one
  // noisy window can never kill a useful splat. Floors computed above still
  // hold: the final cull is a subset of the candidates.
  const kill = new Uint8Array(n);
  let final = 0;
  for (let i = 0; i < n; i++) {
    if (rank[i] === CULLED) {
      votes[i]++;
      if (votes[i] >= votesNeeded) { kill[i] = 1; final++; }
    } else votes[i] = 0;
  }
  // votes persist through a no-cull window (they ARE the window's product)
  if (final < minCull) return { rank: null, kept: n, culled: 0, threshold, floorBound, candidates };
  rank.fill(CULLED);
  kept = 0;
  for (let i = 0; i < n; i++) if (!kill[i]) rank[i] = kept++;
  return { rank, kept, culled: final, threshold, floorBound, candidates };
}

/**
 * Split the compaction into temp-sized chunks: [{ srcBase, count,
 * dstBase, survivors }]. Chunks partition the SOURCE rows; dstBase is the
 * running survivor count, so the caller's tmp→dst copies land
 * contiguously. Empty chunks are omitted (no dispatch, no copy).
 */
export function chunkPlan(rank, n, strideFloats, tempBytes = COMPACT_TEMP_BYTES) {
  if (!(strideFloats >= 1)) throw new Error(`chunkPlan: bad stride ${strideFloats}`);
  const rowsPerChunk = Math.max(1, Math.floor(tempBytes / (strideFloats * 4)));
  const chunks = [];
  let dstBase = 0;
  for (let c0 = 0; c0 < n; c0 += rowsPerChunk) {
    const c1 = Math.min(n, c0 + rowsPerChunk);
    let survivors = 0;
    for (let i = c0; i < c1; i++) if (rank[i] !== CULLED) survivors++;
    if (survivors) chunks.push({ srcBase: c0, count: c1 - c0, dstBase, survivors });
    dstBase += survivors;
  }
  return chunks;
}
