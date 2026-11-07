// cut.js — isolate the person from the trained room.
//
// Avatar mode trains the whole scene (session opts.maskTraining false): the
// masked recipe — few seeds, a random background at the silhouette, alpha
// pressure — is what drove the needle look on skin (needle ratio 344 vs 15
// for the same clip trained as a room, lab-log 2026-09-14b). The mattes were
// kept on the frames for this stage: the sparse cloud through the mattes
// bounds the subject, the visual hull is carved from the silhouettes, and
// every splat whose centre or most of whose body lies outside the hull goes
// (gs/hull.js makeSplatTest — the same test the masked trainer used as
// dead capacity). The survivors are seeded into a fresh session on the same
// frames and cameras, which is what body fit, bind and publish export from.
//
// Known limits (the next steps): the hull is loose, so wall colour can ride
// along at the back of the head and a sliver of floor under the shoes — a
// per-splat vote over the views and a short masked polish are the cure.
import { createSession } from '../../../src/index.js';

export const id = 'cut';
export const needs = ['train'];
const STRIDE = 16;   // gs/init.js: pos3, logScale3, quat4, color3, opacity, ...

export async function run(ctx, manifest, hooks) {
  const src = ctx.session; const log = hooks.log || (() => {});
  hooks.progress?.(0, 4, 'reading the model …');
  const raw = await src.exportRawState();
  // 1. bounds + hull from the mattes (session.maskPoints / _buildHull mutate
  //    recon.points — training is over, the seed is not needed again)
  // the face-mesh seed points are not sparse-cloud evidence: with 20k of them on the
  // head the hull's median-and-MAD bounds shrank to the head (2026-09-14)
  src.recon.points = src.recon.points.filter((p) => !p.faceSeed);
  const before = src.recon.points.length;
  src.maskPoints(0.5);
  // the box must hold the joints and the floor under the feet: the MAD box is
  // symmetric about the cloud's median and cut Tom off at the shins (2026-09-15)
  const lm = manifest && manifest.stages && manifest.stages.landmarks; const include = [];
  if (lm && lm.landmarks) {
    for (const [name, X] of Object.entries(lm.landmarks)) {
      if (!Array.isArray(X) || X.length < 3) continue; include.push(X);
      if (lm.floorY != null && /ank|heel|toe|foot/i.test(name)) include.push([X[0], lm.floorY, X[2]]);
    }
  }
  if (include.length) log(`cut: the hull box holds ${include.length} joint and floor points`);
  // The cut's generosity (2026-09-17, the user: hands, feet and the face chipped; a ceiling column
  // above Filip). Switches, all read from the URL:
  //   ?hulldilate=px   grow the matte before carving (thin parts the matte drops in a few frames)
  //   ?hullsigma=S     probe the splat's extent at S sigma (default 2)
  //   ?hullout=N       probes outside before a splat goes (default 4 of 6)
  //   ?hulltop=F       the hard top, F x the body height above the highest head landmark (default 0.15; 0 = off)
  const Q = new URLSearchParams(location.search);
  const hullOpts = { ...(include.length ? { include } : {}) };
  // the generous verdict is the DEFAULT since 2026-09-17 (Tom: 88,652 -> 93,049 splats kept, the
  // binder's "far" count 0 either way; the user saw chipped hands, feet and face at 2 sigma / 4 of 6):
  // matte dilated 6 px, probes at 1 sigma, 5 of 6 outside before a splat goes. The switches revert.
  hullOpts.dilatePx = Q.get('hulldilate') != null ? +Q.get('hulldilate') : 6;
  hullOpts.sigma = +Q.get('hullsigma') > 0 ? +Q.get('hullsigma') : 1;
  hullOpts.maxOut = +Q.get('hullout') > 0 ? +Q.get('hullout') : 5;
  const topF = Q.get('hulltop') != null ? +Q.get('hulltop') : 0.15;
  if (lm && lm.landmarks && lm.floorY != null && topF > 0) {
    let headY = -Infinity; for (const [name, X] of Object.entries(lm.landmarks)) { if (/nose|eye|ear/i.test(name) && Array.isArray(X) && X.length >= 3 && X[1] > headY) headY = X[1]; }
    // the recon's Y may point either way (camera-frame Y is DOWN): up is whichever side of the
    // floor the head is on; the first cut of this kind killed the whole body because it assumed +Y
    if (Number.isFinite(headY)) { const up = headY >= lm.floorY ? 1 : -1; const bodyH = Math.abs(headY - lm.floorY) || 1e-6; hullOpts.topY = headY + up * topF * bodyH; hullOpts.topUp = up; log(`cut: hard top ${(topF * 100).toFixed(0)} % of the body height above the highest head landmark (up is ${up > 0 ? '+' : '-'}Y)`); }
  }
  const hull = src._buildHull({ hullOpts });
  if (!hull || !src.splatTest) {
    log(`cut: no hull (${src.recon.points.length} of ${before} points on the subject) — the whole scene stays`);
    return { skipped: true, note: 'no hull — scene kept' };
  }
  hooks.progress?.(1, 4, 'cutting the person out …');
  // 2. the test on every splat: centre + extent (rmax = the longest axis)
  const dead = src.splatTest; const { data, n, sh, shK } = raw;
  const keep = new Uint8Array(n); let kept = 0;
  for (let i = 0; i < n; i++) {
    const b = i * STRIDE;
    const r = Math.exp(Math.max(data[b + 3], data[b + 4], data[b + 5]));
    if (!dead(data[b], data[b + 1], data[b + 2], r)) { keep[i] = 1; kept++; }
  }
  const data2 = new Float32Array(kept * STRIDE); const sh2 = sh ? new Float32Array(kept * shK * 3) : null;
  for (let i = 0, j = 0; i < n; i++) {
    if (!keep[i]) continue;
    data2.set(data.subarray(i * STRIDE, (i + 1) * STRIDE), j * STRIDE);
    if (sh2) sh2.set(sh.subarray(i * shK * 3, (i + 1) * shK * 3), j * shK * 3);
    j++;
  }
  log(`cut: hull ${hull.dim.join('x')} voxels of ${hull.cell.toFixed(3)}, ${(hull.fill * 100).toFixed(1)}% solid${hull.capped ? `, ${hull.capped.toLocaleString()} voxels above the top` : ''}${hull.dilatePx ? `, matte dilated ${hull.dilatePx} px` : ''}${hullOpts.sigma || hullOpts.maxOut ? `, verdict ${hullOpts.sigma || 2} sigma / ${hullOpts.maxOut || 4} of 6` : ''}; ${kept.toLocaleString()} of ${n.toLocaleString()} splats are the person`);
  // the isolated person's extent, so the viewer can frame HIM once it adopts
  // this model (app.js adoptSession): the mean of the survivors' centres and
  // the 90th-percentile distance from it — a few strays must not pull the
  // camera back out to where the room was
  let bounds = null;
  if (kept > 32) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < kept; i++) { const b = i * STRIDE; cx += data2[b]; cy += data2[b + 1]; cz += data2[b + 2]; }
    cx /= kept; cy /= kept; cz /= kept;
    const d = new Float32Array(kept);
    for (let i = 0; i < kept; i++) { const b = i * STRIDE; d[i] = Math.hypot(data2[b] - cx, data2[b + 1] - cy, data2[b + 2] - cz); }
    d.sort();
    bounds = { center: [cx, cy, cz], radius: d[Math.floor(kept * 0.9)] || d[kept - 1] || 1 };
  }
  // 3. a fresh session on the same frames and cameras, seeded with the survivors
  hooks.progress?.(2, 4, 'loading the isolated model …');
  const iter = src.trainer.iter;
  // the cut session's allocation is sized to the SURVIVORS, not inherited: `...src.opts`
  // carries the avatar's splat ceiling, and the source session is still resident (nothing
  // disposes it — the app's viewer and S.session still point at it), so inheriting a 2M
  // ceiling means two fully-allocated trainers, ~2.8 GB each, at the same moment. This
  // one never trains (maxIters is the iter it is already at); 2x kept is headroom for a
  // continue, at a fraction of the cost.
  const ses = createSession({ ...src.opts, maxIters: iter, evalSplit: 0, holdout: -1, maskTraining: false,
    trainer: { ...(src.opts.trainer || {}), maxSplats: Math.max(2 * kept, 1000), capMult: 2 } });
  // the SAME decoded frames as the source — no second decode: a 4K clip's 206 frames
  // are gigabytes of pixels and the reload failed with 'Array buffer allocation
  // failed' (Lisa, 2026-09-15). Body cameras only; the crop windows stay behind.
  const bodyCams = src.recon.cams.filter((c) => !c.crop);
  const idx = new Map(); const frames2 = [];
  for (const c of bodyCams) { if (!idx.has(c.imgIdx)) { idx.set(c.imgIdx, frames2.length); frames2.push(src.frames[c.imgIdx]); } }
  ses.useFrames(frames2);
  const cams = bodyCams.map((c) => ({ ...c, imgIdx: idx.get(c.imgIdx) }));
  ses.useReconstruction({ ...src.recon, cams, points: src.recon.points });
  await ses.seedFrom({ data: data2, n: kept, sh: sh2, shK, dc: raw.dc }, { iter });
  hooks.progress?.(4, 4, 'isolated');
  ctx.session = ses; ctx.cutSession = ses; ctx.plyBlob = null; ctx.sogBlob = null;
  return { splats: kept, of: n, hull: { dim: hull.dim, cell: hull.cell, fill: hull.fill }, ...(bounds ? { bounds } : {}), note: `${kept.toLocaleString()} of ${n.toLocaleString()} splats` };
}
