// fit.js — fit the Anny body to a capture, in the tab (the port of
// splat-rigger/tools/anny_fit.py).
//
// Parameters: a global similarity (rotation vector, translation, log scale:
// Anny metres -> scene units), one rotation vector per body bone (fingers
// stay at rest — nothing observes them) and the PCA shape coordinates.
// Residuals: the COCO keypoints against the triangulated landmarks
// (Huber, weighted by how many views agreed), a one-sided Chamfer from a
// vertex subsample to the splat centres (stage 2), and priors that keep the
// pose near rest and the shape near the sampled population. Solved by
// Levenberg-Marquardt with a numeric Jacobian: ~130 parameters, a forward
// pass is 5 ms, so a step costs well under a second and the whole fit a
// few seconds.
// Frames: landmarks and splat arrive in the PLY frame (y down); the model
// lives in Anny's (z up); ply -> anny is (x, z, -y).
import { rotvecToMat, plyToAnny, annyToPly, cullHeadInterior } from './anny.js';

const FINGER = /^(index|middle|pinky|ring|thumb)_/;

/** Uniform grid over points for nearest-neighbour queries (Chamfer). */
class PointGrid {
  constructor(pts, cell) {
    this.pts = pts; this.cell = cell; this.map = new Map();
    for (let i = 0; i < pts.length / 3; i++) { const k = this.key(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]); let a = this.map.get(k); if (!a) { a = []; this.map.set(k, a); } a.push(i); }
  }
  key(x, y, z) { return `${Math.floor(x / this.cell)},${Math.floor(y / this.cell)},${Math.floor(z / this.cell)}`; }
  nearest(x, y, z) {
    const cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell), cz = Math.floor(z / this.cell);
    let best = Infinity;
    for (let r = 0; r <= 2; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
        const a = this.map.get(`${cx + dx},${cy + dy},${cz + dz}`); if (!a) continue;
        for (const i of a) { const d = (this.pts[i * 3] - x) ** 2 + (this.pts[i * 3 + 1] - y) ** 2 + (this.pts[i * 3 + 2] - z) ** 2; if (d < best) best = d; }
      }
      if (best < (r * this.cell) ** 2) break;
    }
    return Math.sqrt(best);
  }
}

/**
 * @param {import('./anny.js').Anny} anny
 * @param {object} o
 * @param {Object<string,number[]>} o.landmarks   name -> [x,y,z] PLY frame (nose, l_sho, ... as POSE_LM)
 * @param {Object<string,{views:number}>} [o.report]
 * @param {Float32Array} [o.splat]   splat centres, PLY frame (xyz), for the surface stage
 * @param {(msg:string, frac:number)=>void} [o.progress]
 * @param {(m:string)=>void} [o.log]
 * @returns {{ params, verts, heads, tails, keypoints, residualsM, surface, markers, phenotypeLike }}
 */
export async function fitBody(anny, o) {
  const log = o.log || (() => {}); const progress = o.progress || (() => {});
  const iters1 = o.iters1 ?? 40, iters2 = o.iters2 ?? 20, huber = o.huber ?? 0.13, poseReg = o.poseReg ?? 0.02, shapeReg = o.shapeReg ?? 0.02;
  // residuals are in metres (scene units / the current scale): a scene's unit is arbitrary
  // (Lisa's orbit solved to ~3 m per unit, Tom's to 7 cm), so every threshold below is metric.
  // The values reproduce what Lisa's frame had validated (huber 0.05 units at scale 0.38).
  const B = anny.B, K = anny.K;
  const body = []; anny.boneLabels.forEach((n, i) => { if (!FINGER.test(n)) body.push(i); });
  const names = anny.kp.map((k) => k.name).filter((n) => o.landmarks[n]);
  const target = names.map((n) => plyToAnny(o.landmarks[n]));
  const conf = names.map((n) => Math.min(1, ((o.report && o.report[n] && o.report[n].views) || 8) / 30));
  // params: [rot(3), t(3), logS(1), body rotvecs (3 each), beta (K)]
  const NP = 7 + body.length * 3 + K;
  const p = new Float64Array(NP);
  const unpack = (q) => {
    const rv = new Float32Array(B * 3); body.forEach((b, i) => { rv[b * 3] = q[7 + i * 3]; rv[b * 3 + 1] = q[7 + i * 3 + 1]; rv[b * 3 + 2] = q[7 + i * 3 + 2]; });
    return { G: rotvecToMat([q[0], q[1], q[2]]), t: [q[3], q[4], q[5]], s: Math.exp(q[6]), rv, beta: Array.from(q.subarray(7 + body.length * 3)) };
  };
  const xf = (u, X) => [ (u.G[0] * X[0] + u.G[1] * X[1] + u.G[2] * X[2]) * u.s + u.t[0], (u.G[3] * X[0] + u.G[4] * X[1] + u.G[5] * X[2]) * u.s + u.t[1], (u.G[6] * X[0] + u.G[7] * X[1] + u.G[8] * X[2]) * u.s + u.t[2] ];
  const forward = (q) => { const u = unpack(q); const out = anny.forward(u.beta, u.rv); return { u, out, kp: anny.keypoints(out.verts) }; };

  // init: scale from the ankle-ear span, translation from the centroid, yaw from the shoulder line
  {
    const { out, kp } = forward(p);
    const zi = names.filter((n) => ['l_ank', 'r_ank', 'l_ear', 'r_ear'].includes(n));
    if (zi.length >= 3) {
      const mz = zi.map((n) => kp[n][2]), tz = zi.map((n) => target[names.indexOf(n)][2]);
      p[6] = Math.log(Math.max(0.05, (Math.max(...tz) - Math.min(...tz)) / Math.max(1e-3, Math.max(...mz) - Math.min(...mz))));
    }
    const s = Math.exp(p[6]);
    const cm = [0, 1, 2].map((a) => names.reduce((acc, n) => acc + kp[n][a], 0) / names.length), ct = [0, 1, 2].map((a) => target.reduce((acc, v) => acc + v[a], 0) / target.length);
    if (names.includes('l_sho') && names.includes('r_sho')) {
      const dm = [kp.r_sho[0] - kp.l_sho[0], kp.r_sho[1] - kp.l_sho[1]], dt = [target[names.indexOf('r_sho')][0] - target[names.indexOf('l_sho')][0], target[names.indexOf('r_sho')][1] - target[names.indexOf('l_sho')][1]];
      p[2] = Math.atan2(dt[1], dt[0]) - Math.atan2(dm[1], dm[0]);
    }
    const G = rotvecToMat([0, 0, p[2]]); const rc = [G[0] * cm[0] + G[1] * cm[1] + G[2] * cm[2], G[3] * cm[0] + G[4] * cm[1] + G[5] * cm[2], G[6] * cm[0] + G[7] * cm[1] + G[8] * cm[2]];
    p[3] = ct[0] - s * rc[0]; p[4] = ct[1] - s * rc[1]; p[5] = ct[2] - s * rc[2];
    void out;
  }

  // splat centres in Anny's frame, subsampled, on a grid
  let grid = null, vsub = null;
  if (o.splat && o.splat.length) {
    const n = o.splat.length / 3, step = Math.max(1, Math.floor(n / 12000)); const pts = [];
    for (let i = 0; i < n; i += step) { const a = plyToAnny([o.splat[i * 3], o.splat[i * 3 + 1], o.splat[i * 3 + 2]]); pts.push(a[0], a[1], a[2]); }
    grid = new PointGrid(Float32Array.from(pts), 0.05 * Math.exp(p[6]) * 2);
    const outer = anny.outerSurfaceMask(); const cand = []; for (let i = 0; i < anny.V; i++) if (outer[i]) cand.push(i);
    vsub = []; for (let i = 0; i < cand.length; i += Math.max(1, Math.floor(cand.length / 1500))) vsub.push(cand[i]);
  }

  // residual vector for a parameter vector
  const residuals = (q, stage) => {
    const { u, out, kp } = forward(q); const r = [];
    for (let i = 0; i < names.length; i++) {
      const X = xf(u, kp[names[i]]); const d = [(X[0] - target[i][0]) / u.s, (X[1] - target[i][1]) / u.s, (X[2] - target[i][2]) / u.s]; const n = Math.hypot(...d);
      const w = Math.sqrt(conf[i]) * (n > huber ? Math.sqrt(huber / n) : 1) * (stage === 2 ? Math.SQRT1_2 : 1);   // Huber as IRLS weight
      r.push(w * d[0], w * d[1], w * d[2]);
    }
    if (stage === 2 && grid) {
      const wS = 1 / Math.sqrt(vsub.length) * 3;
      for (const i of vsub) { const X = xf(u, [out.verts[i * 3], out.verts[i * 3 + 1], out.verts[i * 3 + 2]]); r.push(wS * grid.nearest(X[0], X[1], X[2]) / u.s); }
    }
    for (let i = 0; i < body.length * 3; i++) r.push(Math.sqrt(poseReg) * q[7 + i]);
    for (let k = 0; k < K; k++) r.push(Math.sqrt(shapeReg) * q[7 + body.length * 3 + k] / (anny.sigma[k] || 1));
    return r;
  };

  // Levenberg-Marquardt with a forward-difference Jacobian
  const lm = async (stage, iters, tag) => {
    let lambda = 1e-2; let r0 = residuals(p, stage); let c0 = r0.reduce((a, v) => a + v * v, 0);
    for (let it = 0; it < iters; it++) {
      const M = r0.length; const J = new Array(NP);
      for (let j = 0; j < NP; j++) {
        const h = j < 7 ? 1e-4 : 1e-3; const q = Float64Array.from(p); q[j] += h; const r1 = residuals(q, stage);
        const col = new Float64Array(M); for (let i = 0; i < M; i++) col[i] = (r1[i] - r0[i]) / h; J[j] = col;
      }
      const H = new Float64Array(NP * NP), g = new Float64Array(NP);
      for (let a = 0; a < NP; a++) { const Ja = J[a]; let ga = 0; for (let i = 0; i < M; i++) ga += Ja[i] * r0[i]; g[a] = ga; for (let b = a; b < NP; b++) { const Jb = J[b]; let s = 0; for (let i = 0; i < M; i++) s += Ja[i] * Jb[i]; H[a * NP + b] = s; H[b * NP + a] = s; } }
      let accepted = false;
      for (let tries = 0; tries < 6 && !accepted; tries++) {
        const A = new Float64Array(H); for (let a = 0; a < NP; a++) A[a * NP + a] *= (1 + lambda);
        const dp = solve(A, g.map((v) => -v), NP);
        const q = Float64Array.from(p); for (let a = 0; a < NP; a++) q[a] += dp[a];
        const r1 = residuals(q, stage); const c1 = r1.reduce((a, v) => a + v * v, 0);
        if (c1 < c0) { p.set(q); r0 = r1; c0 = c1; lambda = Math.max(1e-6, lambda / 3); accepted = true; } else lambda *= 10;
      }
      progress(`${tag} · ${it + 1}/${iters}`, (it + 1) / iters);
      await new Promise((res) => setTimeout(res, 0));
      if (!accepted && lambda > 1e6) break;
    }
    return c0;
  };
  const t0 = performance.now();
  await lm(1, iters1, 'fitting the body to the joints');
  if (grid) await lm(2, iters2, 'pulling the body onto the splat');
  log(`body fit: ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // outputs
  const { u, out, kp } = forward(p);
  const resid = {}; let sum = 0;
  names.forEach((n, i) => { const X = xf(u, kp[n]); resid[n] = Math.hypot(X[0] - target[i][0], X[1] - target[i][1], X[2] - target[i][2]) / u.s; sum += resid[n]; });
  log(`body fit: scale ${u.s.toFixed(3)}, landmark residual mean ${(sum / names.length * 100).toFixed(1)} cm`);
  const toPly = (X) => annyToPly(xf(u, X));
  const vertsPly = new Float32Array(anny.V * 3); for (let i = 0; i < anny.V; i++) vertsPly.set(toPly([out.verts[i * 3], out.verts[i * 3 + 1], out.verts[i * 3 + 2]]), i * 3);
  const heads = {}, tails = {};
  anny.boneLabels.forEach((n, b) => { heads[n] = toPly([out.heads[b * 3], out.heads[b * 3 + 1], out.heads[b * 3 + 2]]); tails[n] = toPly([out.tails[b * 3], out.tails[b * 3 + 1], out.tails[b * 3 + 2]]); });
  const kpPly = {}; for (const n of anny.kp.map((k) => k.name)) kpPly[n] = toPly(kp[n]);
  // rig markers by rpm_std proportion along the fitted skeleton (anny_fit.py's rule)
  const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], lerp = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  const shoMid = mid(heads.upperarm_l, heads.upperarm_r), earMid = mid(kpPly.l_ear, kpPly.r_ear), hips = heads.pelvis;
  const markers = {
    Hips: hips, Spine2: lerp(hips, shoMid, 0.72), Head: lerp(shoMid, earMid, 0.59),
    LeftArm: heads.upperarm_l, LeftForeArm: heads.lowerarm_l, LeftHand: heads.hand_l, RightArm: heads.upperarm_r, RightForeArm: heads.lowerarm_r, RightHand: heads.hand_r,
    LeftLeg: heads.calf_l, LeftFoot: heads.foot_l, LeftToe_End: tails.ball_l, RightLeg: heads.calf_r, RightFoot: heads.foot_r, RightToe_End: tails.ball_r,
  };
  // the surface: outer skin only, in export-binding-core's shape
  const outer = anny.outerSurfaceMask(); const remap = new Int32Array(anny.V).fill(-1); let nv = 0;
  for (let i = 0; i < anny.V; i++) if (outer[i]) remap[i] = nv++;
  const vertices = [], boneIndices = [], boneWeights = [];
  for (let i = 0; i < anny.V; i++) {
    if (!outer[i]) continue;
    vertices.push([vertsPly[i * 3], vertsPly[i * 3 + 1], vertsPly[i * 3 + 2]]);
    const bi = [], bw = []; for (let k = 0; k < 6; k++) { bi.push(anny.boneIdx[i * 6 + k]); bw.push(anny.weights[i * 6 + k]); }
    boneIndices.push(bi); boneWeights.push(bw);
  }
  const faces = [];
  for (let t = 0; t < anny.faces.length; t += 3) { const a = remap[anny.faces[t]], b = remap[anny.faces[t + 1]], c = remap[anny.faces[t + 2]]; if (a >= 0 && b >= 0 && c >= 0) faces.push([a, b, c]); }
  const surface = { frame: 'ply (y down)', source: 'anny game_engine rig (in-tab fit)', boneLabels: anny.boneLabels, vertices, faces, boneIndices, boneWeights,
    boneHeads: anny.boneLabels.map((n) => heads[n]), surfaceOnly: 'outer skin (largest connected component, head interior culled)' };
  // the mouth cavity is connected to the lips and the sockets keep a lining:
  // a ray test on the head takes those out (the user's rule: one surface)
  const hb = anny.boneIndex.head; const hwOuter = boneIndices.map((bi, i) => bi.reduce((s, b, k) => s + (b === hb ? boneWeights[i][k] : 0), 0));
  cullHeadInterior(surface, hwOuter, { reach: 0.16 * u.s, cell: 0.05 * u.s });   // 16 cm reach in scene units
  return { params: Array.from(p), scale: u.s, residualsM: resid, markers, landmarks: kpPly, surface, headWeightOf: (i) => anny.headWeight[i], outerRemap: remap };
}

/** Dense symmetric solve (Cholesky with a fallback to Gaussian elimination). */
function solve(A, b, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j]; for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) { if (s <= 1e-18) s = 1e-18; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
    }
  }
  const y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}
