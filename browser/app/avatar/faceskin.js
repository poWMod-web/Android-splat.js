// faceskin.js — a signed-distance field of the FACE SKIN for the trainer's skin
// term (the user's idea, 2026-09-15): splats near the face are pulled onto the
// triangulated face mesh (too far in or out costs), and their extent along the
// skin normal is penalised, so the face trains as a skin, not a volume. Hair and
// everything past the face oval are left alone: the field's weight fades to zero
// toward the mesh boundary and beyond a distance cut-off.
//
// The field: a grid over the face points' box (+ margin); per cell the signed
// distance d to the nearest face triangle (outward positive, toward the
// cameras), the unit normal n (= grad d), and a weight w in [0, 1].
const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const norm = (v) => Math.hypot(v[0], v[1], v[2]);

// closest point on triangle ABC to P (Ericson, Real-Time Collision Detection)
function closestOnTri(P, A, B, C) {
  const ab = sub(B, A), ac = sub(C, A), ap = sub(P, A);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return A;
  const bp = sub(P, B); const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return B;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return [A[0] + ab[0] * v, A[1] + ab[1] * v, A[2] + ab[2] * v]; }
  const cp = sub(P, C); const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return C;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return [A[0] + ac[0] * w, A[1] + ac[1] * w, A[2] + ac[2] * w]; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return [B[0] + (C[0] - B[0]) * w, B[1] + (C[1] - B[1]) * w, B[2] + (C[2] - B[2]) * w]; }
  const den = 1 / (va + vb + vc); const v = vb * den, w = vc * den;
  return [A[0] + ab[0] * v + ac[0] * w, A[1] + ab[1] * v + ac[1] * w, A[2] + ab[2] * v + ac[2] * w];
}

/** @param lm the landmarks stage result (face.points / face.ids in scene units, fit.scale units per metre)
 *  @param cams the body cameras (for the outward orientation)
 *  @returns {{origin:number[], cell:number, dims:number[], data:Float32Array, note:string}|null} */
export async function buildFaceSkinField(lm, cams, { res = 40, marginM = 0.03, fadeM = 0.015, cutoffM = 0.025, log = () => {} } = {}) {
  const face = lm && lm.face; const scale = lm && lm.fit && lm.fit.scale;
  if (!face || !face.points || face.points.length < 100 || !(scale > 0)) { log('skin field: no face — skipped'); return null; }
  const byId = new Map(face.ids.map((id, i) => [id, face.points[i]]));
  const vision = await import(/* @vite-ignore */ `${VISION_CDN}/vision_bundle.mjs`);
  const edges = vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION; const adj = new Map();
  for (const e of edges) { if (!adj.has(e.start)) adj.set(e.start, new Set()); if (!adj.has(e.end)) adj.set(e.end, new Set()); adj.get(e.start).add(e.end); adj.get(e.end).add(e.start); }
  const tris = [];
  for (const e of edges) { const a = Math.min(e.start, e.end), b = Math.max(e.start, e.end); for (const c of adj.get(a)) if (c > b && adj.get(b).has(c) && byId.has(a) && byId.has(b) && byId.has(c)) tris.push([a, b, c]); }
  if (tris.length < 200) { log(`skin field: only ${tris.length} triangles — skipped`); return null; }
  // the face oval: the mesh boundary, where the weight fades (hair, jaw edge)
  const ovalIds = new Set(); for (const e of (vision.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL || [])) { ovalIds.add(e.start); ovalIds.add(e.end); }
  const oval = [...ovalIds].filter((id) => byId.has(id)).map((id) => byId.get(id));
  // per vertex: the distance to the nearest oval vertex (scene units)
  const vDist = new Map();
  for (const [id, p] of byId) { let m = Infinity; for (const q of oval) { const d = norm(sub(p, q)); if (d < m) m = d; } vDist.set(id, m); }
  // outward orientation: toward the cameras' mean position
  const fc = [0, 0, 0]; for (const p of face.points) { fc[0] += p[0] / face.points.length; fc[1] += p[1] / face.points.length; fc[2] += p[2] / face.points.length; }
  const cm = [0, 0, 0]; let nc = 0;
  for (const c of cams) { if (c.crop) continue; const R = c.R, t = c.t; cm[0] -= (R[0] * t[0] + R[3] * t[1] + R[6] * t[2]); cm[1] -= (R[1] * t[0] + R[4] * t[1] + R[7] * t[2]); cm[2] -= (R[2] * t[0] + R[5] * t[1] + R[8] * t[2]); nc++; }
  const toCam = nc ? sub([cm[0] / nc, cm[1] / nc, cm[2] / nc], fc) : [0, 0, -1];
  const T = tris.map(([a, b, c]) => {
    const A = byId.get(a), B = byId.get(b), C = byId.get(c); let n = cross(sub(B, A), sub(C, A)); const l = norm(n) || 1; n = [n[0] / l, n[1] / l, n[2] / l];
    if (dot(n, toCam) < 0) n = [-n[0], -n[1], -n[2]];
    return { A, B, C, n, cen: [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3, (A[2] + B[2] + C[2]) / 3], bd: Math.min(vDist.get(a), vDist.get(b), vDist.get(c)) };
  });
  // the grid
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of face.points) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); }
  const margin = marginM * scale; for (let a = 0; a < 3; a++) { lo[a] -= margin; hi[a] += margin; }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]); const cell = span / res;
  const dims = [0, 1, 2].map((a) => Math.max(2, Math.ceil((hi[a] - lo[a]) / cell)));
  const N = dims[0] * dims[1] * dims[2]; const data = new Float32Array(N * 8);
  const fade = fadeM * scale, cutoff = cutoffM * scale; const K = 12; const t0 = performance.now();
  const cand = new Array(K); let inside = 0;
  for (let iz = 0; iz < dims[2]; iz++) for (let iy = 0; iy < dims[1]; iy++) for (let ix = 0; ix < dims[0]; ix++) {
    const P = [lo[0] + (ix + 0.5) * cell, lo[1] + (iy + 0.5) * cell, lo[2] + (iz + 0.5) * cell];
    // the K nearest triangle centroids, then the exact distance on those
    for (let k = 0; k < K; k++) cand[k] = { d2: Infinity, i: -1 };
    for (let i = 0; i < T.length; i++) {
      const c = T[i].cen; const d2 = (P[0] - c[0]) ** 2 + (P[1] - c[1]) ** 2 + (P[2] - c[2]) ** 2;
      if (d2 < cand[K - 1].d2) { let k = K - 1; while (k > 0 && cand[k - 1].d2 > d2) { cand[k] = cand[k - 1]; k--; } cand[k] = { d2, i }; }
    }
    let best = Infinity, bq = null, bt = null;
    for (let k = 0; k < K; k++) { const tri = T[cand[k].i]; const q = closestOnTri(P, tri.A, tri.B, tri.C); const d = norm(sub(P, q)); if (d < best) { best = d; bq = q; bt = tri; } }
    const sgn = dot(sub(P, bq), bt.n) >= 0 ? 1 : -1; const d = sgn * best;
    let n = sub(P, bq); const nl = norm(n); n = nl > 1e-6 * cell ? [n[0] / nl * sgn, n[1] / nl * sgn, n[2] / nl * sgn] : bt.n;
    // weight: fades at the mesh boundary and beyond the cut-off distance
    const wb = Math.min(1, Math.max(0, bt.bd / fade)); const wd = best < cutoff ? 1 - (best / cutoff) ** 2 : 0; const w = wb * wd;
    const o = ((iz * dims[1] + iy) * dims[0] + ix) * 8;
    data[o] = d; data[o + 1] = n[0]; data[o + 2] = n[1]; data[o + 3] = n[2]; data[o + 4] = w;
    if (w > 0) inside++;
  }
  const note = `${dims.join('x')} cells of ${(cell / scale * 1000).toFixed(1)} mm over the face (${tris.length} triangles), ${inside} cells weighted, ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  log(`skin field: ${note}`);
  return { origin: lo, cell, dims, data, note, scale };
}
