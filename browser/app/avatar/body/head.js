// head.js — put the body model's head ON the person's face, in the tab (the
// port of tests/bench/head_correspond.py + head_deform.py).
//
// 1. Correspondences: the head region of the fitted surface is rendered
//    (flat-shaded, canvas 2D) from a frontal face-crop camera; the same face
//    landmarker that measured the person finds its 468 points on the render;
//    each one is traced through a triangle-id buffer to a barycentric point
//    on the mesh. No hand-picked vertex ids.
// 2. Deformation: a similarity on the head region, then a Laplacian
//    deformation of the head vertices with the correspondences pinned to the
//    triangulated landmarks (neck fixed, outlier pins dropped), solved by
//    conjugate gradients on the normal equations.
// All in the PLY frame.

/** @param {{vertices:number[][], faces:number[][]}} S  surface (PLY frame)
 *  @param {Float32Array|number[]} headWeight  per-vertex head-bone weight (same order as S.vertices)
 *  @param {object} cam   a crop camera {R,t,f,fy,cx,cy,side} (native crop scale)
 *  @param {object} landmarker  a tasks-vision FaceLandmarker (IMAGE mode)
 *  @returns {{corr: Map<number,{tri:number[], bary:number[]}>, render: OffscreenCanvas}} */
export function headCorrespondences(S, headWeight, cam, landmarker) {
  const V = S.vertices, T = S.faces; const SS = 2, W = cam.side * SS;
  const headT = T.filter(([a, b, c]) => headWeight[a] > 0.3 && headWeight[b] > 0.3 && headWeight[c] > 0.3);
  // project
  const X = V.map((v) => [cam.R[0] * v[0] + cam.R[1] * v[1] + cam.R[2] * v[2] + cam.t[0], cam.R[3] * v[0] + cam.R[4] * v[1] + cam.R[5] * v[2] + cam.t[1], cam.R[6] * v[0] + cam.R[7] * v[1] + cam.R[8] * v[2] + cam.t[2]]);
  const u = X.map((x) => (cam.f * x[0] / x[2] + cam.cx) * SS), v = X.map((x) => ((cam.fy ?? cam.f) * x[1] / x[2] + cam.cy) * SS);
  const zbuf = new Float32Array(W * W).fill(Infinity), tid = new Int32Array(W * W).fill(-1), bary = new Float32Array(W * W * 3);
  const img = new Uint8ClampedArray(W * W * 4); img.fill(235); for (let i = 3; i < img.length; i += 4) img[i] = 255;
  const light = [0.3, -0.5, -0.8]; const ln = Math.hypot(...light); light.forEach((q, i) => light[i] = q / ln);
  for (let k = 0; k < headT.length; k++) {
    const [a, b, c] = headT[k]; const pa = X[a], pb = X[b], pc = X[c];
    const nx = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]), ny = (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]), nz = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]);
    const nn = Math.hypot(nx, ny, nz); if (!nn || nz / nn > 0) continue;   // back-facing
    const shade = 0.35 + 0.65 * Math.max(0, (nx * light[0] + ny * light[1] + nz * light[2]) / nn);
    const col = [205 * shade, 175 * shade, 155 * shade];
    const x0 = Math.max(0, Math.floor(Math.min(u[a], u[b], u[c]))), x1 = Math.min(W - 1, Math.ceil(Math.max(u[a], u[b], u[c])));
    const y0 = Math.max(0, Math.floor(Math.min(v[a], v[b], v[c]))), y1 = Math.min(W - 1, Math.ceil(Math.max(v[a], v[b], v[c])));
    const d = (u[b] - u[a]) * (v[c] - v[a]) - (u[c] - u[a]) * (v[b] - v[a]); if (Math.abs(d) < 1e-9) continue;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = x + 0.5, py = y + 0.5;
      const l1 = ((u[b] - px) * (v[c] - py) - (u[c] - px) * (v[b] - py)) / d, l2 = ((u[c] - px) * (v[a] - py) - (u[a] - px) * (v[c] - py)) / d, l3 = 1 - l1 - l2;
      if (l1 < 0 || l2 < 0 || l3 < 0) continue;
      const z = l1 * X[a][2] + l2 * X[b][2] + l3 * X[c][2]; const o = y * W + x;
      if (z >= zbuf[o]) continue;
      zbuf[o] = z; tid[o] = k; bary[o * 3] = l1; bary[o * 3 + 1] = l2; bary[o * 3 + 2] = l3;
      img[o * 4] = col[0]; img[o * 4 + 1] = col[1]; img[o * 4 + 2] = col[2];
    }
  }
  const big = new OffscreenCanvas(W, W); big.getContext('2d').putImageData(new ImageData(img, W, W), 0, 0);
  const small = new OffscreenCanvas(cam.side, cam.side); small.getContext('2d').drawImage(big, 0, 0, cam.side, cam.side);
  const res = landmarker.detect(small);
  const corr = new Map();
  if (res.faceLandmarks && res.faceLandmarks[0]) {
    const L = res.faceLandmarks[0];
    for (let i = 0; i < 468; i++) {
      let px = Math.round(L[i].x * W), py = Math.round(L[i].y * W);
      let found = px >= 0 && px < W && py >= 0 && py < W && tid[py * W + px] >= 0;
      for (let r = 1; r <= 12 && !found; r++) for (let dy = -r; dy <= r && !found; dy++) for (let dx = -r; dx <= r && !found; dx++) {
        const yy = py + dy, xx = px + dx; if (yy < 0 || yy >= W || xx < 0 || xx >= W || tid[yy * W + xx] < 0) continue; px = xx; py = yy; found = true;
      }
      if (!found) continue;
      const o = py * W + px; corr.set(i, { tri: headT[tid[o]], bary: [bary[o * 3], bary[o * 3 + 1], bary[o * 3 + 2]] });
    }
  }
  return { corr, render: small };
}

/** Deform the head region of S onto the face points. face: Map(id -> [x,y,z]) PLY frame. Mutates S.vertices. */
export function deformHead(S, headWeight, corr, face, log = () => {}, scale = 1) {   // scale: scene units per metre
  const V = S.vertices.map((v) => v.slice()); const n = V.length;
  const pins = []; for (const [id, c] of corr) if (face.has(id)) pins.push({ tri: c.tri, bary: c.bary, P: face.get(id) });
  const src = (Vv) => pins.map((p) => [0, 1, 2].map((a) => p.bary[0] * Vv[p.tri[0]][a] + p.bary[1] * Vv[p.tri[1]][a] + p.bary[2] * Vv[p.tri[2]][a]));
  const dist = (A) => A.map((s, i) => Math.hypot(s[0] - pins[i].P[0], s[1] - pins[i].P[1], s[2] - pins[i].P[2]));
  const d0 = dist(src(V)); const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  // similarity on the head region
  const s0 = src(V); const ms = [0, 1, 2].map((a) => mean(s0.map((q) => q[a]))), mp = [0, 1, 2].map((a) => mean(pins.map((q) => q.P[a])));
  const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; let sq = 0;
  for (let i = 0; i < pins.length; i++) { const a = [0, 1, 2].map((k) => s0[i][k] - ms[k]), b = [0, 1, 2].map((k) => pins[i].P[k] - mp[k]); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] += b[r] * a[c]; sq += a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }
  const { U, S: Sv, Vt } = svd3(H); let det = det3(mul3(U, Vt)); const D = [1, 1, det < 0 ? -1 : 1];
  const R = mul3(mul3(U, [[D[0], 0, 0], [0, D[1], 0], [0, 0, D[2]]]), Vt); const sc = (Sv[0] * D[0] + Sv[1] * D[1] + Sv[2] * D[2]) / sq;
  const t = [0, 1, 2].map((k) => mp[k] - sc * (R[k][0] * ms[0] + R[k][1] * ms[1] + R[k][2] * ms[2]));
  for (let i = 0; i < n; i++) { const w = headWeight[i]; if (!w) continue; const v = V[i]; const q = [0, 1, 2].map((k) => sc * (R[k][0] * v[0] + R[k][1] * v[1] + R[k][2] * v[2]) + t[k]); V[i] = [v[0] + w * (q[0] - v[0]), v[1] + w * (q[1] - v[1]), v[2] + w * (q[2] - v[2])]; }
  const d1 = dist(src(V)); const med = [...d1].sort((a, b) => a - b)[d1.length >> 1]; const cut = Math.max(0.032 * scale, 2.5 * med);
  const keep = pins.filter((_, i) => d1[i] < cut);
  log(`head: ${pins.length} correspondences, mean ${(mean(d0) / scale * 1000).toFixed(1)} -> ${(mean(d1) / scale * 1000).toFixed(1)} mm after the similarity; ${keep.length} pins kept`);
  // Laplacian deformation of the free (head) vertices, pins soft, neck fixed
  const free = []; const idx = new Int32Array(n).fill(-1); for (let i = 0; i < n; i++) if (headWeight[i] > 0.3) { idx[i] = free.length; free.push(i); }
  const adj = Array.from({ length: n }, () => new Set()); for (const [a, b, c] of S.faces) { adj[a].add(b).add(c); adj[b].add(a).add(c); adj[c].add(a).add(b); }
  const nf = free.length, W_PIN = 1.0, W_PRIOR = 0.1;
  // rows: laplacian (nf), pins (keep.length), prior (nf); as sparse row lists
  const rows = [], rhs = [];
  for (const v of free) { const nb = [...adj[v]]; const r = [[idx[v], 1]]; const d = [V[v][0], V[v][1], V[v][2]]; for (const u2 of nb) { if (idx[u2] >= 0) r.push([idx[u2], -1 / nb.length]); for (let a = 0; a < 3; a++) d[a] -= V[u2][a] / nb.length; }
    // delta with the fixed neighbours' share already inside: rhs = L·V (only inside weights)
    const dl = [V[v][0], V[v][1], V[v][2]]; for (const u2 of nb) if (idx[u2] >= 0) for (let a = 0; a < 3; a++) dl[a] -= V[u2][a] / nb.length;
    rows.push(r); rhs.push(dl); }
  for (const p of keep) { const r = []; for (let j = 0; j < 3; j++) if (idx[p.tri[j]] >= 0) r.push([idx[p.tri[j]], p.bary[j] * W_PIN]); if (r.length) { rows.push(r); rhs.push(p.P.map((q) => q * W_PIN)); } }
  for (const v of free) { rows.push([[idx[v], W_PRIOR]]); rhs.push(V[v].map((q) => q * W_PRIOR)); }
  // normal equations A^T A x = A^T b, solved per axis by CG
  const ATA = Array.from({ length: nf }, () => new Map());
  const ATb = [new Float64Array(nf), new Float64Array(nf), new Float64Array(nf)];
  for (let r = 0; r < rows.length; r++) { const row = rows[r]; for (const [i, wi] of row) { for (const [j, wj] of row) ATA[i].set(j, (ATA[i].get(j) || 0) + wi * wj); for (let a = 0; a < 3; a++) ATb[a][i] += wi * rhs[r][a]; } }
  const mulA = (x, out) => { for (let i = 0; i < nf; i++) { let s = 0; for (const [j, w] of ATA[i]) s += w * x[j]; out[i] = s; } };
  const X = [0, 1, 2].map((a) => { const x = new Float64Array(nf); for (let i = 0; i < nf; i++) x[i] = V[free[i]][a]; return cg(mulA, ATb[a], x, nf, 400); });
  let maxMove = 0;
  for (let i = 0; i < nf; i++) { const v = free[i]; const q = [X[0][i], X[1][i], X[2][i]]; maxMove = Math.max(maxMove, Math.hypot(q[0] - V[v][0], q[1] - V[v][1], q[2] - V[v][2])); V[v] = q; }
  const d2 = dist(src(V));
  log(`head: after the Laplacian deformation mean ${(mean(d2) / scale * 1000).toFixed(1)} mm, max vertex move ${(maxMove / scale * 1000).toFixed(0)} mm`);
  S.vertices = V; S.headRegistered = { pins: keep.length, meanMm: mean(d2) / scale * 1000 };
  return S;
}

function cg(mulA, b, x0, n, iters) {
  const x = Float64Array.from(x0), r = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
  mulA(x, Ap); for (let i = 0; i < n; i++) { r[i] = b[i] - Ap[i]; p[i] = r[i]; }
  let rr = 0; for (let i = 0; i < n; i++) rr += r[i] * r[i]; const rr0 = rr;
  for (let it = 0; it < iters && rr > 1e-14 * (rr0 || 1); it++) {
    mulA(p, Ap); let pAp = 0; for (let i = 0; i < n; i++) pAp += p[i] * Ap[i]; const alpha = rr / (pAp || 1e-30);
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    let rr1 = 0; for (let i = 0; i < n; i++) rr1 += r[i] * r[i]; const beta = rr1 / rr; rr = rr1;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
  }
  return x;
}
const mul3 = (A, B) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));
const det3 = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
/** SVD of a 3x3 via Jacobi eigen-decomposition of MᵀM: M = U S Vᵀ. */
function svd3(M) {
  const MtM = mul3([[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]], M);
  const A = MtM.map((r) => r.slice()); const Vm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) off += A[i][j] * A[i][j]; if (off < 1e-24) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]); const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)); const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = Vm[k][p], vkq = Vm[k][q]; Vm[k][p] = c * vkp - s * vkq; Vm[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = [0, 1, 2].sort((a, b) => A[b][b] - A[a][a]);
  const S = order.map((i) => Math.sqrt(Math.max(0, A[i][i]))); const Vs = [0, 1, 2].map((r) => order.map((i) => Vm[r][i]));
  // U = M V / S
  const U = [0, 1, 2].map((r) => [0, 1, 2].map((c) => { const s = S[c] || 1e-30; return (M[r][0] * Vs[0][c] + M[r][1] * Vs[1][c] + M[r][2] * Vs[2][c]) / s; }));
  const Vt = [0, 1, 2].map((r) => [0, 1, 2].map((c) => Vs[c][r]));
  return { U, S, Vt };
}
