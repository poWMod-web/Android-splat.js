// anny.js — the Anny body model (game_engine rig) in the tab.
//
// A snapshot exported by tests/bench/export_anny.py: the shape space as a PCA
// basis over the six phenotypes (vertices + bone heads + bone tails, so the
// rig follows the body), the skinning weights, the bone rolls that orient a
// bone from its head and tail (Blender's convention), the kinematic tree and
// the COCO keypoint regressor. Forward pass = Anny's: rest shape -> rest bone
// poses -> forward kinematics with per-bone local rotations -> linear blend
// skinning. Frame: Anny's (x lateral, y depth, z up, metres).
//
//   const anny = await loadAnny(url);
//   const out = anny.forward(beta, rotvecs);   // { verts, heads, tails, restHeads, poses, transforms }
//   anny.keypoints(out.verts)                 // { nose: [x,y,z], l_sho: ..., ... }

export async function loadAnny(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return parseAnny(buf);
}

export function parseAnny(buf) {
  const u8 = new Uint8Array(buf); const dv = new DataView(buf);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== 'ANNY') throw new Error('not an Anny snapshot');
  const hl = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)).replace(/\0+$/, ''));
  let off = 8 + hl;
  const take = (Ctor, n) => { const bytes = n * Ctor.BYTES_PER_ELEMENT; const a = new Ctor(buf.slice(off, off + bytes)); off += bytes + ((4 - bytes % 4) % 4); return a; };
  const { V, B, K, D, F } = meta;
  const mean = take(Float32Array, D), comps = take(Int8Array, K * D), compScale = take(Float32Array, K), sigma = take(Float32Array, K);
  const faces = take(Uint16Array, F * 3), weightsU8 = take(Uint8Array, V * 6), boneIdx = take(Uint8Array, V * 6);
  const rolls = take(Float32Array, B * 9), yAxis = take(Float32Array, 3), degenerate = take(Float32Array, 9), parents = take(Int8Array, B);
  const kpIdx = take(Uint16Array, meta.kpTotal), kpW = take(Float32Array, meta.kpTotal);
  const weights = new Float32Array(V * 6);
  for (let i = 0; i < V; i++) { let s = 0; for (let k = 0; k < 6; k++) s += weightsU8[i * 6 + k]; for (let k = 0; k < 6; k++) weights[i * 6 + k] = s ? weightsU8[i * 6 + k] / s : 0; }
  const keypoints = []; let o = 0;
  for (const e of meta.keypoints) { keypoints.push({ name: e.name, idx: kpIdx.subarray(o, o + e.n), w: kpW.subarray(o, o + e.n) }); o += e.n; }
  // topological order for FK (parents before children)
  const order = []; const seen = new Uint8Array(B);
  const visit = (b) => { if (seen[b]) return; if (parents[b] >= 0) visit(parents[b]); seen[b] = 1; order.push(b); };
  for (let b = 0; b < B; b++) visit(b);
  const boneIndex = Object.fromEntries(meta.boneLabels.map((n, i) => [n, i]));
  const headBone = boneIndex.head;
  const headWeight = new Float32Array(V);
  for (let i = 0; i < V; i++) for (let k = 0; k < 6; k++) if (boneIdx[i * 6 + k] === headBone) headWeight[i] += weights[i * 6 + k];
  return new Anny({ meta, V, B, K, D, mean, comps, compScale, sigma, faces, weights, boneIdx, rolls, yAxis, degenerate, parents, order, kp: keypoints, boneIndex, headWeight });
}

const m3 = {
  mul: (a, b) => [a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8]],
  T: (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]],
  apply: (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]],
};
/** Rodrigues: rotation vector -> row-major 3x3. */
export function rotvecToMat(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const k = [r[0] / th, r[1] / th, r[2] / th], c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [c + k[0] * k[0] * C, k[0] * k[1] * C - k[2] * s, k[0] * k[2] * C + k[1] * s,
    k[1] * k[0] * C + k[2] * s, c + k[1] * k[1] * C, k[1] * k[2] * C - k[0] * s,
    k[2] * k[0] * C - k[1] * s, k[2] * k[1] * C + k[0] * s, c + k[2] * k[2] * C];
}
// rigid 4x4 as { R: 3x3 row-major, t: [3] }
const rig = {
  mul: (a, b) => ({ R: m3.mul(a.R, b.R), t: add(m3.apply(a.R, b.t), a.t) }),
  inv: (a) => { const Rt = m3.T(a.R); const t = m3.apply(Rt, a.t); return { R: Rt, t: [-t[0], -t[1], -t[2]] }; },
  apply: (a, v) => add(m3.apply(a.R, v), a.t),
};
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

export class Anny {
  constructor(d) { Object.assign(this, d); this.boneLabels = d.meta.boneLabels; }

  /** PCA shape: beta (K) -> { verts: Float32Array(V*3), heads: Float32Array(B*3), tails: Float32Array(B*3) } */
  shape(beta) {
    const { V, B, K, D, mean, comps, compScale } = this;
    const x = new Float32Array(D);
    x.set(mean);
    for (let k = 0; k < K; k++) {
      const b = (beta[k] || 0) * compScale[k]; if (!b) continue;
      const base = k * D;
      for (let i = 0; i < D; i++) x[i] += b * comps[base + i];
    }
    return { verts: x.subarray(0, V * 3), heads: x.subarray(V * 3, V * 3 + B * 3), tails: x.subarray(V * 3 + B * 3, D) };
  }

  /** Rest bone poses from heads, tails and rolls (Anny/Blender: the bone's y
   *  axis along head->tail by the smallest rotation, then the roll). */
  restPoses(heads, tails) {
    const { B, rolls, yAxis, degenerate } = this; const poses = [];
    for (let b = 0; b < B; b++) {
      const h = [heads[b * 3], heads[b * 3 + 1], heads[b * 3 + 2]], t = [tails[b * 3], tails[b * 3 + 1], tails[b * 3 + 2]];
      const v = sub(t, h); const n = Math.hypot(...v) || 1; const y = [v[0] / n, v[1] / n, v[2] / n];
      const cr = [y[1] * yAxis[2] - y[2] * yAxis[1], y[2] * yAxis[0] - y[0] * yAxis[2], y[0] * yAxis[1] - y[1] * yAxis[0]];
      const cn = Math.hypot(...cr); const dot = y[0] * yAxis[0] + y[1] * yAxis[1] + y[2] * yAxis[2];
      const angle = Math.atan2(cn, dot);
      let R;
      if (cn > 1e-8) { const ax = [cr[0] / cn, cr[1] / cn, cr[2] / cn]; R = rotvecToMat([-angle * ax[0], -angle * ax[1], -angle * ax[2]]); }
      else R = Array.from(degenerate);
      R = m3.mul(R, Array.from(rolls.subarray(b * 9, b * 9 + 9)));
      poses.push({ R: b === 0 ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : R, t: h });   // root keeps the identity orientation (Anny's root_identity_orientation)
    }
    return poses;
  }

  /** Forward kinematics with per-bone local rotations (rotation vectors in
   *  the bone's own rest frame, Anny's 'local-bone'): pose_b = pose_parent *
   *  rest_parent^-1 * rest_b * delta_b. Returns bone poses and the skinning
   *  transforms pose_b * rest_b^-1. */
  fk(rest, rotvecs) {
    const { B, parents, order } = this; const poses = new Array(B), transforms = new Array(B);
    for (const b of order) {
      const delta = { R: rotvecToMat(rotvecs ? [rotvecs[b * 3], rotvecs[b * 3 + 1], rotvecs[b * 3 + 2]] : [0, 0, 0]), t: [0, 0, 0] };
      const T = rig.mul(rest[b], delta);
      const p = parents[b];
      // Anny's 'local-bone': the chain hangs off the ROOT's rest pose inverse,
      // so the model comes out root-relative (pelvis at the origin)
      poses[b] = p >= 0 ? rig.mul(transforms[p], T) : rig.mul(rig.inv(rest[b]), T);
      transforms[b] = rig.mul(poses[b], rig.inv(rest[b]));
    }
    return { poses, transforms };
  }

  /** Linear blend skinning of rest vertices (Float32Array V*3). */
  skin(verts, transforms, out = null) {
    const { V, weights, boneIdx } = this; out = out || new Float32Array(V * 3);
    for (let i = 0; i < V; i++) {
      const x = verts[i * 3], y = verts[i * 3 + 1], z = verts[i * 3 + 2]; let ox = 0, oy = 0, oz = 0;
      for (let k = 0; k < 6; k++) {
        const w = weights[i * 6 + k]; if (!w) continue;
        const T = transforms[boneIdx[i * 6 + k]]; const R = T.R, t = T.t;
        ox += w * (R[0] * x + R[1] * y + R[2] * z + t[0]); oy += w * (R[3] * x + R[4] * y + R[5] * z + t[1]); oz += w * (R[6] * x + R[7] * y + R[8] * z + t[2]);
      }
      out[i * 3] = ox; out[i * 3 + 1] = oy; out[i * 3 + 2] = oz;
    }
    return out;
  }

  /** The whole model: shape + pose -> posed vertices, posed bone heads/tails. */
  forward(beta, rotvecs) {
    const s = this.shape(beta); const rest = this.restPoses(s.heads, s.tails); const { poses, transforms } = this.fk(rest, rotvecs);
    const verts = this.skin(s.verts, transforms);
    const B = this.B; const heads = new Float32Array(B * 3), tails = new Float32Array(B * 3);
    for (let b = 0; b < B; b++) {
      const h = rig.apply(transforms[b], [s.heads[b * 3], s.heads[b * 3 + 1], s.heads[b * 3 + 2]]);
      const t = rig.apply(transforms[b], [s.tails[b * 3], s.tails[b * 3 + 1], s.tails[b * 3 + 2]]);
      heads.set(h, b * 3); tails.set(t, b * 3);
    }
    return { verts, heads, tails, restVerts: s.verts, restHeads: s.heads, restTails: s.tails, rest, poses, transforms };
  }

  /** COCO keypoints from posed vertices: { name: [x,y,z] } */
  keypoints(verts) {
    const out = {};
    for (const k of this.kp) { let x = 0, y = 0, z = 0; for (let j = 0; j < k.idx.length; j++) { const i = k.idx[j] * 3, w = k.w[j]; x += w * verts[i]; y += w * verts[i + 1]; z += w * verts[i + 2]; } out[k.name] = [x, y, z]; }
    return out;
  }

  /** Vertices of the largest connected component (the outer skin; eyeballs
   *  and the mouth interior are separate closed meshes). */
  outerSurfaceMask() {
    if (this._outer) return this._outer;
    const { V, faces } = this; const parent = new Int32Array(V); for (let i = 0; i < V; i++) parent[i] = i;
    const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (let t = 0; t < faces.length; t += 3) { const a = find(faces[t]), b = find(faces[t + 1]), c = find(faces[t + 2]); parent[b] = a; parent[c] = find(a); }
    const count = new Map(); let best = -1, bestN = 0;
    for (let i = 0; i < V; i++) { const r = find(i); const n = (count.get(r) || 0) + 1; count.set(r, n); if (n > bestN) { bestN = n; best = r; } }
    const mask = new Uint8Array(V); for (let i = 0; i < V; i++) mask[i] = find(i) === best ? 1 : 0;
    return (this._outer = mask);
  }
}

/** Strip interior geometry from a surface's head: the mouth cavity (connected
 *  to the lips, so no component test finds it; its walls nearly touch, so a
 *  single normal ray misses it) and the eye-socket lining. A head VERTEX is
 *  interior when almost none of a set of rays from it escapes the mesh
 *  within `reach` — skin sees open space in half its directions, a cavity
 *  wall in none. Faces with an interior vertex go. Head only: elsewhere
 *  fingers and armpits would fail the test. Mutates and returns S. */
export function cullHeadInterior(S, headWeight, { reach = 0.06, cell = 0.02, minHead = 0.3, rays = 20, escapeMin = 0.2 } = {}) {   // reach, cell: the mesh's units
  const V = S.vertices, F = S.faces; const nF = F.length, nV = V.length;
  const headV = []; for (let i = 0; i < nV; i++) if (headWeight[i] > minHead) headV.push(i);
  if (!headV.length) return S;
  const headF = []; for (let i = 0; i < nF; i++) { const f = F[i]; if (headWeight[f[0]] > minHead || headWeight[f[1]] > minHead || headWeight[f[2]] > minHead) headF.push(i); }
  const grid = new Map();
  for (const i of headF) { const [a, b, c] = F[i]; const xs = [V[a][0], V[b][0], V[c][0]], ys = [V[a][1], V[b][1], V[c][1]], zs = [V[a][2], V[b][2], V[c][2]];
    for (let x = Math.floor(Math.min(...xs) / cell); x <= Math.floor(Math.max(...xs) / cell); x++) for (let y = Math.floor(Math.min(...ys) / cell); y <= Math.floor(Math.max(...ys) / cell); y++) for (let z = Math.floor(Math.min(...zs) / cell); z <= Math.floor(Math.max(...zs) / cell); z++) { const k = `${x},${y},${z}`; let arr = grid.get(k); if (!arr) { arr = []; grid.set(k, arr); } arr.push(i); } }
  const vertFaces = Array.from({ length: nV }, () => []); for (const i of headF) for (const v of F[i]) vertFaces[v].push(i);
  const hits = (o, d, skip) => {
    const steps = Math.ceil(reach / cell) + 1; const seen = new Set(skip);
    for (let st = 0; st <= steps; st++) {
      const px = o[0] + d[0] * st * cell, py = o[1] + d[1] * st * cell, pz = o[2] + d[2] * st * cell;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const arr = grid.get(`${Math.floor(px / cell) + dx},${Math.floor(py / cell) + dy},${Math.floor(pz / cell) + dz}`); if (!arr) continue;
        for (const i of arr) {
          if (seen.has(i)) continue; seen.add(i);
          const [a, b, c] = F[i]; const e1 = [V[b][0] - V[a][0], V[b][1] - V[a][1], V[b][2] - V[a][2]], e2 = [V[c][0] - V[a][0], V[c][1] - V[a][1], V[c][2] - V[a][2]];
          const h = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]]; const det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2]; if (Math.abs(det) < 1e-12) continue;
          const f = 1 / det; const sv = [o[0] - V[a][0], o[1] - V[a][1], o[2] - V[a][2]]; const u = f * (sv[0] * h[0] + sv[1] * h[1] + sv[2] * h[2]); if (u < 0 || u > 1) continue;
          const q = [sv[1] * e1[2] - sv[2] * e1[1], sv[2] * e1[0] - sv[0] * e1[2], sv[0] * e1[1] - sv[1] * e1[0]]; const v = f * (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]); if (v < 0 || u + v > 1) continue;
          const t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]); if (t > 1e-6 && t < reach) return true;
        }
      }
    }
    return false;
  };
  // fixed, well-spread directions (Fibonacci sphere)
  const dirs = []; for (let i = 0; i < rays; i++) { const y = 1 - 2 * (i + 0.5) / rays, r = Math.sqrt(1 - y * y), ph = i * 2.399963; dirs.push([r * Math.cos(ph), y, r * Math.sin(ph)]); }
  const interior = new Uint8Array(nV); let nInt = 0;
  for (const v of headV) {
    let esc = 0; const o = V[v];
    for (const d of dirs) if (!hits(o, d, vertFaces[v])) esc++;
    if (esc / rays < escapeMin) { interior[v] = 1; nInt++; }
  }
  const dead = new Uint8Array(nF); let culled = 0;
  for (let i = 0; i < nF; i++) { const f = F[i]; if (interior[f[0]] || interior[f[1]] || interior[f[2]]) { dead[i] = 1; culled++; } }
  const used = new Uint8Array(nV); const faces = []; for (let i = 0; i < nF; i++) if (!dead[i]) { faces.push(F[i]); for (const v of F[i]) used[v] = 1; }
  const remap = new Int32Array(nV).fill(-1); let nv = 0; for (let i = 0; i < nV; i++) if (used[i]) remap[i] = nv++;
  S.vertices = V.filter((_, i) => used[i]); S.faces = faces.map((f) => f.map((v) => remap[v]));
  if (S.boneIndices) S.boneIndices = S.boneIndices.filter((_, i) => used[i]); if (S.boneWeights) S.boneWeights = S.boneWeights.filter((_, i) => used[i]);
  S.interiorCulled = { faces: culled, vertices: nV - nv, interiorVerts: nInt }; S.vertexRemap = remap;
  return S;
}

/** Frames: the recon's PLY frame (y down) <-> Anny's (z up): A = (px, pz, -py). */
export const plyToAnny = (p) => [p[0], p[2], -p[1]];
export const annyToPly = (a) => [a[0], -a[2], a[1]];
