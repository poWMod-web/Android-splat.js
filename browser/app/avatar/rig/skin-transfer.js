// Transfers skin weights from a skinned triangle mesh to gaussian splat centers:
// for each splat center, find the nearest triangle, interpolate its per-vertex
// joints/weights barycentrically, keep the top 4 influences.
// Pure JS (no engine deps) so it runs in node tests and the browser.

function fill6(o, x, y, z, u, v, w) {
  o[0] = x; o[1] = y; o[2] = z; o[3] = u; o[4] = v; o[5] = w;
  return o;
}

// Closest point on triangle abc to point p (Ericson, Real-Time Collision Detection).
// Returns [x, y, z, u, v, w] with barycentric coords for a, b, c. Pass `out`
// (length-6 array) to avoid the per-call allocation on hot paths.
export function closestPointOnTriangle(p, a, b, c, out) {
  const o = out || [0, 0, 0, 0, 0, 0];
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return fill6(o, a[0], a[1], a[2], 1, 0, 0);
  const bpx = p[0] - b[0], bpy = p[1] - b[1], bpz = p[2] - b[2];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return fill6(o, b[0], b[1], b[2], 0, 1, 0);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return fill6(o, a[0] + abx * v, a[1] + aby * v, a[2] + abz * v, 1 - v, v, 0);
  }
  const cpx = p[0] - c[0], cpy = p[1] - c[1], cpz = p[2] - c[2];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return fill6(o, c[0], c[1], c[2], 0, 0, 1);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return fill6(o, a[0] + acx * w, a[1] + acy * w, a[2] + acz * w, 1 - w, 0, w);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return fill6(o, b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w, b[2] + (c[2] - b[2]) * w, 0, 1 - w, w);
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return fill6(o,
    a[0] + abx * v + acx * w,
    a[1] + aby * v + acy * w,
    a[2] + abz * v + acz * w,
    1 - v - w, v, w);
}

// Uniform grid over triangle AABBs for nearest-triangle queries.
// Cell keys are packed ints (10 bits/axis, wrap at 1024 cells); key collisions
// from the wrap only add extra distance-tested candidates, never wrong results.
const packCell = (gx, gy, gz) => (gx & 1023) | ((gy & 1023) << 10) | ((gz & 1023) << 20);

export class TriangleGrid {
  constructor(verts, indices, cellSize = 0.05) {
    this.verts = verts;
    this.indices = indices;
    this.cell = cellSize;
    this.map = new Map(); // packed cell key -> tri index array
    const numTris = indices.length / 3;
    for (let t = 0; t < numTris; t++) {
      let minx = Infinity, miny = Infinity, minz = Infinity;
      let maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
      for (let k = 0; k < 3; k++) {
        const vi = indices[t * 3 + k] * 3;
        const x = verts[vi], y = verts[vi + 1], z = verts[vi + 2];
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < minz) minz = z; if (z > maxz) maxz = z;
      }
      const c = this.cell;
      for (let gx = Math.floor(minx / c); gx <= Math.floor(maxx / c); gx++)
        for (let gy = Math.floor(miny / c); gy <= Math.floor(maxy / c); gy++)
          for (let gz = Math.floor(minz / c); gz <= Math.floor(maxz / c); gz++) {
            const key = packCell(gx, gy, gz);
            let arr = this.map.get(key);
            if (!arr) { arr = []; this.map.set(key, arr); }
            arr.push(t);
          }
    }
    // per-query triangle visit stamps (replaces a fresh Set per query)
    this._stamp = new Uint32Array(numTris);
    this._queryId = 0;
    this._scratch = [0, 0, 0, 0, 0, 0];
  }

  // Nearest triangle to p. Returns {tri, dist, point: [x,y,z], bary: [u,v,w]} or null.
  // maxDist: stop expanding once no hit closer than this is possible — callers
  // that only care about "is anything within farDist" avoid the full ring scan.
  nearest(p, maxRings = 40, maxDist = Infinity) {
    const c = this.cell;
    const cx = Math.floor(p[0] / c), cy = Math.floor(p[1] / c), cz = Math.floor(p[2] / c);
    let best = null;
    const qid = ++this._queryId;
    const stamp = this._stamp;
    const scratch = this._scratch;
    const a = [0, 0, 0], b = [0, 0, 0], cc = [0, 0, 0];
    for (let ring = 0; ring <= maxRings; ring++) {
      // stop when the ring cannot contain anything closer than the current
      // best (or anything within maxDist at all)
      const limit = best ? Math.min(best.dist, maxDist) : maxDist;
      if ((ring - 1) * c > limit) break;
      let any = false;
      for (let gx = cx - ring; gx <= cx + ring; gx++)
        for (let gy = cy - ring; gy <= cy + ring; gy++)
          for (let gz = cz - ring; gz <= cz + ring; gz++) {
            // shell only
            if (Math.max(Math.abs(gx - cx), Math.abs(gy - cy), Math.abs(gz - cz)) !== ring) continue;
            const arr = this.map.get(packCell(gx, gy, gz));
            if (!arr) continue;
            any = true;
            for (const t of arr) {
              if (stamp[t] === qid) continue;
              stamp[t] = qid;
              const i0 = this.indices[t * 3] * 3, i1 = this.indices[t * 3 + 1] * 3, i2 = this.indices[t * 3 + 2] * 3;
              a[0] = this.verts[i0]; a[1] = this.verts[i0 + 1]; a[2] = this.verts[i0 + 2];
              b[0] = this.verts[i1]; b[1] = this.verts[i1 + 1]; b[2] = this.verts[i1 + 2];
              cc[0] = this.verts[i2]; cc[1] = this.verts[i2 + 1]; cc[2] = this.verts[i2 + 2];
              const r = closestPointOnTriangle(p, a, b, cc, scratch);
              const dx = r[0] - p[0], dy = r[1] - p[1], dz = r[2] - p[2];
              const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
              if (!best || d < best.dist) {
                best = { tri: t, dist: d, point: [r[0], r[1], r[2]], bary: [r[3], r[4], r[5]] };
              }
            }
          }
      if (ring > 2 && !any && best) break;
    }
    return best;
  }
}

// Bind splat centers to the skinned mesh.
// centers: Float32Array n*3 (same space as skinnedVerts)
// returns { indices: Float32Array n*4, weights: Float32Array n*4, dists: Float32Array n }
// farJoint: joint index to bind splats farther than farDist to (e.g. Hips), rigid.
// jointPos (optional Float32Array numJoints*3, same space): when given, far
// splats bind to their NEAREST joint instead of farJoint — keeps outliers like
// mascot ears/manes on the head rather than pinning them to the pelvis.
// Generator core: yields fractional progress every ~2k splats so callers can
// either drain it synchronously (bindSplats) or yield to the event loop between
// chunks for live progress UI (bindSplatsAsync).
// Outward direction from the bone segment a-b to p, normalized into `out`.
// The relighting normal for splats with no surface hit (hair, loose clothing,
// mascot parts): they hang off a bone rather than sitting on the mesh, so the
// best available "outward" is radially away from that bone's axis.
function normalFromSegment(out, px, py, pz, ax, ay, az, bx, by, bz) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const len2 = abx * abx + aby * aby + abz * abz;
  let t = len2 > 0 ? ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  let dx = px - (ax + abx * t), dy = py - (ay + aby * t), dz = pz - (az + abz * t);
  const l = Math.hypot(dx, dy, dz);
  // a splat sitting exactly on the bone axis has no radial direction — up is as
  // good a guess as any and keeps the vector unit-length
  if (l < 1e-9) { out[0] = 0; out[1] = 1; out[2] = 0; return out; }
  out[0] = dx / l; out[1] = dy / l; out[2] = dz / l;
  return out;
}

// squared distance from p to the segment a-b
function distToSegment2(px, py, pz, ax, ay, az, bx, by, bz) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const len2 = abx * abx + aby * aby + abz * abz;
  let t = len2 > 0 ? ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const dx = px - (ax + abx * t), dy = py - (ay + aby * t), dz = pz - (az + abz * t);
  return dx * dx + dy * dy + dz * dz;
}

// All-pairs hop distance over the skeleton tree (255 = unreachable). Bones may
// share weights on one splat only when kinematically close — spatial closeness
// (a hand resting at the hip in the scan pose) is not a reason to blend.
function jointHopMatrix(jointParent) {
  const nj = jointParent.length;
  const adj = Array.from({ length: nj }, () => []);
  for (let j = 0; j < nj; j++) {
    const par = jointParent[j];
    if (par >= 0) { adj[j].push(par); adj[par].push(j); }
  }
  const hops = new Uint8Array(nj * nj).fill(255);
  for (let src = 0; src < nj; src++) {
    hops[src * nj + src] = 0;
    const q = [src];
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi], d = hops[src * nj + u];
      if (d >= 254) continue;
      for (const v of adj[u]) {
        if (hops[src * nj + v] === 255) { hops[src * nj + v] = d + 1; q.push(v); }
      }
    }
  }
  return hops;
}

function* bindSplatsGen(centers, skinnedVerts, triIndices, meshJoints, meshWeights, opts = {}) {
  const {
    cellSize = 0.05, farDist = 0.25, farJoint = 0, jointPos = null,
    jointParent = null,        // per-joint parent joint index (-1 root): enables segment-based far binding
    smoothIters = 0,           // neighbor-smoothing passes over the final weights
    smoothRadius = 0,          // world-space neighbor radius for smoothing + propagation graph
    smoothAlpha = 0.5,         // 0..1: how far weights relax toward the neighborhood mean
    propagate = true,          // far splats inherit weights from surface-anchored splats via the neighbor graph
    mixHops = 2,               // max skeleton-tree distance between bones allowed to mix on/across splats (0 = off)
    jointBias = null,          // per-joint influence factor (effective dist = dist / bias): boost hands to
                               // claim held accessories, demote legs so a helmet at the thigh lets go
    meshNormals = null         // skinned vertex normals (glb.js skinNormals), same layout/space as
                               // skinnedVerts: enables the per-splat relighting normal in `normals`
  } = opts;
  const numJoints = jointPos ? jointPos.length / 3 : 0;
  const nj = jointParent ? jointParent.length : 0;
  const hopM = mixHops > 0 && nj ? jointHopMatrix(jointParent) : null;
  const compat = (a, b) => !hopM || hopM[a * nj + b] <= mixHops;
  const bias = (j) => (jointBias && jointBias[j]) || 1;
  // joints boosted past 1 compete with surface hits for nearby splats
  const boosted = [];
  if (jointBias) for (let j = 0; j < numJoints; j++) if (jointBias[j] > 1) boosted.push(j);
  const grid = new TriangleGrid(skinnedVerts, triIndices, cellSize);
  const n = centers.length / 3;
  const useGraph = smoothRadius > 0 && Number.isFinite(smoothRadius) && (smoothIters > 0 || propagate);
  // progress units: bind = n, adjacency = 0.8n, propagation + smoothing = 0.4n
  const total = useGraph ? n * 2.2 : n;
  let outIdx = new Float32Array(n * 4);
  let outWgt = new Float32Array(n * 4);
  const outDist = new Float32Array(n);
  const outNrm = meshNormals ? new Float32Array(n * 3) : null;
  const isFar = new Uint8Array(n);
  const p = [0, 0, 0];
  const nrmTmp = [0, 0, 0];
  const acc = new Map(); // joint -> weight accumulator
  for (let i = 0; i < n; i++) {
    p[0] = centers[i * 3]; p[1] = centers[i * 3 + 1]; p[2] = centers[i * 3 + 2];
    // bias-aware routing: the surface hit competes at dist/bias(dominant) and
    // boosted bones' capsules can steal splats the mesh would otherwise claim
    const maxBias = jointBias ? 4 : 1;
    const hit = grid.nearest(p, 40, farDist * maxBias);
    let toFar = !hit || hit.dist > farDist;
    if (hit && jointBias && !toFar) {
      // dominant joint of the surface hit (highest barycentric-blended weight)
      acc.clear();
      for (let k = 0; k < 3; k++) {
        const vi = triIndices[hit.tri * 3 + k];
        const bw = hit.bary[k];
        if (bw === 0) continue;
        for (let s2 = 0; s2 < 4; s2++) {
          const w = meshWeights[vi * 4 + s2] * bw;
          if (w === 0) continue;
          const j = meshJoints[vi * 4 + s2];
          acc.set(j, (acc.get(j) || 0) + w);
        }
      }
      let domJ = 0, domW = -1;
      for (const [j, w] of acc) if (w > domW) { domW = w; domJ = j; }
      const dsEff = hit.dist / bias(domJ);
      if (dsEff > farDist) toFar = true; // demoted bone's surface lets go
      else if (boosted.length && jointPos) {
        for (const q of boosted) {
          if (compat(domJ, q)) continue; // same region — no steal, mesh weights are better
          const par = jointParent ? jointParent[q] : -1;
          let d2;
          const qx = jointPos[q * 3], qy = jointPos[q * 3 + 1], qz = jointPos[q * 3 + 2];
          if (par >= 0) {
            d2 = distToSegment2(p[0], p[1], p[2],
              jointPos[par * 3], jointPos[par * 3 + 1], jointPos[par * 3 + 2], qx, qy, qz);
          } else {
            const dx = qx - p[0], dy = qy - p[1], dz = qz - p[2];
            d2 = dx * dx + dy * dy + dz * dz;
          }
          if (Math.sqrt(d2) / bias(q) < dsEff) { toFar = true; break; } // boosted bone wins
        }
      }
    }
    if (toFar) {
      isFar[i] = 1;
      if (jointPos) {
        // Soft-bind to the nearest bone SEGMENTS (joint->parent capsules) with
        // an inverse-square blend of the 4 closest. Loose geometry (clothing,
        // hair, mascot parts) then bridges between bones instead of snapping
        // rigidly to a single joint. Without jointParent, joints are points.
        const bi = [-1, -1, -1, -1], bd = [Infinity, Infinity, Infinity, Infinity];
        for (let q = 0; q < numJoints; q++) {
          const qx = jointPos[q * 3], qy = jointPos[q * 3 + 1], qz = jointPos[q * 3 + 2];
          const par = jointParent ? jointParent[q] : -1;
          let d2;
          if (par >= 0) {
            d2 = distToSegment2(p[0], p[1], p[2],
              jointPos[par * 3], jointPos[par * 3 + 1], jointPos[par * 3 + 2], qx, qy, qz);
          } else {
            const dx = qx - p[0], dy = qy - p[1], dz = qz - p[2];
            d2 = dx * dx + dy * dy + dz * dz;
          }
          const bq = bias(q);
          if (bq !== 1) d2 /= bq * bq;
          for (let k = 0; k < 4; k++) {
            if (d2 < bd[k]) {
              for (let m = 3; m > k; m--) { bd[m] = bd[m - 1]; bi[m] = bi[m - 1]; }
              bd[k] = d2; bi[k] = q;
              break;
            }
          }
        }
        let sum = 0;
        const w4 = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) {
          if (bi[k] < 0) continue;
          // only capsules kinematically NEAR the closest one may blend in — a hand
          // capsule passing near the hip contributes nothing to hip fluff
          if (k > 0 && !compat(bi[0], bi[k])) { bi[k] = -1; continue; }
          w4[k] = 1 / (bd[k] + 1e-6);
          sum += w4[k];
        }
        for (let k = 0; k < 4; k++) {
          outIdx[i * 4 + k] = bi[k] >= 0 ? bi[k] : 0;
          outWgt[i * 4 + k] = bi[k] >= 0 ? w4[k] / sum : 0;
        }
        // relighting normal: radially out from the winning bone segment
        if (outNrm && bi[0] >= 0) {
          const q = bi[0], par = jointParent ? jointParent[q] : -1;
          const qx = jointPos[q * 3], qy = jointPos[q * 3 + 1], qz = jointPos[q * 3 + 2];
          const ax = par >= 0 ? jointPos[par * 3] : qx;
          const ay = par >= 0 ? jointPos[par * 3 + 1] : qy;
          const az = par >= 0 ? jointPos[par * 3 + 2] : qz;
          normalFromSegment(nrmTmp, p[0], p[1], p[2], ax, ay, az, qx, qy, qz);
          outNrm[i * 3] = nrmTmp[0]; outNrm[i * 3 + 1] = nrmTmp[1]; outNrm[i * 3 + 2] = nrmTmp[2];
        }
      } else {
        outIdx[i * 4] = farJoint;
        outWgt[i * 4] = 1;
      }
      outDist[i] = hit ? hit.dist : Infinity;
    } else {
      acc.clear();
      for (let k = 0; k < 3; k++) {
        const vi = triIndices[hit.tri * 3 + k];
        const bw = hit.bary[k];
        if (bw === 0) continue;
        for (let s = 0; s < 4; s++) {
          const w = meshWeights[vi * 4 + s] * bw;
          if (w === 0) continue;
          const j = meshJoints[vi * 4 + s];
          acc.set(j, (acc.get(j) || 0) + w);
        }
      }
      const top = [...acc.entries()].sort((x, y) => y[1] - x[1]).slice(0, 4);
      let sum = 0;
      for (const [, w] of top) sum += w;
      for (let k = 0; k < 4; k++) {
        outIdx[i * 4 + k] = k < top.length ? top[k][0] : 0;
        outWgt[i * 4 + k] = k < top.length ? top[k][1] / sum : 0;
      }
      // relighting normal: the hit triangle's SMOOTH normal, barycentrically
      // blended the same way the weights are. Deliberately not the face normal —
      // lighting off flat facets shows every triangle of the rig.
      if (outNrm) {
        let nx = 0, ny = 0, nz = 0;
        for (let k = 0; k < 3; k++) {
          const vi = triIndices[hit.tri * 3 + k] * 3;
          const bw = hit.bary[k];
          nx += meshNormals[vi] * bw;
          ny += meshNormals[vi + 1] * bw;
          nz += meshNormals[vi + 2] * bw;
        }
        const l = Math.hypot(nx, ny, nz) || 1;
        outNrm[i * 3] = nx / l; outNrm[i * 3 + 1] = ny / l; outNrm[i * 3 + 2] = nz / l;
      }
      outDist[i] = hit.dist;
    }
    if ((i & 2047) === 2047) yield (i + 1) / total;
  }

  // Neighbor smoothing: relax each splat's weights toward the kernel-weighted
  // mean of splats within smoothRadius (linear falloff), then re-truncate to
  // the top 4. Softens hard seams between bone regions — especially where far
  // splats meet surface-bound ones.
  // Structured for speed: the neighbor adjacency is gathered ONCE (integer
  // cell keys, distance-filtered so key collisions from the 1024-cell wrap are
  // harmless), then every iteration is a cheap flat-array accumulation.
  if (useGraph) {
    const r = smoothRadius, r2 = r * r, rInv = 1 / r;
    // dominant joint per splat: the adjacency (smoothing + propagation) only
    // connects splats whose dominant bones are kinematically compatible, so a
    // hand resting against the hip never smoothes/leaks into hip splats
    const dom = hopM ? new Uint8Array(n) : null;
    if (dom) for (let i = 0; i < n; i++) dom[i] = outIdx[i * 4];
    // cap neighbors per splat: an oversized radius (huge auto-fit scales, e.g.
    // mm-unit scans) would otherwise build an O(n^2) adjacency and blow past
    // the max array length
    const MAX_NBRS = 64;
    const ckey = (cx, cy, cz) => (cx & 1023) | ((cy & 1023) << 10) | ((cz & 1023) << 20);
    const cellMap = new Map();
    for (let i = 0; i < n; i++) {
      const key = ckey(Math.floor(centers[i * 3] * rInv), Math.floor(centers[i * 3 + 1] * rInv), Math.floor(centers[i * 3 + 2] * rInv));
      let arr = cellMap.get(key);
      if (!arr) cellMap.set(key, arr = []);
      arr.push(i);
    }
    // adjacency: nbr indices + kernel values per splat, CSR-style
    const nStart = new Uint32Array(n + 1);
    const nbrJ = [], nbrK = [];
    for (let i = 0; i < n; i++) {
      nStart[i] = nbrJ.length;
      const px = centers[i * 3], py = centers[i * 3 + 1], pz = centers[i * 3 + 2];
      const cx = Math.floor(px * rInv), cy = Math.floor(py * rInv), cz = Math.floor(pz * rInv);
      let count = 0;
      outer:
      for (let gx = cx - 1; gx <= cx + 1; gx++)
        for (let gy = cy - 1; gy <= cy + 1; gy++)
          for (let gz = cz - 1; gz <= cz + 1; gz++) {
            const arr = cellMap.get(ckey(gx, gy, gz));
            if (!arr) continue;
            for (const j of arr) {
              if (j === i) continue;
              if (dom && !compat(dom[i], dom[j])) continue; // kinematically incompatible regions stay separate
              const dx = centers[j * 3] - px, dy = centers[j * 3 + 1] - py, dz = centers[j * 3 + 2] - pz;
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 > r2) continue;
              nbrJ.push(j);
              nbrK.push(1 - Math.sqrt(d2) * rInv);
              if (++count >= MAX_NBRS) break outer;
            }
          }
      if ((i & 2047) === 2047) yield (n + i * 0.8) / total;
    }
    nStart[n] = nbrJ.length;

    // Anchor propagation: far splats inherit the weights of the nearest
    // surface-anchored splat measured THROUGH the cloud (multi-source Dijkstra
    // over the neighbor graph). Attachment beats straight-line proximity: a
    // mascot head bulging past the rig's head bone still follows the head,
    // because its only connection to the body runs through the neck anchors.
    // Far splats no propagation front reaches (disconnected fluff) keep their
    // segment-distance fallback weights.
    if (propagate) {
      const dist = new Float64Array(n).fill(Infinity);
      const src = new Int32Array(n).fill(-1);
      // binary min-heap with lazy deletion
      const hD = [], hV = [];
      const hpush = (d, v) => {
        let i = hD.length;
        hD.push(d); hV.push(v);
        while (i > 0) {
          const par = (i - 1) >> 1;
          if (hD[par] <= hD[i]) break;
          const td = hD[par]; hD[par] = hD[i]; hD[i] = td;
          const tv = hV[par]; hV[par] = hV[i]; hV[i] = tv;
          i = par;
        }
      };
      const hpop = () => {
        const d = hD[0], v = hV[0];
        const ld = hD.pop(), lv = hV.pop();
        if (hD.length) {
          hD[0] = ld; hV[0] = lv;
          let i = 0;
          for (;;) {
            const l = i * 2 + 1, rr = l + 1;
            let m = i;
            if (l < hD.length && hD[l] < hD[m]) m = l;
            if (rr < hD.length && hD[rr] < hD[m]) m = rr;
            if (m === i) break;
            const td = hD[m]; hD[m] = hD[i]; hD[i] = td;
            const tv = hV[m]; hV[m] = hV[i]; hV[i] = tv;
            i = m;
          }
        }
        return [d, v];
      };
      for (let i = 0; i < n; i++) {
        if (!isFar[i]) { dist[i] = 0; src[i] = i; hpush(0, i); }
      }
      let settled = 0;
      while (hD.length) {
        const [d, v] = hpop();
        if (d > dist[v]) continue; // stale heap entry
        if (((++settled) & 8191) === 0) yield (n * 1.8 + Math.min(0.19, (settled / n) * 0.2) * n) / total;
        for (let nn = nStart[v]; nn < nStart[v + 1]; nn++) {
          const u = nbrJ[nn];
          const nd = d + (1 - nbrK[nn]) * r; // recover edge length from the kernel
          if (nd < dist[u]) { dist[u] = nd; src[u] = src[v]; hpush(nd, u); }
        }
      }
      // Weights only — NOT the normal. The anchor's normal belongs to its own
      // spot on the surface: hair on both sides of a head propagates from the
      // same scalp anchors and would end up lit as if it faced one way. The
      // segment-radial normal assigned above is cruder but points outward from
      // the right side of the bone.
      for (let i = 0; i < n; i++) {
        const a = src[i];
        if (!isFar[i] || a < 0 || a === i) continue;
        for (let s = 0; s < 4; s++) {
          outIdx[i * 4 + s] = outIdx[a * 4 + s];
          outWgt[i * 4 + s] = outWgt[a * 4 + s];
        }
      }
    }

    if (smoothIters > 0) {
    let maxJoint = 0;
    for (let i = 0; i < n * 4; i++) if (outIdx[i] > maxJoint) maxJoint = outIdx[i];
    const accW = new Float32Array(maxJoint + 1);
    const touched = [];
    let srcI = outIdx, srcW = outWgt;
    let dstI = new Float32Array(n * 4), dstW = new Float32Array(n * 4);
    for (let it = 0; it < smoothIters; it++) {
      for (let i = 0; i < n; i++) {
        let ksum = 0;
        for (let nn = nStart[i]; nn < nStart[i + 1]; nn++) ksum += nbrK[nn];
        if (ksum === 0) {
          for (let s = 0; s < 4; s++) { dstI[i * 4 + s] = srcI[i * 4 + s]; dstW[i * 4 + s] = srcW[i * 4 + s]; }
          continue;
        }
        // combined = (1-alpha) * self + alpha * neighborhood mean
        touched.length = 0;
        const nk = smoothAlpha / ksum;
        for (let nn = nStart[i]; nn < nStart[i + 1]; nn++) {
          const j = nbrJ[nn], k = nbrK[nn] * nk;
          for (let s = 0; s < 4; s++) {
            const w = srcW[j * 4 + s];
            if (w === 0) continue;
            const jj = srcI[j * 4 + s];
            if (accW[jj] === 0) touched.push(jj);
            accW[jj] += w * k;
          }
        }
        for (let s = 0; s < 4; s++) {
          const w = srcW[i * 4 + s];
          if (w === 0) continue;
          const jj = srcI[i * 4 + s];
          if (accW[jj] === 0) touched.push(jj);
          accW[jj] += (1 - smoothAlpha) * w;
        }
        // top-4 selection (insertion into 4 slots, descending)
        const ti = [0, 0, 0, 0], tw = [0, 0, 0, 0];
        for (const jj of touched) {
          const w = accW[jj];
          accW[jj] = 0; // reset scratch as we go
          for (let k = 0; k < 4; k++) {
            if (w > tw[k]) {
              for (let m = 3; m > k; m--) { tw[m] = tw[m - 1]; ti[m] = ti[m - 1]; }
              tw[k] = w; ti[k] = jj;
              break;
            }
          }
        }
        const sum = (tw[0] + tw[1] + tw[2] + tw[3]) || 1;
        for (let s = 0; s < 4; s++) {
          dstI[i * 4 + s] = ti[s];
          dstW[i * 4 + s] = tw[s] / sum;
        }
        if ((i & 8191) === 8191) yield (n * 2 + (it + i / n) * n * 0.2 / smoothIters) / total;
      }
      // swap buffers for the next iteration
      let t = srcI; srcI = dstI; dstI = t;
      t = srcW; srcW = dstW; dstW = t;
    }
    outIdx = srcI;
    outWgt = srcW;
    }
  }

  return { indices: outIdx, weights: outWgt, dists: outDist, normals: outNrm };
}

// Synchronous drain — same signature/behavior as always (node tests, exporter).
export function bindSplats(centers, skinnedVerts, triIndices, meshJoints, meshWeights, opts = {}) {
  const it = bindSplatsGen(centers, skinnedVerts, triIndices, meshJoints, meshWeights, opts);
  let r;
  do { r = it.next(); } while (!r.done);
  return r.value;
}

// Chunked drain: reports progress and yields to the event loop between chunks
// so the page keeps rendering and progress UI actually paints.
export async function bindSplatsAsync(centers, skinnedVerts, triIndices, meshJoints, meshWeights, opts = {}) {
  const { onProgress = null, ...rest } = opts;
  const it = bindSplatsGen(centers, skinnedVerts, triIndices, meshJoints, meshWeights, rest);
  let r;
  do {
    r = it.next();
    if (!r.done) {
      onProgress?.(r.value);
      // scheduler.yield avoids the nested-setTimeout 4ms clamp where available
      if (typeof scheduler !== 'undefined' && scheduler.yield) await scheduler.yield();
      else await new Promise((res) => setTimeout(res, 0));
    }
  } while (!r.done);
  return r.value;
}
