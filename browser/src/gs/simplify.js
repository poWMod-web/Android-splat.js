// simplify.js — training-free splat simplification, NanoGS method.
//
// Takes a finished model ({ data, n, sh, shK, dc } in trainer layout) and
// reduces it toward a target fraction: an opacity prune first, then passes
// of pairwise merges (kNN candidates, cheapest-first greedy matching,
// moment-matched replacements) until the target count is reached. No
// images, no retraining — this runs on exports, never inside the loop.
//
// Method reference: "NanoGS: Training-Free Gaussian Splat Simplification"
// (arXiv 2603.16103, https://github.com/RongLiu-Leo/NanoGS). Clean-room
// implementation: the pipeline shape (prune, kNN edges, Monte-Carlo KL
// merge cost plus color distance, greedy disjoint matching, moment-match
// merge) follows the published method, but every line below is original —
// nothing is copied from the upstream project (which is CC BY-NC 4.0).
//
// Cost model, briefly: a merge candidate (i, j) costs KL(p || q) + color,
// where p is the mass-weighted 2-Gaussian mixture and q the moment-matched
// single Gaussian. E_p[-log q] is closed form; E_p[log p] is Monte Carlo
// over draws from both components with ONE shared normal sequence (common
// random numbers, so pairwise comparisons share their noise). Masses are
// integrated opacity (2pi)^3/2 * op * sx*sy*sz. Color is squared distance
// in linear DC plus SH-rest coefficient space.

const SH_C0 = 0.28209479177387814;
const TAU_POW_15 = Math.pow(2 * Math.PI, 1.5);
const LOG2PI = Math.log(2 * Math.PI);
const DEG_OF_SHK = { 0: 0, 3: 1, 8: 2, 15: 3 };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const logit = (p) => Math.log(p / (1 - p));
const yieldTick = () => new Promise((r) => setTimeout(r, 0));

/** Deterministic RNG (32-bit hash stream) for the shared MC draws. */
function makeRng(seed) {
  let a = (seed >>> 0) || 0x9e3779b9;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function medianOf(values) {
  const s = Float32Array.from(values);
  s.sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

function quatToMat(w, x, y, z, R, o) {
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  R[o] = 1 - 2 * (yy + zz); R[o + 1] = 2 * (xy - wz);     R[o + 2] = 2 * (xz + wy);
  R[o + 3] = 2 * (xy + wz); R[o + 4] = 1 - 2 * (xx + zz); R[o + 5] = 2 * (yz - wx);
  R[o + 6] = 2 * (xz - wy); R[o + 7] = 2 * (yz + wx);     R[o + 8] = 1 - 2 * (xx + yy);
}

function det3(m, o) {
  return m[o] * (m[o + 4] * m[o + 8] - m[o + 5] * m[o + 7]) -
    m[o + 1] * (m[o + 3] * m[o + 8] - m[o + 5] * m[o + 6]) +
    m[o + 2] * (m[o + 3] * m[o + 7] - m[o + 4] * m[o + 6]);
}

/** Cyclic Jacobi eigensolver for symmetric 3x3: A holds eigenvals on its
 *  diagonal afterwards, V its eigenvectors as columns. Fixed sweep count
 *  keeps it deterministic. */
function eigenSym3(A, V) {
  V[0] = 1; V[1] = 0; V[2] = 0;
  V[3] = 0; V[4] = 1; V[5] = 0;
  V[6] = 0; V[7] = 0; V[8] = 1;
  for (let sweep = 0; sweep < 12; sweep++) {
    for (let pq = 0; pq < 3; pq++) {
      const p = pq === 2 ? 1 : 0, q = pq === 0 ? 1 : 2;
      const apq = A[p * 3 + q];
      if (Math.abs(apq) < 1e-18) continue;
      const app = A[p * 3 + p], aqq = A[q * 3 + q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = c * t;
      for (let r = 0; r < 3; r++) {
        const arp = A[r * 3 + p], arq = A[r * 3 + q];
        A[r * 3 + p] = c * arp - s * arq;
        A[r * 3 + q] = s * arp + c * arq;
      }
      for (let r = 0; r < 3; r++) {
        const apr = A[p * 3 + r], aqr = A[q * 3 + r];
        A[p * 3 + r] = c * apr - s * aqr;
        A[q * 3 + r] = s * apr + c * aqr;
      }
      for (let r = 0; r < 3; r++) {
        const vrp = V[r * 3 + p], vrq = V[r * 3 + q];
        V[r * 3 + p] = c * vrp - s * vrq;
        V[r * 3 + q] = s * vrp + c * vrq;
      }
    }
  }
}

/** Rotation matrix (orthonormal columns) -> quat (w,x,y,z), w canonicalized
 *  non-negative so identical inputs merge byte-identically. */
function matToQuat(R, o, q) {
  const tr = R[o] + R[o + 4] + R[o + 8];
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (R[o + 7] - R[o + 5]) / s;
    y = (R[o + 2] - R[o + 6]) / s;
    z = (R[o + 3] - R[o + 1]) / s;
  } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
    const s = Math.sqrt(1 + R[o] - R[o + 4] - R[o + 8]) * 2;
    w = (R[o + 7] - R[o + 5]) / s;
    x = 0.25 * s;
    y = (R[o + 1] + R[o + 3]) / s;
    z = (R[o + 2] + R[o + 6]) / s;
  } else if (R[o + 4] > R[o + 8]) {
    const s = Math.sqrt(1 + R[o + 4] - R[o] - R[o + 8]) * 2;
    w = (R[o + 2] - R[o + 6]) / s;
    x = (R[o + 1] + R[o + 3]) / s;
    y = 0.25 * s;
    z = (R[o + 5] + R[o + 7]) / s;
  } else {
    const s = Math.sqrt(1 + R[o + 8] - R[o] - R[o + 4]) * 2;
    w = (R[o + 3] - R[o + 1]) / s;
    x = (R[o + 2] + R[o + 6]) / s;
    y = (R[o + 5] + R[o + 7]) / s;
    z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  const f = (w < 0 ? -1 : 1) / n;
  q[0] = w * f; q[1] = x * f; q[2] = y * f; q[3] = z * f;
}

/** Decode trainer rows to linear working space. Scales exponentiate,
 *  opacity/DC sigmoid-activate ('sh' DC un-offsets instead); SH rest
 *  bands are linear coefficients already. */
function decodeModel(g) {
  const n = g.n;
  let shK = g.sh ? (g.shK ?? (n > 0 ? g.sh.length / (n * 3) : 0)) : 0;
  if (!Number.isInteger(shK) || shK < 0) throw new Error(`simplifyGaussians: bad shK ${shK} for n=${n}`);
  const pos = new Float64Array(n * 3);
  const scl = new Float64Array(n * 3);
  const quat = new Float64Array(n * 4);
  const op = new Float64Array(n);
  const col = new Float64Array(n * 3);
  const rest = shK ? new Float64Array(n * shK * 3) : null;
  const isSh = g.dc === 'sh';
  for (let i = 0; i < n; i++) {
    const b = i * 16;
    pos[i * 3] = g.data[b]; pos[i * 3 + 1] = g.data[b + 1]; pos[i * 3 + 2] = g.data[b + 2];
    scl[i * 3] = Math.max(Math.exp(g.data[b + 3]), 1e-12);
    scl[i * 3 + 1] = Math.max(Math.exp(g.data[b + 4]), 1e-12);
    scl[i * 3 + 2] = Math.max(Math.exp(g.data[b + 5]), 1e-12);
    const qn = Math.hypot(g.data[b + 6], g.data[b + 7], g.data[b + 8], g.data[b + 9]) || 1;
    quat[i * 4] = g.data[b + 6] / qn; quat[i * 4 + 1] = g.data[b + 7] / qn;
    quat[i * 4 + 2] = g.data[b + 8] / qn; quat[i * 4 + 3] = g.data[b + 9] / qn;
    op[i] = sigmoid(g.data[b + 13]);
    for (let c = 0; c < 3; c++) {
      col[i * 3 + c] = isSh
        ? SH_C0 * g.data[b + 10 + c] + 0.5
        : sigmoid(g.data[b + 10 + c]);
    }
    if (rest) rest.set(g.sh.subarray(i * shK * 3, (i + 1) * shK * 3), i * shK * 3);
  }
  return { n, shK, dc: g.dc || 'sigmoid', pos, scl, quat, op, col, rest, src: g };
}

function encodeModel(w, keep, merged) {
  const n = keep.length + merged.length;
  const data = new Float32Array(n * 16);
  const sh = w.shK ? new Float32Array(n * w.shK * 3) : null;
  let r = 0;
  // unmerged rows pass through bit-exactly — only merged rows recompute
  for (const i of keep) {
    data.set(w.src.data.subarray(i * 16, i * 16 + 16), r * 16);
    if (sh) sh.set(w.src.sh.subarray(i * w.shK * 3, (i + 1) * w.shK * 3), r * w.shK * 3);
    r++;
  }
  const isSh = w.dc === 'sh';
  for (const m of merged) {
    const b = r * 16;
    data[b] = m.pos[0]; data[b + 1] = m.pos[1]; data[b + 2] = m.pos[2];
    data[b + 3] = Math.log(clamp(m.scl[0], 1e-9, 1e9));
    data[b + 4] = Math.log(clamp(m.scl[1], 1e-9, 1e9));
    data[b + 5] = Math.log(clamp(m.scl[2], 1e-9, 1e9));
    data[b + 6] = m.quat[0]; data[b + 7] = m.quat[1];
    data[b + 8] = m.quat[2]; data[b + 9] = m.quat[3];
    for (let c = 0; c < 3; c++) {
      data[b + 10 + c] = isSh
        ? (clamp(m.col[c], 0, 1) - 0.5) / SH_C0
        : logit(clamp(m.col[c], 1e-4, 1 - 1e-4));
    }
    data[b + 13] = logit(clamp(m.op, 1e-4, 1 - 1e-4));
    data[b + 14] = 0; data[b + 15] = 0;
    if (sh) sh.set(m.rest, r * w.shK * 3);
    r++;
  }
  return { data, n, sh, shK: w.shK, dc: w.dc };
}

/** Uniform-grid kNN: cell targets ~8 splats; expanding rings stop once no
 *  unvisited cell can improve the top-k (plus degenerate guards). Returns
 *  the UNDIRECTED edge set — each pair emitted once via the asymmetric
 *  rule (j > i, or i absent from j's own list), no hash-set blowup. */
function knnEdges(pos, n, k, onStatus) {
  if (n < 2) return new Uint32Array(0);
  let loX = Infinity, loY = Infinity, loZ = Infinity;
  let hiX = -Infinity, hiY = -Infinity, hiZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    if (x < loX) loX = x; if (x > hiX) hiX = x;
    if (y < loY) loY = y; if (y > hiY) hiY = y;
    if (z < loZ) loZ = z; if (z > hiZ) hiZ = z;
  }
  const extent = Math.max(hiX - loX, hiY - loY, hiZ - loZ, 1e-9);
  const cell = extent / Math.cbrt(Math.max(1, n / 8));
  const keyOf = (ix, iy, iz) => ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
  const grid = new Map();
  const cellOf = new Int32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const ix = Math.floor((pos[i * 3] - loX) / cell);
    const iy = Math.floor((pos[i * 3 + 1] - loY) / cell);
    const iz = Math.floor((pos[i * 3 + 2] - loZ) / cell);
    cellOf[i * 3] = ix; cellOf[i * 3 + 1] = iy; cellOf[i * 3 + 2] = iz;
    const key = keyOf(ix, iy, iz);
    let bucket = grid.get(key);
    if (!bucket) grid.set(key, (bucket = []));
    bucket.push(i);
  }
  const kEff = Math.min(k, n - 1);
  const lists = new Array(n);
  const tickEvery = Math.max(1, Math.floor(n / 20));
  for (let i = 0; i < n; i++) {
    if (onStatus && i % tickEvery === 0) onStatus(`neighbors ${i}/${n}`);
    const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    const best = [];   // [d2, j] ascending, capped at kEff + 1
    const consider = (j) => {
      if (j === i) return;
      const dx = pos[j * 3] - px, dy = pos[j * 3 + 1] - py, dz = pos[j * 3 + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (best.length > kEff && d2 >= best[best.length - 1][0]) return;
      let at = best.length;
      while (at > 0 && best[at - 1][0] > d2) at--;
      best.splice(at, 0, [d2, j]);
      if (best.length > kEff + 1) best.pop();
    };
    let scanned = 0;
    for (let ring = 0; ring <= 8; ring++) {
      const cx = cellOf[i * 3], cy = cellOf[i * 3 + 1], cz = cellOf[i * 3 + 2];
      for (let ax = -ring; ax <= ring; ax++) {
        for (let ay = -ring; ay <= ring; ay++) {
          for (let az = -ring; az <= ring; az++) {
            if (Math.max(Math.abs(ax), Math.abs(ay), Math.abs(az)) !== ring) continue;
            const bucket = grid.get(keyOf(cx + ax, cy + ay, cz + az));
            if (!bucket) continue;
            for (const j of bucket) {
              consider(j);
              if (++scanned > 4096) break;
            }
            if (scanned > 4096) break;
          }
          if (scanned > 4096) break;
        }
        if (scanned > 4096) break;
      }
      const worst = best.length > kEff ? best[best.length - 1][0] : Infinity;
      if (ring >= 1 && ring * ring * cell * cell > worst) break;   // rings beyond cannot improve
      if (scanned > 4096) break;
    }
    lists[i] = best.map(([, j]) => j);
  }
  const out = [];
  const inList = (list, v) => {
    for (let t = 0; t < list.length; t++) if (list[t] === v) return true;
    return false;
  };
  for (let i = 0; i < n; i++) {
    for (const j of lists[i]) {
      if (j > i || (j < i && !inList(lists[j], i))) out.push(i, j);
    }
  }
  return Uint32Array.from(out);
}

/** Per-splat cache: rotation, covariance, log-determinant, opacity mass. */
function buildCache(w, epsCov) {
  const n = w.n;
  const R = new Float64Array(n * 9);
  const v = new Float64Array(n * 3);
  const inv = new Float64Array(n * 3);
  const logdet = new Float64Array(n);
  const sigma = new Float64Array(n * 9);
  const mass = new Float64Array(n);
  const Rm = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const vx = w.scl[i * 3] ** 2 + epsCov;
    const vy = w.scl[i * 3 + 1] ** 2 + epsCov;
    const vz = w.scl[i * 3 + 2] ** 2 + epsCov;
    v[i * 3] = vx; v[i * 3 + 1] = vy; v[i * 3 + 2] = vz;
    inv[i * 3] = 1 / Math.max(vx, 1e-30);
    inv[i * 3 + 1] = 1 / Math.max(vy, 1e-30);
    inv[i * 3 + 2] = 1 / Math.max(vz, 1e-30);
    logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));
    quatToMat(w.quat[i * 4], w.quat[i * 4 + 1], w.quat[i * 4 + 2], w.quat[i * 4 + 3], Rm, 0);
    R.set(Rm, i * 9);
    // sigma = R diag(v) R^T, row by row
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        sigma[i * 9 + a * 3 + b] =
          Rm[a * 3] * vx * Rm[b * 3] + Rm[a * 3 + 1] * vy * Rm[b * 3 + 1] + Rm[a * 3 + 2] * vz * Rm[b * 3 + 2];
      }
    }
    mass[i] = TAU_POW_15 * w.op[i] * w.scl[i * 3] * w.scl[i * 3 + 1] * w.scl[i * 3 + 2] + 1e-12;
  }
  return { R, v, inv, logdet, sigma, mass };
}

/** Moment-matched single Gaussian for the pair: mass-weighted mean and
 *  covariance plus the between-mean spread. Shared by cost and merge so
 *  both rank and build the same replacement. */
function momentMatch(w, cache, i, j, epsCov, out) {
  const wi = cache.mass[i], wj = cache.mass[j];
  const pi = clamp(wi / (wi + wj), 1e-12, 1 - 1e-12), pj = 1 - pi;
  const mx = pi * w.pos[i * 3] + pj * w.pos[j * 3];
  const my = pi * w.pos[i * 3 + 1] + pj * w.pos[j * 3 + 1];
  const mz = pi * w.pos[i * 3 + 2] + pj * w.pos[j * 3 + 2];
  const dix = w.pos[i * 3] - mx, diy = w.pos[i * 3 + 1] - my, diz = w.pos[i * 3 + 2] - mz;
  const djx = w.pos[j * 3] - mx, djy = w.pos[j * 3 + 1] - my, djz = w.pos[j * 3 + 2] - mz;
  const S = out.S;
  for (let a = 0; a < 9; a++) S[a] = pi * cache.sigma[i * 9 + a] + pj * cache.sigma[j * 9 + a];
  S[0] += pi * dix * dix + pj * djx * djx;
  S[1] += pi * dix * diy + pj * djx * djy;
  S[2] += pi * dix * diz + pj * djx * djz;
  S[3] += pi * diy * dix + pj * djy * djx;
  S[4] += pi * diy * diy + pj * djy * djy;
  S[5] += pi * diy * diz + pj * djy * djz;
  S[6] += pi * diz * dix + pj * djz * djx;
  S[7] += pi * diz * diy + pj * djz * djy;
  S[8] += pi * diz * diz + pj * djz * djz;
  const s01 = 0.5 * (S[1] + S[3]), s02 = 0.5 * (S[2] + S[6]), s12 = 0.5 * (S[5] + S[7]);
  S[1] = S[3] = s01; S[2] = S[6] = s02; S[5] = S[7] = s12;
  S[0] += epsCov; S[4] += epsCov; S[8] += epsCov;
  out.pi = pi; out.pj = pj; out.mx = mx; out.my = my; out.mz = mz;
  return out;
}

/** Merge cost: Monte-Carlo KL(2-mixture || moment-match) plus squared
 *  color distance. The normal draws Z are shared across every edge
 *  (common random numbers) so rankings compare signal, not noise. */
function edgeCost(w, cache, i, j, Z, epsCov, lamGeo, lamColor, scratch) {
  const mm = momentMatch(w, cache, i, j, epsCov, scratch);
  const logdetm = Math.log(Math.max(det3(mm.S, 0), 1e-30));
  const negLogQ = 0.5 * (3 * LOG2PI + logdetm + 3);
  const logPi = Math.log(mm.pi), logPj = Math.log(mm.pj);
  const stdi = [Math.sqrt(cache.v[i * 3]), Math.sqrt(cache.v[i * 3 + 1]), Math.sqrt(cache.v[i * 3 + 2])];
  const stdj = [Math.sqrt(cache.v[j * 3]), Math.sqrt(cache.v[j * 3 + 1]), Math.sqrt(cache.v[j * 3 + 2])];
  const Ri = cache.R, Rj = cache.R;
  const logN = (x, y, z, c) => {
    const dx = x - w.pos[c * 3], dy = y - w.pos[c * 3 + 1], dz = z - w.pos[c * 3 + 2];
    const o = c * 9;
    const y0 = Rj[o] * dx + Rj[o + 3] * dy + Rj[o + 6] * dz;
    const y1 = Rj[o + 1] * dx + Rj[o + 4] * dy + Rj[o + 7] * dz;
    const y2 = Rj[o + 2] * dx + Rj[o + 5] * dy + Rj[o + 8] * dz;
    const m2 = y0 * y0 * cache.inv[c * 3] + y1 * y1 * cache.inv[c * 3 + 1] + y2 * y2 * cache.inv[c * 3 + 2];
    return -0.5 * (3 * LOG2PI + cache.logdet[c] + m2);
  };
  const logAdd = (a, b) => (a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));
  let sumI = 0, sumJ = 0;
  for (let s = 0; s < Z.length; s += 3) {
    const z0 = Z[s], z1 = Z[s + 1], z2 = Z[s + 2];
    const oi = i * 9, oj = j * 9;
    const xix = w.pos[i * 3] + z0 * stdi[0] * Ri[oi] + z1 * stdi[1] * Ri[oi + 1] + z2 * stdi[2] * Ri[oi + 2];
    const xiy = w.pos[i * 3 + 1] + z0 * stdi[0] * Ri[oi + 3] + z1 * stdi[1] * Ri[oi + 4] + z2 * stdi[2] * Ri[oi + 5];
    const xiz = w.pos[i * 3 + 2] + z0 * stdi[0] * Ri[oi + 6] + z1 * stdi[1] * Ri[oi + 7] + z2 * stdi[2] * Ri[oi + 8];
    const xjx = w.pos[j * 3] + z0 * stdj[0] * Ri[oj] + z1 * stdj[1] * Ri[oj + 1] + z2 * stdj[2] * Ri[oj + 2];
    const xjy = w.pos[j * 3 + 1] + z0 * stdj[0] * Ri[oj + 3] + z1 * stdj[1] * Ri[oj + 4] + z2 * stdj[2] * Ri[oj + 5];
    const xjz = w.pos[j * 3 + 2] + z0 * stdj[0] * Ri[oj + 6] + z1 * stdj[1] * Ri[oj + 7] + z2 * stdj[2] * Ri[oj + 8];
    sumI += logAdd(logPi + logN(xix, xiy, xiz, i), logPj + logN(xix, xiy, xiz, j));
    sumJ += logAdd(logPi + logN(xjx, xjy, xjz, i), logPj + logN(xjx, xjy, xjz, j));
  }
  const draws = Z.length / 3;
  const geo = mm.pi * (sumI / draws) + mm.pj * (sumJ / draws) + negLogQ;
  let color = 0;
  for (let c = 0; c < 3; c++) {
    const d = w.col[i * 3 + c] - w.col[j * 3 + c];
    color += d * d;
  }
  if (w.rest) {
    for (let k = 0; k < w.shK * 3; k++) {
      const d = w.rest[i * w.shK * 3 + k] - w.rest[j * w.shK * 3 + k];
      color += d * d;
    }
  }
  return lamGeo * geo + lamColor * color;
}

/** Greedy disjoint matching over cost-ranked edges, up to P pairs. */
function greedyPairs(costs, edgeU, edgeV, n, P) {
  const m = costs.length;
  const idx = new Uint32Array(m);
  let valid = 0;
  for (let e = 0; e < m; e++) if (Number.isFinite(costs[e])) idx[valid++] = e;
  const order = Array.prototype.slice.call(idx.subarray(0, valid));
  order.sort((a, b) => (costs[a] < costs[b] ? -1 : costs[a] > costs[b] ? 1 : a - b));
  const used = new Uint8Array(n);
  const pairs = [];
  for (const e of order) {
    const u = edgeU[e], v = edgeV[e];
    if (used[u] || used[v]) continue;
    used[u] = 1; used[v] = 1;
    pairs.push([u, v]);
    if (P != null && pairs.length >= P) break;
  }
  return pairs;
}

/** Build the merged replacement: moment-matched shape eigendecomposed
 *  back to scales + rotation, mass-weighted color, and the smaller of
 *  mass-conserving vs compositing-union opacity (never brighter than the
 *  pair's coincident peak, never more massive than its parts). */
function mergeOne(w, cache, i, j, epsCov, scratch) {
  const mm = momentMatch(w, cache, i, j, epsCov, scratch);
  const A = Float64Array.from(mm.S);
  const V = new Float64Array(9);
  eigenSym3(A, V);
  // eigensolver convergence failure (~never) falls back to the heavier twin
  let lam = [A[0], A[4], A[8]];
  if (!(lam[0] > 0 && lam[1] > 0 && lam[2] > 0)) {
    const heavy = mm.pi >= 0.5 ? i : j;
    return {
      pos: [w.pos[heavy * 3], w.pos[heavy * 3 + 1], w.pos[heavy * 3 + 2]],
      scl: [w.scl[heavy * 3], w.scl[heavy * 3 + 1], w.scl[heavy * 3 + 2]],
      quat: [w.quat[heavy * 4], w.quat[heavy * 4 + 1], w.quat[heavy * 4 + 2], w.quat[heavy * 4 + 3]],
      op: w.op[heavy],
      col: [w.col[heavy * 3], w.col[heavy * 3 + 1], w.col[heavy * 3 + 2]],
      rest: w.rest ? Array.from(w.rest.subarray(heavy * w.shK * 3, (heavy + 1) * w.shK * 3)) : null,
    };
  }
  // sort eigenpairs descending so the axes order deterministically
  const order = [0, 1, 2].sort((a, b) => lam[b] - lam[a]);
  lam = order.map((o) => lam[o]);
  const Rm = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    Rm[c] = V[order[c]];
    Rm[3 + c] = V[3 + order[c]];
    Rm[6 + c] = V[6 + order[c]];
  }
  if (det3(Rm, 0) < 0) { Rm[2] = -Rm[2]; Rm[5] = -Rm[5]; Rm[8] = -Rm[8]; }
  const quat = [0, 0, 0, 0];
  matToQuat(Rm, 0, quat);
  const scl = [Math.sqrt(lam[0]), Math.sqrt(lam[1]), Math.sqrt(lam[2])];
  const W = cache.mass[i] + cache.mass[j];
  const opMass = W / (TAU_POW_15 * scl[0] * scl[1] * scl[2]);
  const opUnion = 1 - (1 - w.op[i]) * (1 - w.op[j]);
  const op = clamp(Math.min(opMass, opUnion), 1e-6, 0.99);
  const col = [0, 1, 2].map((c) => mm.pi * w.col[i * 3 + c] + mm.pj * w.col[j * 3 + c]);
  let rest = null;
  if (w.rest) {
    rest = new Array(w.shK * 3);
    for (let k = 0; k < w.shK * 3; k++) {
      rest[k] = mm.pi * w.rest[i * w.shK * 3 + k] + mm.pj * w.rest[j * w.shK * 3 + k];
    }
  }
  return { pos: [mm.mx, mm.my, mm.mz], scl, quat, op, col, rest };
}

/**
 * Training-free simplification to a target fraction of the splats.
 *
 * g: { data (n*16 trainer rows), n, sh (n blocks of 3*shK, INRIA f_rest
 *      order, | null), shK, dc ('sigmoid' | 'sh') } — never mutated.
 * Returns the same layout with fewer rows (unmerged rows bit-exact).
 */
export async function simplifyGaussians(g, {
  ratio = 0.5,
  k = null,               // per-splat merge candidates (auto: 16, or 8 past 400k)
  opacityThreshold = 0.1,
  passCapRatio = 0.5,     // merges per pass, fraction of the original count
  lamGeo = 1,
  lamColor = 1,
  mcSamples = 4,
  seed = 0,
  epsCov = 1e-8,
  onStatus = () => {},
  onProgress = null,
} = {}) {
  if (!(ratio > 0 && ratio < 1)) throw new Error(`simplifyGaussians: ratio must be in (0, 1), got ${ratio}`);
  const progress = (p) => { if (onProgress) onProgress(p); };
  let w = decodeModel(g);
  const N0 = w.n;
  const target = Math.max(Math.ceil(N0 * ratio), 1);
  const kEff = k ?? (N0 > 400000 ? 8 : 16);
  onStatus(`simplify: ${N0} splats -> target ${target}`);
  progress({ stage: 'start', currentCount: N0, targetCount: target });

  // shared Monte-Carlo draws (common random numbers across every edge)
  const rng = makeRng(seed);
  const Z = new Float64Array(Math.max(1, mcSamples) * 3);
  for (let s = 0; s < Z.length; s += 2) {
    const u = Math.max(rng(), 1e-12), v = rng();
    const r = Math.sqrt(-2 * Math.log(u)), t = 2 * Math.PI * v;
    Z[s] = r * Math.cos(t);
    if (s + 1 < Z.length) Z[s + 1] = r * Math.sin(t);
  }

  // pass 0: opacity prune (at most half the model — the median caps it)
  if (w.n > 0) {
    const med = medianOf(w.op);
    const thr = Math.min(opacityThreshold, med);
    const keep = [];
    for (let i = 0; i < w.n; i++) if (w.op[i] >= thr) keep.push(i);
    onStatus(`simplify: opacity prune kept ${keep.length}/${w.n} (threshold ${thr.toFixed(4)})`);
    w = subsetWorking(w, keep);
    progress({ stage: 'prune', currentCount: w.n, targetCount: target });
  }

  const scratch = { S: new Float64Array(9), pi: 0, pj: 0, mx: 0, my: 0, mz: 0 };
  let pass = 0;
  while (w.n > target) {
    pass++;
    const N = w.n;
    onStatus(`simplify: pass ${pass} at ${N} splats`);
    progress({ stage: 'pass-start', pass, currentCount: N, targetCount: target });
    const cache = buildCache(w, epsCov);
    const flat = knnEdges(w.pos, N, kEff, (m) => onStatus(`simplify: pass ${pass} ${m}`));
    const m = flat.length / 2;
    if (!m) { onStatus('simplify: no merge candidates left'); break; }
    const edgeU = new Uint32Array(m), edgeV = new Uint32Array(m);
    for (let e = 0; e < m; e++) { edgeU[e] = flat[e * 2]; edgeV[e] = flat[e * 2 + 1]; }
    const costs = new Float64Array(m);
    const BLOCK = 8192;
    for (let e0 = 0; e0 < m; e0 += BLOCK) {
      const e1 = Math.min(m, e0 + BLOCK);
      for (let e = e0; e < e1; e++) {
        costs[e] = edgeCost(w, cache, edgeU[e], edgeV[e], Z, epsCov, lamGeo, lamColor, scratch);
      }
      onStatus(`simplify: pass ${pass} scored ${e1}/${m} pairs`);
      await yieldTick();
    }
    const need = N - target;
    const P = Math.min(need, Math.max(1, Math.floor(passCapRatio * N0)));
    const pairs = greedyPairs(costs, edgeU, edgeV, N, P);
    onStatus(`simplify: pass ${pass} merging ${pairs.length} pairs (need ${need})`);
    if (!pairs.length) break;
    const used = new Uint8Array(N);
    for (const [u, v] of pairs) { used[u] = 1; used[v] = 1; }
    const keep = [];
    for (let i = 0; i < N; i++) if (!used[i]) keep.push(i);
    const merged = [];
    for (let p = 0; p < pairs.length; p++) {
      merged.push(mergeOne(w, cache, pairs[p][0], pairs[p][1], epsCov, scratch));
      if (p % 4096 === 4095) await yieldTick();
    }
    // rebuild working space from the merged model (re-encoded, re-decoded:
    // the next pass caches off canonical rows, like a fresh load)
    const next = encodeModel(w, keep, merged);
    w = decodeModel(next);
    w.src = next;
    progress({ stage: 'pass', pass, currentCount: w.n, targetCount: target });
    await yieldTick();
  }

  onStatus(`simplify: final ${w.n} splats`);
  progress({ stage: 'done', currentCount: w.n, targetCount: target });
  // the working set already mirrors an encoded model — return it directly
  return { data: w.src.data, n: w.n, sh: w.src.sh, shK: w.shK, dc: w.dc };
}

/** Compact a working set to a kept subset (shares nothing with the input). */
function subsetWorking(w, keep) {
  const n = keep.length;
  const out = {
    n, shK: w.shK, dc: w.dc,
    pos: new Float64Array(n * 3), scl: new Float64Array(n * 3),
    quat: new Float64Array(n * 4), op: new Float64Array(n),
    col: new Float64Array(n * 3),
    rest: w.rest ? new Float64Array(n * w.shK * 3) : null,
    src: null,
  };
  keep.forEach((src, dst) => {
    out.pos.set(w.pos.subarray(src * 3, src * 3 + 3), dst * 3);
    out.scl.set(w.scl.subarray(src * 3, src * 3 + 3), dst * 3);
    out.quat.set(w.quat.subarray(src * 4, src * 4 + 4), dst * 4);
    out.op[dst] = w.op[src];
    out.col.set(w.col.subarray(src * 3, src * 3 + 3), dst * 3);
    if (out.rest) out.rest.set(w.rest.subarray(src * w.shK * 3, (src + 1) * w.shK * 3), dst * w.shK * 3);
  });
  // re-encode so src rows line up with the compacted set (encodeModel's
  // bit-exact passthrough needs row-aligned sources)
  const reenc = { data: new Float32Array(n * 16), n, sh: out.rest ? new Float32Array(n * w.shK * 3) : null, shK: w.shK, dc: w.dc };
  for (let i = 0; i < n; i++) {
    const orig = keep[i];
    reenc.data.set(w.src.data.subarray(orig * 16, orig * 16 + 16), i * 16);
    if (reenc.sh) reenc.sh.set(w.src.sh.subarray(orig * w.shK * 3, (orig + 1) * w.shK * 3), i * w.shK * 3);
  }
  out.src = reenc;
  return out;
}
