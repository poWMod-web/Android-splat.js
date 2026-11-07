// walk.js — the finished avatar, walking, in this tab.
//
// The rigger's runtime skins the splat on the GPU inside PlayCanvas
// (client_git/splat-rigger/src/splat-skin.js: bone deltas in a data texture, a
// work-buffer modifier shader). Splat.js has no engine to hang that on, so the
// same maths runs on the CPU here and the posed splats are written straight
// into the trainer's parameter buffer between frames. At ~90k splats that is
// ~5.6 MB a frame, which a desktop GPU takes without noticing; the tile sort is
// rebuilt by every render anyway, so the depth order comes out right for free.
//
// The chain, one frame:
//   clip (rig/walking_anim.glb, 1.03 s, rotations only)  ->  local rotations
//   + the fit's own bone offsets/lengths for everything the clip leaves alone
//   -> forward kinematics (glb.js computeNodeWorldMatrices)
//   -> bone delta   D_b = world_b(t) · invFit_b      (invFit is in the binding)
//   -> into splat space   Bm_b = M⁻¹ · D_b · M,  M = S(1/s)·T(−p)·R·F
//   -> per splat    S = Σ w_k·Bm_k ,  centre = S·centre ,  quat = q(S)⊗quat
//
// Scale, colour and opacity are left alone — the same as the reference shader.
import {
  parseAvatar, m4Identity, m4Mul, m4Invert, m4FromTRS, quatFromEulerDeg, quatMul,
} from './rig/glb.js';
import { loadClip, sampleClip, sampleMoves, poseWorld } from './anim.js';

const STRIDE = 16;   // gs/init.js: pos3, logScale3, quat4 (w,x,y,z), colour3, opacity, pad2

/** SBA1 (rig/export-binding-core.js buildBinding):
 *  "SBA1" | u32 metaLen | JSON meta | f32 invFit[bones*16] | u8 idx[n*4]
 *  | u8 wgt[n*4] | f32 centers[n*3] | i8 nrm[n*3] (version 3 only) */
export function parseBinding(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'SBA1') throw new Error('not an SBA1 binding');
  const metaLen = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + metaLen)).replace(/\0+$/, ''));
  const n = meta.numSplats, nb = meta.numBones;
  let o = 8 + metaLen;
  // the f32 blocks are only 4-byte aligned inside the file — copy, do not view
  const invFit = new Float32Array(u8.slice(o, o + nb * 64).buffer); o += nb * 64;
  const idx = u8.subarray(o, o + n * 4); o += n * 4;
  const wgt = u8.subarray(o, o + n * 4); o += n * 4;
  const centers = new Float32Array(u8.slice(o, o + n * 12).buffer); o += n * 12;
  return { meta, invFit, idx, wgt, centers, n, nb };
}

/** rig space -> splat (PLY) space, as the binding's meta.fit describes it:
 *  p_avatar = (R·F·p_ply − position) / scale, so M = S(1/s)·T(−p)·R·F */
function fitMatrix(fit) {
  const F = m4Identity(); F[5] = -1; F[10] = -1;                       // 180 degrees about X
  const yaw = fit && fit.rotation ? fit.rotation[1] || 0 : 0;
  const R = m4FromTRS(m4Identity(), [0, 0, 0], quatFromEulerDeg(0, yaw, 0), [1, 1, 1]);
  const p = (fit && fit.position) || [0, 0, 0];
  const T = m4Identity(); T[12] = -p[0]; T[13] = -p[1]; T[14] = -p[2];
  const s = 1 / ((fit && fit.scale) || 1);
  const S = m4Identity(); S[0] = s; S[5] = s; S[10] = s;
  const a = m4Identity(), b = m4Identity(), M = m4Identity();
  m4Mul(a, R, F); m4Mul(b, T, a); m4Mul(M, S, b);
  return M;
}

/**
 * @param {object} o
 * @param {import('../../src/session.js').Session} o.session  the finished (isolated) model
 * @param {Uint8Array} o.binding   SBA1 bytes, as bind.js produced them
 * @param {object} o.fit           the full fit JSON (its per-bone offsets pose
 *                                 everything the clip does not animate)
 * @param {ArrayBuffer} o.rigGlb   the rig the binding was authored against
 */
export async function createWalker({ session, binding, fit, rigGlb, log = () => {} }) {
  const B = parseBinding(binding);
  const avatar = parseAvatar(rigGlb);
  const clip = await loadClip();
  const rest = await session.exportRawState();          // the pose everything is skinned FROM
  const base = rest.data, rows = rest.n;

  // the binding indexes splats in PLY-export order, which DROPS dead splats and
  // compacts the rest — so a binding index is not a row in the trainer's buffer.
  // Positions survive the export untouched, so they are the key: exact float
  // bits, no tolerance, no search.
  const rowOf = new Int32Array(B.n).fill(-1);
  {
    const bits = new Int32Array(base.buffer, base.byteOffset, base.length);
    const map = new Map();
    for (let r = 0; r < rows; r++) {
      const k = `${bits[r * STRIDE]},${bits[r * STRIDE + 1]},${bits[r * STRIDE + 2]}`;
      if (!map.has(k)) map.set(k, r);
    }
    const cb = new Int32Array(B.centers.buffer);
    let hit = 0;
    for (let i = 0; i < B.n; i++) {
      const r = map.get(`${cb[i * 3]},${cb[i * 3 + 1]},${cb[i * 3 + 2]}`);
      if (r !== undefined) { rowOf[i] = r; hit++; }
    }
    log(`walk: ${hit.toLocaleString()} of ${B.n.toLocaleString()} bound splats matched to the live model (${rows.toLocaleString()} rows)`);
    if (hit < B.n * 0.5) throw new Error('the binding does not match this model');
  }

  // the bones, in the binding's own order, as nodes of this rig
  const nodeOf = B.meta.jointNames.map((nm) => {
    const nd = avatar.nodes.find((x) => x.name === nm);
    return nd ? nd.index : -1;
  });
  const missing = nodeOf.filter((i) => i < 0).length;
  if (missing) log(`walk: ${missing} of ${B.nb} bones are not in this rig — they will not move`);

  const fitOffsets = (fit && fit.offsets) || {};
  const lengths = (fit && fit.lengths) || (B.meta.fit && B.meta.fit.lengths) || null;
  const M = fitMatrix(B.meta.fit || fit);
  const Minv = m4Invert(m4Identity(), M);
  if (!Minv) throw new Error('the fit transform is singular');

  const posed = base.slice();                 // scale, colour and opacity ride along untouched
  const Bm = new Float32Array(B.nb * 16);     // per bone, splat space
  const tmpA = m4Identity(), tmpB = m4Identity(), D = m4Identity();
  // The clip's ROOT rotation (Mixamo's Armature, -90 degrees about X) and its
  // HIPS translation are both part of the pose: the bone locals are authored
  // under that root, and the hips keys put the figure back where the root
  // rotation took it from. Measured on Tom's binding (scratch/walk_probe.mjs),
  // averaged over the cycle, against the pose the model was trained in:
  //
  //     no root, rest hips   swing 80.5 deg   drift  5.4 %   (face down)
  //     root,    rest hips   swing 10.5 deg   drift 86.2 %   (upright, a body away)
  //     root,    clip hips   swing 10.5 deg   drift  2.4 %   <- this one
  //     no root, clip hips   swing 80.5 deg   drift 80.1 %
  //
  // The 10.5 degrees and the 2.4 % are the walk's own lean and sway. The rigger
  // asserts the same root by hand (main.js CLIP_ROOT_EULER_X). ?walkroot=0 and
  // ?walkhips=0 drop them, for a clip authored the other way round.
  const Q = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null;
  const useRoot = !Q || Q.get('walkroot') !== '0';
  const useHips = !Q || Q.get('walkhips') !== '0';

  /** the pose at `time`, as bone deltas in splat space */
  function poseBones(time) {
    const abs = sampleClip(clip, time);
    const rot = {}, trans = useHips ? sampleMoves(clip, time) : {};
    for (const n of avatar.nodes) {
      const c = abs[n.name];
      if (c && (useRoot || n.parent >= 0)) rot[n.name] = c;            // the clip's own local rotation
      else if (fitOffsets[n.name]) rot[n.name] = quatMul(n.rotation, fitOffsets[n.name]);   // the fit's pose, for what the clip leaves alone
    }
    const world = poseWorld(avatar, { rot, trans, lengths });
    for (let b = 0; b < B.nb; b++) {
      const ni = nodeOf[b];
      if (ni < 0) { Bm.set(m4Identity(), b * 16); continue; }
      m4Mul(D, world[ni], B.invFit.subarray(b * 16, b * 16 + 16));   // rig space
      m4Mul(tmpA, D, M);
      m4Mul(tmpB, Minv, tmpA);                                       // splat space
      Bm.set(tmpB, b * 16);
    }
  }

  /** one frame: skin every bound splat from `base` into `posed` */
  function skin() {
    const { idx, wgt, n } = B;
    for (let i = 0; i < n; i++) {
      const r = rowOf[i]; if (r < 0) continue;
      let a0 = 0, a1 = 0, a2 = 0, a4 = 0, a5 = 0, a6 = 0, a8 = 0, a9 = 0, a10 = 0, a12 = 0, a13 = 0, a14 = 0;
      for (let k = 0; k < 4; k++) {
        const w = wgt[i * 4 + k]; if (!w) continue;
        const f = w / 255, o = idx[i * 4 + k] * 16;
        a0 += f * Bm[o]; a1 += f * Bm[o + 1]; a2 += f * Bm[o + 2];
        a4 += f * Bm[o + 4]; a5 += f * Bm[o + 5]; a6 += f * Bm[o + 6];
        a8 += f * Bm[o + 8]; a9 += f * Bm[o + 9]; a10 += f * Bm[o + 10];
        a12 += f * Bm[o + 12]; a13 += f * Bm[o + 13]; a14 += f * Bm[o + 14];
      }
      const b = r * STRIDE;
      const x = base[b], y = base[b + 1], z = base[b + 2];
      posed[b] = a0 * x + a4 * y + a8 * z + a12;
      posed[b + 1] = a1 * x + a5 * y + a9 * z + a13;
      posed[b + 2] = a2 * x + a6 * y + a10 * z + a14;
      // the blend of several bones is not a rotation: orthonormalise its
      // columns (Gram-Schmidt, as the reference shader does) before taking a
      // quaternion, or the gaussians shear at every joint seam
      let r0x = a0, r0y = a1, r0z = a2;
      let l = Math.hypot(r0x, r0y, r0z) || 1; r0x /= l; r0y /= l; r0z /= l;
      let r1x = a4, r1y = a5, r1z = a6;
      const d = r0x * r1x + r0y * r1y + r0z * r1z;
      r1x -= d * r0x; r1y -= d * r0y; r1z -= d * r0z;
      l = Math.hypot(r1x, r1y, r1z) || 1; r1x /= l; r1y /= l; r1z /= l;
      const r2x = r0y * r1z - r0z * r1y, r2y = r0z * r1x - r0x * r1z, r2z = r0x * r1y - r0y * r1x;
      // quaternion of the rotation whose COLUMNS are r0, r1, r2
      let qx, qy, qz, qw;
      const tr = r0x + r1y + r2z;
      if (tr > 0) {
        const s = Math.sqrt(tr + 1) * 2;
        qw = 0.25 * s; qx = (r1z - r2y) / s; qy = (r2x - r0z) / s; qz = (r0y - r1x) / s;
      } else if (r0x > r1y && r0x > r2z) {
        const s = Math.sqrt(1 + r0x - r1y - r2z) * 2;
        qw = (r1z - r2y) / s; qx = 0.25 * s; qy = (r1x + r0y) / s; qz = (r2x + r0z) / s;
      } else if (r1y > r2z) {
        const s = Math.sqrt(1 + r1y - r0x - r2z) * 2;
        qw = (r2x - r0z) / s; qx = (r1x + r0y) / s; qy = 0.25 * s; qz = (r2y + r1z) / s;
      } else {
        const s = Math.sqrt(1 + r2z - r0x - r1y) * 2;
        qw = (r0y - r1x) / s; qx = (r2x + r0z) / s; qy = (r2y + r1z) / s; qz = 0.25 * s;
      }
      // the trainer keeps (w,x,y,z); pre-multiply, as the reference does
      const bw = base[b + 6], bx = base[b + 7], by = base[b + 8], bz = base[b + 9];
      posed[b + 6] = qw * bw - qx * bx - qy * by - qz * bz;
      posed[b + 7] = qw * bx + qx * bw + qy * bz - qz * by;
      posed[b + 8] = qw * by - qx * bz + qy * bw + qz * bx;
      posed[b + 9] = qw * bz + qx * by - qy * bx + qz * bw;
    }
  }

  const tr = session.trainer;
  const device = tr.device;
  let t0 = 0, last = -1, ms = 0, frames = 0;

  return {
    clip: clip.name,
    duration: clip.duration,
    splats: B.n,
    /** @param {number} nowSec  wall time; the clip loops on its own */
    step(nowSec, speed = 1) {
      if (!t0) t0 = nowSec;
      const t = (nowSec - t0) * speed;
      if (t === last) return;
      last = t;
      const t1 = performance.now();
      poseBones(t);
      skin();
      device.queue.writeBuffer(tr.bufParams, 0, posed.buffer, posed.byteOffset, Math.min(rows, tr.cap) * STRIDE * 4);
      ms = ms ? ms * 0.9 + (performance.now() - t1) * 0.1 : performance.now() - t1;
      // one measurement, once the average has settled: this is a CPU skin, and
      // how much of a frame it eats is the thing to know before growing it
      if (++frames === 90) log(`walk: ${ms.toFixed(1)} ms of CPU per frame for ${B.n.toLocaleString()} splats (${(rows * STRIDE * 4 / 1e6).toFixed(1)} MB written)`);
    },
    /** back to the pose the model was trained in */
    rest() {
      device.queue.writeBuffer(tr.bufParams, 0, base.buffer, base.byteOffset, Math.min(rows, tr.cap) * STRIDE * 4);
      t0 = 0; last = -1;
    },
    get cpuMs() { return ms; },
  };
}
