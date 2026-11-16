// export-binding-core.js — the SBA1 sidecar, pure.
//
// What tools/export-binding.mjs does minus the file system, so the browser
// (Splat.js avatar mode) can bind a splat to a fit in the tab. Inputs are
// buffers and objects; output is the sidecar bytes plus the numbers the CLI
// prints. Frames: splat centres arrive in the PLY frame (y down); the fit is
// the rigger's (y up, 180° X flip), the optional surface is in the PLY frame.
import {
  parseAvatar, remapFingersToHands, computeNodeWorldMatrices, computeJointMatrices,
  skinVertices, skinNormals, quatFromEulerDeg, quatRotateVec, m4Identity, m4Invert,
} from './glb.js';
import { bindSplats } from './skin-transfer.js';
import { jointBiasFromGroups } from './align-solver.js';

// Bones of the game_engine rig (Anny / MakeHuman / Unreal naming) -> rpm_std.
// Fingers fold into the hand, the clavicle stays a clavicle, the toe ball is
// the toe base. Anything unmapped falls back to Hips.
export const SURFACE_BONE_MAP = {
  Root: 'Hips', pelvis: 'Hips', spine_01: 'Spine', spine_02: 'Spine1', spine_03: 'Spine2',
  neck_01: 'Neck', head: 'Head',
  clavicle_l: 'LeftShoulder', upperarm_l: 'LeftArm', lowerarm_l: 'LeftForeArm', hand_l: 'LeftHand',
  clavicle_r: 'RightShoulder', upperarm_r: 'RightArm', lowerarm_r: 'RightForeArm', hand_r: 'RightHand',
  thigh_l: 'LeftUpLeg', calf_l: 'LeftLeg', foot_l: 'LeftFoot', ball_l: 'LeftToeBase',
  thigh_r: 'RightUpLeg', calf_r: 'RightLeg', foot_r: 'RightFoot', ball_r: 'RightToeBase',
};

/** Splat centres out of a binary little-endian PLY (x, y, z first). */
export function plyCenters(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const headText = new TextDecoder('latin1').decode(u8.subarray(0, Math.min(u8.length, 65536)));
  const at = headText.indexOf('end_header\n');
  if (at < 0) throw new Error('not a PLY');
  const header = headText.slice(0, at + 11);
  const n = parseInt(header.match(/element vertex (\d+)/)[1], 10);
  const props = (header.match(/property float/g) || []).length;
  // a fresh, aligned copy (a Node Buffer's slice is a view at an odd offset)
  const body = new Uint8Array(n * props * 4); body.set(u8.subarray(at + 11, at + 11 + n * props * 4));
  const raw = new Float32Array(body.buffer);
  const centers = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { centers[i * 3] = raw[i * props]; centers[i * 3 + 1] = raw[i * props + 1]; centers[i * 3 + 2] = raw[i * props + 2]; }
  return centers;
}

/**
 * @param {object} o
 * @param {ArrayBuffer} o.avatarGlb   the rpm_std rig (data/glb_avatar.glb)
 * @param {Float32Array} o.centers    splat centres, PLY frame, xyz per splat
 * @param {object} [o.fit]            fit JSON (autofit / the rigger); without it armHang poses the arms
 * @param {number} [o.armHang=100]
 * @param {object} [o.surface]        fitted body surface JSON (anny_fit.py / head_deform.py), PLY frame
 * @param {boolean} [o.smooth=true]
 * @param {boolean} [o.propagate=true]
 * @param {(m: string) => void} [o.log]
 * @returns {{ bytes: Uint8Array, stats: object }}
 */
export function buildBinding(o) {
  const log = o.log || (() => {});
  const avatar = parseAvatar(o.avatarGlb);
  const remapped = remapFingersToHands(avatar);
  const fitJson = o.fit || null;
  let offsets, lengths = null, fitMeta = null;
  if (fitJson) {
    offsets = fitJson.offsets; lengths = fitJson.lengths || null;
    fitMeta = { scale: fitJson.scale, position: fitJson.rootY != null ? [fitJson.position[0], fitJson.rootY, fitJson.position[2]] : fitJson.position };
    if (fitJson.splatRotation) fitMeta.rotation = fitJson.splatRotation;
    if (lengths) fitMeta.lengths = lengths;
  } else {
    const s = (o.armHang ?? 100) / 100;
    offsets = { LeftArm: quatFromEulerDeg(12.5 * s, 50 * s, 5 * s), RightArm: quatFromEulerDeg(12.5 * s, -50 * s, -5 * s) };
  }
  const fitWorld = computeNodeWorldMatrices(avatar, offsets, lengths);
  const jointMats = computeJointMatrices(avatar, fitWorld);
  const skinned = skinVertices(avatar, jointMats);
  const skinnedNrm = skinNormals(avatar, jointMats);

  // PLY frame -> avatar space: flip, optional pre-rotation, minus position, / scale
  const numSplats = o.centers.length / 3;
  const inv = fitMeta ? 1 / fitMeta.scale : 1;
  const p = fitMeta ? fitMeta.position : [0, 0, 0];
  const rr = fitMeta?.rotation;
  const rq = rr ? quatFromEulerDeg(rr[0], rr[1], rr[2]) : null;
  const bx = rq ? quatRotateVec(rq, [1, 0, 0]) : [1, 0, 0];
  const by = rq ? quatRotateVec(rq, [0, 1, 0]) : [0, 1, 0];
  const bz = rq ? quatRotateVec(rq, [0, 0, 1]) : [0, 0, 1];
  const toAvatar = (x, y, z, out, k) => {
    const fx = x, fy = -y, fz = -z;
    out[k] = (bx[0] * fx + by[0] * fy + bz[0] * fz - p[0]) * inv;
    out[k + 1] = (bx[1] * fx + by[1] * fy + bz[1] * fz - p[1]) * inv;
    out[k + 2] = (bx[2] * fx + by[2] * fy + bz[2] * fz - p[2]) * inv;
  };
  const centers = new Float32Array(numSplats * 3);
  for (let i = 0; i < numSplats; i++) toAvatar(o.centers[i * 3], o.centers[i * 3 + 1], o.centers[i * 3 + 2], centers, i * 3);

  let bindVerts = skinned, bindIndices = avatar.indices, bindJoints = avatar.joints, bindWeights = avatar.weights, bindNormals = skinnedNrm;
  const stats = { numSplats, remapped, surface: null };
  if (o.surface) {
    const S = o.surface;
    const nv = S.vertices.length;
    bindVerts = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++) { const v = S.vertices[i]; toAvatar(v[0], v[1], v[2], bindVerts, i * 3); }
    bindIndices = new Uint32Array(S.faces.flat());
    const toRpm = S.boneLabels.map((b) => {
      let name = SURFACE_BONE_MAP[b];
      if (!name) { const m = b.match(/^(index|middle|pinky|ring|thumb)_\d+_([lr])$/); if (m) name = m[2] === 'l' ? 'LeftHand' : 'RightHand'; }
      const j = name ? avatar.jointNames.indexOf(name) : -1;
      return j < 0 ? avatar.jointNames.indexOf('Hips') : j;
    });
    bindJoints = new Uint16Array(nv * 4); bindWeights = new Float32Array(nv * 4);
    let folded = 0;
    for (let i = 0; i < nv; i++) {
      const acc = new Map();
      S.boneIndices[i].forEach((b, k) => { const w = S.boneWeights[i][k]; if (!(w > 0)) return; const j = toRpm[b]; acc.set(j, (acc.get(j) || 0) + w); });
      const top = [...acc.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
      const sum = top.reduce((a, [, w]) => a + w, 0) || 1;
      if (acc.size > 4) folded++;
      top.forEach(([j, w], k) => { bindJoints[i * 4 + k] = j; bindWeights[i * 4 + k] = w / sum; });
    }
    bindNormals = new Float32Array(nv * 3);
    for (let t = 0; t < bindIndices.length; t += 3) {
      const a = bindIndices[t] * 3, b = bindIndices[t + 1] * 3, c = bindIndices[t + 2] * 3;
      const ux = bindVerts[b] - bindVerts[a], uy = bindVerts[b + 1] - bindVerts[a + 1], uz = bindVerts[b + 2] - bindVerts[a + 2];
      const vx = bindVerts[c] - bindVerts[a], vy = bindVerts[c + 1] - bindVerts[a + 1], vz = bindVerts[c + 2] - bindVerts[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const q of [a, b, c]) { bindNormals[q] += nx; bindNormals[q + 1] += ny; bindNormals[q + 2] += nz; }
    }
    for (let i = 0; i < nv; i++) {
      const l = Math.hypot(bindNormals[i * 3], bindNormals[i * 3 + 1], bindNormals[i * 3 + 2]) || 1;
      bindNormals[i * 3] /= l; bindNormals[i * 3 + 1] /= l; bindNormals[i * 3 + 2] /= l;
    }
    stats.surface = { verts: nv, tris: bindIndices.length / 3, folded, source: S.source };
    log(`surface: ${nv} verts, ${bindIndices.length / 3} tris (${S.source}); ${folded} verts had >4 rpm influences`);
  }

  const hips = avatar.jointNames.indexOf('Hips');
  const jointPos = new Float32Array(avatar.jointNames.length * 3);
  for (let j = 0; j < avatar.jointNames.length; j++) {
    const m = fitWorld[avatar.skinJoints[j]];
    jointPos[j * 3] = m[12]; jointPos[j * 3 + 1] = m[13]; jointPos[j * 3 + 2] = m[14];
  }
  const nodeToJoint = new Map(avatar.skinJoints.map((ni, j) => [ni, j]));
  const parentJoint = avatar.skinJoints.map((ni) => {
    let pp = avatar.nodes[ni].parent;
    while (pp !== -1 && !nodeToJoint.has(pp)) pp = avatar.nodes[pp].parent;
    return pp === -1 ? -1 : nodeToJoint.get(pp);
  });
  const t0 = Date.now();
  const binding = bindSplats(centers, bindVerts, bindIndices, bindJoints, bindWeights, {
    farDist: 0.25, farJoint: hips, jointPos, jointParent: parentJoint,
    smoothIters: o.smooth === false ? 0 : 2, smoothRadius: 0.03,
    propagate: o.propagate !== false,
    meshNormals: bindNormals,
    jointBias: fitJson && fitJson.boneBias ? jointBiasFromGroups(avatar.jointNames, fitJson.boneBias) : null,
  });
  let sumD = 0, far = 0, nNear = 0;
  for (let i = 0; i < numSplats; i++) {
    const d = binding.dists[i];
    if (!Number.isFinite(d) || d > 0.25) far++; else { sumD += d; nNear++; }
  }
  stats.meanNearCm = +(sumD / Math.max(1, nNear) * 100).toFixed(2); stats.far = far; stats.bindSec = +((Date.now() - t0) / 1000).toFixed(1);
  log(`bound ${numSplats} splats in ${stats.bindSec}s, mean near dist ${stats.meanNearCm}cm, ${far} far`);

  const numBones = avatar.jointNames.length;
  const invFit = new Float32Array(numBones * 16);
  for (let j = 0; j < numBones; j++) { const m = m4Identity(); m4Invert(m, fitWorld[avatar.skinJoints[j]]); invFit.set(m, j * 16); }
  const idxU8 = new Uint8Array(numSplats * 4), wgtU8 = new Uint8Array(numSplats * 4);
  for (let i = 0; i < numSplats * 4; i++) { idxU8[i] = binding.indices[i]; wgtU8[i] = Math.round(Math.min(1, Math.max(0, binding.weights[i])) * 255); }
  const nrmI8 = binding.normals ? new Int8Array(numSplats * 3) : null;
  if (nrmI8) for (let i = 0; i < numSplats * 3; i++) nrmI8[i] = Math.max(-127, Math.min(127, Math.round(binding.normals[i] * 127)));
  const meta = JSON.stringify({
    version: nrmI8 ? 3 : 2, numSplats, numBones, jointNames: avatar.jointNames, parentJoint,
    armHang: fitJson ? null : (o.armHang ?? 100), fit: fitMeta,
    layout: 'invFit:f32[numBones*16], idx:u8[numSplats*4], wgt:u8[numSplats*4], centers:f32[numSplats*3]' + (nrmI8 ? ', nrm:i8[numSplats*3]' : ''),
    space: 'avatar-root-relative, splat flipped 180x (x,-y,-z); centers in raw ply space',
    surface: o.surface ? 'fitted body model, bones remapped to rpm_std' : 'rig mesh',
  });
  const metaBytes = new TextEncoder().encode(meta);
  const pad = (4 - (metaBytes.length % 4)) % 4;
  const rawCenters = o.centers instanceof Float32Array ? o.centers : Float32Array.from(o.centers);
  const total = 8 + metaBytes.length + pad + invFit.byteLength + idxU8.byteLength + wgtU8.byteLength + rawCenters.byteLength + (nrmI8 ? nrmI8.byteLength : 0);
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out.set([0x53, 0x42, 0x41, 0x31], 0);
  dv.setUint32(4, metaBytes.length + pad, true);
  out.set(metaBytes, 8);
  let off = 8 + metaBytes.length + pad;
  out.set(new Uint8Array(invFit.buffer), off); off += invFit.byteLength;
  out.set(idxU8, off); off += idxU8.byteLength;
  out.set(wgtU8, off); off += wgtU8.byteLength;
  out.set(new Uint8Array(rawCenters.buffer, rawCenters.byteOffset, rawCenters.byteLength), off); off += rawCenters.byteLength;
  if (nrmI8) out.set(new Uint8Array(nrmI8.buffer), off);
  return { bytes: out, stats, fitMeta };
}
