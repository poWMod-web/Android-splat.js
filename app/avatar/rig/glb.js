// Minimal GLB parser for skinned avatars + column-major mat4 helpers.
// Pure JS (no engine deps) so it runs in node tests and the browser.

// ---------- mat4 (column-major, like glTF/WebGL) ----------

export function m4Identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function m4Mul(out, a, b) {
  // out = a * b
  const r = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r_ = 0; r_ < 4; r_++) {
      r[c * 4 + r_] =
        a[0 * 4 + r_] * b[c * 4 + 0] +
        a[1 * 4 + r_] * b[c * 4 + 1] +
        a[2 * 4 + r_] * b[c * 4 + 2] +
        a[3 * 4 + r_] * b[c * 4 + 3];
    }
  }
  out.set(r);
  return out;
}

export function m4FromTRS(out, t, q, s) {
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  out[0] = (1 - (yy + zz)) * s[0]; out[1] = (xy + wz) * s[0]; out[2] = (xz - wy) * s[0]; out[3] = 0;
  out[4] = (xy - wz) * s[1]; out[5] = (1 - (xx + zz)) * s[1]; out[6] = (yz + wx) * s[1]; out[7] = 0;
  out[8] = (xz + wy) * s[2]; out[9] = (yz - wx) * s[2]; out[10] = (1 - (xx + yy)) * s[2]; out[11] = 0;
  out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1;
  return out;
}

export function m4Invert(out, m) {
  // general 4x4 inverse (column-major)
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1.0 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export function m4TransformPoint(out, m, p) {
  const x = p[0], y = p[1], z = p[2];
  out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
  out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
  out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  return out;
}

export function quatMul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}

export function quatFromEulerDeg(x, y, z) {
  // XYZ intrinsic order, degrees, returns (x,y,z,w)
  const d = Math.PI / 360; // half angle in rad per deg
  const cx = Math.cos(x * d), sx = Math.sin(x * d);
  const cy = Math.cos(y * d), sy = Math.sin(y * d);
  const cz = Math.cos(z * d), sz = Math.sin(z * d);
  return [
    sx * cy * cz + cx * sy * sz,
    cx * sy * cz - sx * cy * sz,
    cx * cy * sz + sx * sy * cz,
    cx * cy * cz - sx * sy * sz
  ];
}

// ---- quaternion helpers ([x,y,z,w]) ----
export function quatConjugate(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function quatRotateVec(q, v) {
  const t = quatMul(quatMul(q, [v[0], v[1], v[2], 0]), quatConjugate(q));
  return [t[0], t[1], t[2]];
}

// rotation taking unit vector a onto unit vector b
export function quatFromShortestArc(a, b) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (d > 0.999999) return [0, 0, 0, 1];
  if (d < -0.999999) {
    // 180deg: pick any axis perpendicular to a
    const ax = Math.abs(a[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
    const c = [a[1] * ax[2] - a[2] * ax[1], a[2] * ax[0] - a[0] * ax[2], a[0] * ax[1] - a[1] * ax[0]];
    const l = Math.hypot(c[0], c[1], c[2]);
    return [c[0] / l, c[1] / l, c[2] / l, 0];
  }
  const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const q = [c[0], c[1], c[2], 1 + d];
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

// slerp(identity, q, t) — scales the rotation angle by t
export function quatScaleAngle(q, t) {
  if (q[3] < 0) q = [-q[0], -q[1], -q[2], -q[3]]; // canonicalize: shortest path
  const w = Math.min(1, Math.max(-1, q[3]));
  const ang = 2 * Math.acos(w);
  if (ang < 1e-7) return [0, 0, 0, 1];
  const s = Math.sin(ang / 2);
  const h = (ang * t) / 2;
  const sh = Math.sin(h);
  return [(q[0] / s) * sh, (q[1] / s) * sh, (q[2] / s) * sh, Math.cos(h)];
}

// rotation part of a column-major mat4 (no shear; normalizes away uniform scale)
export function quatFromMat4(m) {
  const t = m[0] + m[5] + m[10];
  let q;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [(m[6] - m[9]) / s, (m[8] - m[2]) / s, (m[1] - m[4]) / s, s / 4];
  } else if (m[0] > m[5] && m[0] > m[10]) {
    const s = Math.sqrt(1 + m[0] - m[5] - m[10]) * 2;
    q = [s / 4, (m[4] + m[1]) / s, (m[8] + m[2]) / s, (m[6] - m[9]) / s];
  } else if (m[5] > m[10]) {
    const s = Math.sqrt(1 + m[5] - m[0] - m[10]) * 2;
    q = [(m[4] + m[1]) / s, s / 4, (m[9] + m[6]) / s, (m[8] - m[2]) / s];
  } else {
    const s = Math.sqrt(1 + m[10] - m[0] - m[5]) * 2;
    q = [(m[8] + m[2]) / s, (m[9] + m[6]) / s, s / 4, (m[1] - m[4]) / s];
  }
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

// ---------- GLB parsing ----------

const COMPONENT_ARRAYS = {
  5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
  5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array
};
const TYPE_SIZE = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

export function parseGlb(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a GLB');
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, 20, jsonLen)));
  let bin = null;
  let off = 20 + jsonLen;
  while (off < dv.getUint32(8, true)) {
    const clen = dv.getUint32(off, true);
    const ctype = dv.getUint32(off + 4, true);
    if (ctype === 0x004E4942) bin = new Uint8Array(arrayBuffer, off + 8, clen);
    off += 8 + clen;
  }
  return { json, bin };
}

export function readAccessor(glb, accessorIndex) {
  // returns { data: TypedArray (tightly packed), count, numComp, normalized }
  const { json, bin } = glb;
  const acc = json.accessors[accessorIndex];
  const bv = json.bufferViews[acc.bufferView];
  const Arr = COMPONENT_ARRAYS[acc.componentType];
  const numComp = TYPE_SIZE[acc.type];
  const compSize = Arr.BYTES_PER_ELEMENT;
  const stride = bv.byteStride || numComp * compSize;
  const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const out = new Arr(acc.count * numComp);
  if (stride === numComp * compSize) {
    const src = new Arr(bin.buffer, bin.byteOffset + base, acc.count * numComp);
    out.set(src);
  } else {
    const dvb = new DataView(bin.buffer, bin.byteOffset);
    for (let i = 0; i < acc.count; i++) {
      for (let c = 0; c < numComp; c++) {
        const byteOff = base + i * stride + c * compSize;
        let v;
        switch (acc.componentType) {
          case 5126: v = dvb.getFloat32(byteOff, true); break;
          case 5123: v = dvb.getUint16(byteOff, true); break;
          case 5125: v = dvb.getUint32(byteOff, true); break;
          case 5121: v = dvb.getUint8(byteOff); break;
          case 5122: v = dvb.getInt16(byteOff, true); break;
          case 5120: v = dvb.getInt8(byteOff); break;
        }
        out[i * numComp + c] = v;
      }
    }
  }
  return { data: out, count: acc.count, numComp, normalized: !!acc.normalized };
}

function weightsToFloat(att) {
  // normalize integer weight formats to float
  if (att.data instanceof Float32Array) return att.data;
  const scale = att.data instanceof Uint8Array ? 1 / 255 : 1 / 65535;
  const out = new Float32Array(att.data.length);
  for (let i = 0; i < att.data.length; i++) out[i] = att.data[i] * scale;
  return out;
}

// Extracts everything needed for skin transfer from a skinned avatar GLB.
// excludeMeshes: mesh names (or regexes) to leave OUT of the binding soup.
// The rig is a Ready Player Me avatar whose clothing meshes sit on top of a full
// body mesh — the trousers (Wolf3D_Outfit_Bottom) duplicate leg surface that
// Wolf3D_Body already provides. Since binding is nearest-triangle, that outer
// shell steals the leg splats and offsets them by the clothing thickness. Passing
// it here removes it from the SURFACE the bind sees, not just from what renders.
export function parseAvatar(arrayBuffer, { excludeMeshes = [] } = {}) {
  const glb = parseGlb(arrayBuffer);
  const { json } = glb;

  // node hierarchy with parent links
  const nodes = json.nodes.map((n, i) => ({
    index: i,
    name: n.name || `node_${i}`,
    translation: n.translation || [0, 0, 0],
    rotation: n.rotation || [0, 0, 0, 1],
    scale: n.scale || [1, 1, 1],
    children: n.children || [],
    parent: -1
  }));
  nodes.forEach(n => n.children.forEach(c => { nodes[c].parent = n.index; }));

  const skin = json.skins[0];
  const ibms = readAccessor(glb, skin.inverseBindMatrices).data; // numJoints * 16
  const jointNames = skin.joints.map(j => nodes[j].name);

  // merge all skinned primitives into one soup, minus any excluded meshes
  const positions = [], normals = [], joints = [], weights = [], indices = [];
  const skippedMeshes = [];
  let vertBase = 0;
  for (const mesh of json.meshes) {
    if (excludeMeshes.some(x => (x instanceof RegExp ? x.test(mesh.name) : x === mesh.name))) {
      skippedMeshes.push(mesh.name);
      continue;
    }
    for (const prim of mesh.primitives) {
      const pos = readAccessor(glb, prim.attributes.POSITION).data;
      const jnt = readAccessor(glb, prim.attributes.JOINTS_0).data;
      const wgt = weightsToFloat(readAccessor(glb, prim.attributes.WEIGHTS_0));
      const idx = readAccessor(glb, prim.indices).data;
      // NORMAL drives splat relighting (see skinNormals). Every mesh in the
      // stock rig has it; a primitive without one still has to contribute a
      // same-length block or the concatenated soup desyncs from positions.
      const nrm = prim.attributes.NORMAL !== undefined
        ? readAccessor(glb, prim.attributes.NORMAL).data
        : new Float32Array(pos.length);
      positions.push(pos); normals.push(nrm); joints.push(jnt); weights.push(wgt);
      const rebased = new Uint32Array(idx.length);
      for (let i = 0; i < idx.length; i++) rebased[i] = idx[i] + vertBase;
      indices.push(rebased);
      vertBase += pos.length / 3;
    }
  }
  const concat = (arrs, Arr) => {
    const total = arrs.reduce((s, a) => s + a.length, 0);
    const out = new Arr(total);
    let o = 0;
    for (const a of arrs) { out.set(a, o); o += a.length; }
    return out;
  };

  if (skippedMeshes.length) {
    console.log(`[glb] binding surface excludes: ${skippedMeshes.join(', ')}`);
  }
  for (const x of excludeMeshes) {
    const name = x instanceof RegExp ? x.source : x;
    if (!skippedMeshes.some(s2 => (x instanceof RegExp ? x.test(s2) : x === s2))) {
      console.warn(`[glb] excludeMeshes entry matched nothing: ${name}`);
    }
  }

  return {
    nodes,
    sceneRoots: json.scenes[json.scene || 0].nodes,
    skinJoints: skin.joints,      // node index per joint
    jointNames,                   // name per joint (same order)
    ibms,                         // Float32Array numJoints*16
    positions: concat(positions, Float32Array),
    normals: concat(normals, Float32Array),
    joints: concat(joints, Uint16Array),
    weights: concat(weights, Float32Array),
    indices: concat(indices, Uint32Array)
  };
}

// Rigid hands: remap every joint below the wrist (fingers) to the hand bone itself.
// Finger joints keep their bind-relative pose in the fit, so this changes nothing for
// the fitted mesh — it only stops splat binding from using noisy finger bones.
export function remapFingersToHands(av) {
  const nodeToJoint = new Map(av.skinJoints.map((ni, j) => [ni, j]));
  const remap = new Uint16Array(av.jointNames.length);
  for (let j = 0; j < av.jointNames.length; j++) {
    remap[j] = j;
    let ni = av.nodes[av.skinJoints[j]].parent;
    while (ni !== -1) {
      const name = av.nodes[ni].name;
      if (name === 'LeftHand' || name === 'RightHand') {
        remap[j] = nodeToJoint.get(ni);
        break;
      }
      ni = av.nodes[ni].parent;
    }
  }
  let changed = 0;
  for (let k = 0; k < av.joints.length; k++) {
    const r = remap[av.joints[k]];
    if (r !== av.joints[k]) { av.joints[k] = r; changed++; }
  }
  return changed;
}

// Compute world (glb-scene-space) matrix per node, applying optional per-bone
// local rotation offsets: poseOffsets = { nodeName: [qx,qy,qz,qw] }.
// lengths = { nodeName: factor } scales that node's local translation (i.e. the
// parent->node bone segment) — per-bone stretch on top of the uniform fit scale.
export function computeNodeWorldMatrices(avatar, poseOffsets = {}, lengths = null) {
  const { nodes, sceneRoots } = avatar;
  const world = nodes.map(() => m4Identity());
  const local = m4Identity();
  const t = [0, 0, 0];
  const visit = (ni, parentWorld) => {
    const n = nodes[ni];
    let rot = n.rotation;
    const off = poseOffsets[n.name];
    if (off) rot = quatMul(n.rotation, off);
    let trans = n.translation;
    const f = lengths?.[n.name];
    if (f) {
      t[0] = trans[0] * f; t[1] = trans[1] * f; t[2] = trans[2] * f;
      trans = t;
    }
    m4FromTRS(local, trans, rot, n.scale);
    m4Mul(world[ni], parentWorld, local);
    for (const c of n.children) visit(c, world[ni]);
  };
  const id = m4Identity();
  for (const r of sceneRoots) visit(r, id);
  return world;
}

// Per-joint skinning matrix in glb scene space: world[joint] * ibm[joint]
export function computeJointMatrices(avatar, nodeWorld) {
  const { skinJoints, ibms } = avatar;
  const out = [];
  for (let j = 0; j < skinJoints.length; j++) {
    const m = m4Identity();
    m4Mul(m, nodeWorld[skinJoints[j]], ibms.subarray(j * 16, j * 16 + 16));
    out.push(m);
  }
  return out;
}

// CPU linear blend skinning of all vertices with given per-joint matrices.
export function skinVertices(avatar, jointMats) {
  const { positions, joints, weights } = avatar;
  const n = positions.length / 3;
  const out = new Float32Array(positions.length);
  const p = [0, 0, 0], tp = [0, 0, 0];
  for (let v = 0; v < n; v++) {
    p[0] = positions[v * 3]; p[1] = positions[v * 3 + 1]; p[2] = positions[v * 3 + 2];
    let ox = 0, oy = 0, oz = 0;
    for (let k = 0; k < 4; k++) {
      const w = weights[v * 4 + k];
      if (w === 0) continue;
      m4TransformPoint(tp, jointMats[joints[v * 4 + k]], p);
      ox += tp[0] * w; oy += tp[1] * w; oz += tp[2] * w;
    }
    out[v * 3] = ox; out[v * 3 + 1] = oy; out[v * 3 + 2] = oz;
  }
  return out;
}

// CPU linear blend skinning of the vertex NORMALS with the same joint matrices.
// Rotation part only (no translation), renormalized — the joint matrices carry at
// most a uniform scale here (the fit's placement), which normalizing divides out,
// so the inverse-transpose a general skin would need is unnecessary.
//
// Used to give every splat a surface normal at bind time (skin-transfer.js), which
// is what makes relighting possible: a gaussian has no normal of its own that is
// reliable enough to light from — see the shortest-axis measurements — but the
// surface it was bound to does.
export function skinNormals(avatar, jointMats) {
  const { normals, joints, weights } = avatar;
  const n = normals.length / 3;
  const out = new Float32Array(normals.length);
  for (let v = 0; v < n; v++) {
    const nx = normals[v * 3], ny = normals[v * 3 + 1], nz = normals[v * 3 + 2];
    let ox = 0, oy = 0, oz = 0;
    for (let k = 0; k < 4; k++) {
      const w = weights[v * 4 + k];
      if (w === 0) continue;
      const m = jointMats[joints[v * 4 + k]];
      ox += w * (m[0] * nx + m[4] * ny + m[8] * nz);
      oy += w * (m[1] * nx + m[5] * ny + m[9] * nz);
      oz += w * (m[2] * nx + m[6] * ny + m[10] * nz);
    }
    const l = Math.hypot(ox, oy, oz) || 1;
    out[v * 3] = ox / l; out[v * 3 + 1] = oy / l; out[v * 3 + 2] = oz / l;
  }
  return out;
}
