// anim.js — the walk cycle, on the finished avatar, in the tab.
//
// The rigger's runtime skins the splat on the GPU inside PlayCanvas
// (client_git/splat-rigger/plugin/splat-avatar.mjs). Splat.js has no engine to
// hang that on, so the same maths runs here on the CPU: sample the clip, pose
// the rig, blend each splat by its binding, and write the posed centres and
// rotations straight into the trainer's parameter buffer between frames.
//
// The clip is `rig/walking_anim.glb`, the same file the rigger plays (synced by
// scripts/sync_rig.mjs). ROTATION TRACKS ONLY: a walk clip's hips translation
// carries the figure forward and out of frame, and what is wanted here is a
// walk on the spot to judge the avatar by.
import {
  parseGlb, readAccessor, quatMul, quatConjugate,
  computeNodeWorldMatrices, computeJointMatrices,
  m4Identity, m4Mul, m4Invert, m4FromTRS,
} from './rig/glb.js';

export const CLIPS = {
  walk: new URL('./rig/walking_anim.glb', import.meta.url).href,
};

/** Load a glTF clip as rotation tracks keyed by node NAME — the clip's node
 *  indices are its own; a rig is matched to it by name, never by index. */
export async function loadClip(url = CLIPS.walk) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`clip ${res.status}`);
  const glb = parseGlb(await res.arrayBuffer());
  const anim = glb.json.animations && glb.json.animations[0];
  if (!anim) throw new Error('no animation in the clip');
  const tracks = [], moves = [];
  let duration = 0;
  for (const ch of anim.channels) {
    if (!ch.target || ch.target.node == null) continue;
    if (ch.target.path !== 'rotation' && ch.target.path !== 'translation') continue;
    const node = glb.json.nodes[ch.target.node];
    if (!node || !node.name) continue;
    const s = anim.samplers[ch.sampler];
    const t = readAccessor(glb, s.input).data;
    const v = readAccessor(glb, s.output).data;
    const dim = ch.target.path === 'rotation' ? 4 : 3;
    if (!t.length || v.length < t.length * dim) continue;
    duration = Math.max(duration, t[t.length - 1]);
    (dim === 4 ? tracks : moves).push({ name: node.name, t, q: v, step: s.interpolation === 'STEP' });
  }
  if (!tracks.length) throw new Error('the clip has no rotation tracks');
  return { name: anim.name || 'clip', tracks, moves, duration };
}

function slerp(out, q, i, j, u) {
  let ax = q[i * 4], ay = q[i * 4 + 1], az = q[i * 4 + 2], aw = q[i * 4 + 3];
  const bx = q[j * 4], by = q[j * 4 + 1], bz = q[j * 4 + 2], bw = q[j * 4 + 3];
  let d = ax * bx + ay * by + az * bz + aw * bw;
  if (d < 0) { ax = -ax; ay = -ay; az = -az; aw = -aw; d = -d; }   // the short way round
  let s0 = 1 - u, s1 = u;
  if (d < 0.9995) {
    const th = Math.acos(Math.min(1, d)), sn = Math.sin(th);
    s0 = Math.sin((1 - u) * th) / sn; s1 = Math.sin(u * th) / sn;
  }
  out[0] = ax * s0 + bx * s1; out[1] = ay * s0 + by * s1;
  out[2] = az * s0 + bz * s1; out[3] = aw * s0 + bw * s1;
  const L = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
  out[0] /= L; out[1] /= L; out[2] /= L; out[3] /= L;
  return out;
}

/** The clip's absolute local rotations at `time` (seconds, wrapped). */
export function sampleClip(clip, time) {
  const out = {};
  const T = clip.duration > 0 ? ((time % clip.duration) + clip.duration) % clip.duration : 0;
  const q = [0, 0, 0, 1];
  for (const tr of clip.tracks) {
    const t = tr.t;
    let i = 0, hi = t.length - 1;
    while (i < hi) { const m = (i + hi + 1) >> 1; if (t[m] <= T) i = m; else hi = m - 1; }
    const j = Math.min(i + 1, t.length - 1);
    const span = t[j] - t[i];
    const u = tr.step || span <= 0 ? 0 : (T - t[i]) / span;
    out[tr.name] = slerp(q.slice(), tr.q, i, j, u);
  }
  return out;
}

/** The clip's local translations at `time` (only the tracks it has — a walk
 *  clip has one, on the hips). */
export function sampleMoves(clip, time) {
  const out = {};
  if (!clip.moves || !clip.moves.length) return out;
  const T = clip.duration > 0 ? ((time % clip.duration) + clip.duration) % clip.duration : 0;
  for (const tr of clip.moves) {
    const t = tr.t;
    let i = 0, hi = t.length - 1;
    while (i < hi) { const m = (i + hi + 1) >> 1; if (t[m] <= T) i = m; else hi = m - 1; }
    const j = Math.min(i + 1, t.length - 1);
    const span = t[j] - t[i];
    const u = tr.step || span <= 0 ? 0 : (T - t[i]) / span;
    out[tr.name] = [0, 1, 2].map((k) => tr.q[i * 3 + k] + (tr.q[j * 3 + k] - tr.q[i * 3 + k]) * u);
  }
  return out;
}

/** The clip as `poseOffsets` for computeNodeWorldMatrices: it multiplies the
 *  offset ONTO the node's rest rotation (quatMul(rest, off)), so an absolute
 *  local rotation from the clip becomes rest⁻¹ · clip. */
export function poseOffsets(avatar, clip, time) {
  const abs = sampleClip(clip, time);
  const out = {};
  for (const n of avatar.nodes) {
    const c = abs[n.name];
    if (c) out[n.name] = quatMul(quatConjugate(n.rotation), c);
  }
  return out;
}

/** Forward kinematics with TRANSLATION overrides, which glb.js's own
 *  computeNodeWorldMatrices does not take (it poses rotations and scales bone
 *  lengths). A clip's root translation has to go in as it was authored, or the
 *  figure ends up a metre from where its bones think it is.
 *  @param {object} o
 *  @param {Object<string,number[]>} o.rot     absolute LOCAL rotation per node name
 *  @param {Object<string,number[]>} o.trans   absolute LOCAL translation per node name
 *  @param {Object<string,number>}   o.lengths bone-length factor per node name (the fit's)
 */
export function poseWorld(avatar, { rot = {}, trans = {}, lengths = null } = {}) {
  const { nodes, sceneRoots } = avatar;
  const world = nodes.map(() => m4Identity());
  const local = m4Identity();
  const visit = (ni, parentWorld) => {
    const n = nodes[ni];
    let t = trans[n.name] || n.translation;
    const f = lengths && lengths[n.name];
    if (f && !trans[n.name]) t = [t[0] * f, t[1] * f, t[2] * f];
    m4FromTRS(local, t, rot[n.name] || n.rotation, n.scale);
    m4Mul(world[ni], parentWorld, local);
    for (const c of n.children) visit(c, world[ni]);
  };
  const id = m4Identity();
  for (const r of sceneRoots) visit(r, id);
  return world;
}

/** Per-joint skinning matrices for a pose, in the rig's own space. */
export function jointMatricesAt(avatar, clip, time) {
  return computeJointMatrices(avatar, computeNodeWorldMatrices(avatar, clip ? poseOffsets(avatar, clip, time) : {}));
}

/** The rig's joint matrices expressed in SPLAT space: fit⁻¹ · J · fit, so a
 *  splat centre can be skinned where it already lives, with no per-splat trip
 *  through the fit. `fit` is the rig→splat matrix (bodyfit's scale/rotation/
 *  translation), `jointMats` the matrices from jointMatricesAt. */
export function jointMatricesInSplatSpace(jointMats, fit) {
  if (!fit) return jointMats;
  const inv = m4Invert(m4Identity(), fit);
  const tmp = m4Identity();
  return jointMats.map((J) => {
    const m = m4Identity();
    m4Mul(tmp, J, inv);     // rig-space J applied to a point already un-fitted
    m4Mul(m, fit, tmp);     // and back into splat space
    return m;
  });
}
