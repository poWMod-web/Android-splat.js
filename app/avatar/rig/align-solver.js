// Turns dragged marker positions into a fit: uniform scale + root position +
// per-joint rotation offsets (same post-multiplied shape as the viewer's
// fitOffsets()). Pure JS — runs in node for tests and in the browser.
import {
  computeNodeWorldMatrices, quatMul, quatConjugate,
  quatFromShortestArc, quatScaleAngle, quatFromMat4
} from './glb.js';

// markers are keyed by the joint whose position they mark
// Bone groups for influence biasing (helmet-in-hand fixes): factor > 1 extends
// a group's claim radius in binding, < 1 shrinks it. Regexes over joint names.
export const BONE_GROUPS = {
  // legacy combined keys first (older fit JSONs) — sided keys below override
  Hands: /Hand/, Arms: /Shoulder|Arm/, Legs: /Leg/, Feet: /Foot|Toe/,
  Head: /Head|Neck/, Spine: /Spine|Hips/,
  LeftHand: /^LeftHand/, RightHand: /^RightHand/,
  LeftArm: /^Left(Shoulder|Arm|ForeArm)$/, RightArm: /^Right(Shoulder|Arm|ForeArm)$/,
  LeftLeg: /^Left(UpLeg|Leg)$/, RightLeg: /^Right(UpLeg|Leg)$/,
  LeftFoot: /^Left(Foot|Toe)/, RightFoot: /^Right(Foot|Toe)/
};
// { Hands: 2, Legs: 0.5, ... } -> per-joint Float32Array (null when all 1)
export function jointBiasFromGroups(jointNames, groups) {
  if (!groups) return null;
  let any = false;
  const out = new Float32Array(jointNames.length).fill(1);
  for (const [g, rx] of Object.entries(BONE_GROUPS)) {
    const f = groups[g];
    if (!f || f === 1) continue;
    any = true;
    jointNames.forEach((n, j) => { if (rx.test(n)) out[j] = f; });
  }
  return any ? out : null;
}

export const MARKER_JOINTS = [
  'Head', 'Spine2', 'Hips',
  'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftLeg', 'LeftFoot', 'RightLeg', 'RightFoot',
  'LeftToe_End', 'RightToe_End'
];

export const MIRROR_PAIRS = [
  ['LeftArm', 'RightArm'], ['LeftForeArm', 'RightForeArm'], ['LeftHand', 'RightHand'],
  ['LeftLeg', 'RightLeg'], ['LeftFoot', 'RightFoot'], ['LeftToe_End', 'RightToe_End']
];

// [joint to rotate, marker joint it aims at, stretch-bearing children?] —
// parent before child, FK re-evaluated between steps so each aim is exact.
// The optional third entry lists the local translations a stretch solve scales
// for this step (defaults to [marker joint]); the toe tip spans TWO segments
// (ankle->toeBase->tip), so both scale together to keep the aim exact.
const AIM_STEPS = [
  ['LeftShoulder', 'LeftArm'], ['RightShoulder', 'RightArm'],
  ['LeftArm', 'LeftForeArm'], ['RightArm', 'RightForeArm'],
  ['LeftForeArm', 'LeftHand'], ['RightForeArm', 'RightHand'],
  ['LeftUpLeg', 'LeftLeg'], ['RightUpLeg', 'RightLeg'],
  ['LeftLeg', 'LeftFoot'], ['RightLeg', 'RightFoot'],
  ['Neck', 'Head'],
  ['LeftFoot', 'LeftToe_End', ['LeftToeBase', 'LeftToe_End']],
  ['RightFoot', 'RightToe_End', ['RightToeBase', 'RightToe_End']]
];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

function jointPos(avatar, world, name) {
  const m = world[avatar.skinJoints[avatar.jointNames.indexOf(name)]];
  return [m[12], m[13], m[14]];
}
function jointRot(avatar, world, name) {
  return quatFromMat4(world[avatar.skinJoints[avatar.jointNames.indexOf(name)]]);
}

export function solveFit(avatar, markers, opts = {}) {
  const rest = computeNodeWorldMatrices(avatar, {});
  const restHips = jointPos(avatar, rest, 'Hips');
  const restSpan = dist(jointPos(avatar, rest, 'Head'),
    mid(jointPos(avatar, rest, 'LeftFoot'), jointPos(avatar, rest, 'RightFoot')));
  const span = dist(markers.Head, mid(markers.LeftFoot, markers.RightFoot));
  // floor the span at 20% of rest so a marker dragged onto/near another (e.g.
  // Head dragged down to the feet) can't collapse scale toward 0.
  // fixedScale (manual slider override) skips the marker-derived auto scale.
  const scale = opts.fixedScale ?? (Math.max(span, 0.2 * restSpan) / (restSpan || 1));
  const position = sub(markers.Hips,
    [restHips[0] * scale, restHips[1] * scale, restHips[2] * scale]);

  // markers -> avatar rest space
  const ma = {};
  for (const k of Object.keys(markers)) {
    ma[k] = [
      (markers[k][0] - position[0]) / scale,
      (markers[k][1] - position[1]) / scale,
      (markers[k][2] - position[2]) / scale
    ];
  }

  const offsets = {};
  // per-bone stretch (opts.stretch): the child joint's local translation is
  // scaled so each aimed segment also MATCHES the marker distance instead of
  // only pointing at it. Factors accumulate multiplicatively per solve pass
  // and are clamped so a marker dropped onto its parent can't collapse a bone.
  const lengths = opts.stretch ? {} : null;
  const clampLen = (f) => Math.min(2.5, Math.max(0.4, f));
  const fk = () => computeNodeWorldMatrices(avatar, offsets, lengths);
  const applySwing = (world, jName, W) => {
    const Rw = jointRot(avatar, world, jName);
    const off = quatMul(quatMul(quatConjugate(Rw), W), Rw);
    offsets[jName] = offsets[jName] ? quatMul(offsets[jName], off) : off;
  };

  // hip twist first (it rotates the whole body): roll the pelvis about the
  // spine axis so the knee line (LeftLeg->RightLeg) matches the markers'. The
  // spine swing + chest twist below then re-square the upper body to ITS
  // markers, so pelvis and chest twist independently. Mirrored markers are
  // symmetric -> ~zero twist, so the default workflow is unaffected.
  {
    const world = computeNodeWorldMatrices(avatar, offsets, lengths);
    const axis = norm(sub(jointPos(avatar, world, 'Spine2'), jointPos(avatar, world, 'Hips')));
    const flat = (v) => {
      const d = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
      return [v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]];
    };
    const cur = flat(sub(jointPos(avatar, world, 'RightLeg'), jointPos(avatar, world, 'LeftLeg')));
    const tgt = flat(sub(ma.RightLeg, ma.LeftLeg));
    const lc = Math.hypot(cur[0], cur[1], cur[2]), lt = Math.hypot(tgt[0], tgt[1], tgt[2]);
    if (lc > 1e-6 && lt > 1e-6) {
      applySwing(world, 'Hips', quatFromShortestArc(
        [cur[0] / lc, cur[1] / lc, cur[2] / lc],
        [tgt[0] / lt, tgt[1] / lt, tgt[2] / lt]));
    }
  }

  // user hip-rotation offset (opts.hipsTwistDeg, the Bone influence panel's
  // "Hips ↻" slider): EXTRA pelvis roll about the same spine axis on top of
  // the knee-line twist above. The spine/chest stages and the leg AIM_STEPS
  // below re-square the upper body and re-aim knees/feet/toes onto their
  // markers afterwards, so only the pelvis itself ends up turned.
  if (opts.hipsTwistDeg) {
    const world = fk();
    const axis = norm(sub(jointPos(avatar, world, 'Spine2'), jointPos(avatar, world, 'Hips')));
    const a = (opts.hipsTwistDeg * Math.PI) / 360; // half-angle for the quat
    const s = Math.sin(a);
    applySwing(world, 'Hips', [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(a)]);
  }

  // spine first: distribute the hips->chest swing over Spine + Spine1
  {
    const world = fk();
    const pH = jointPos(avatar, world, 'Hips');
    const cur = norm(sub(jointPos(avatar, world, 'Spine2'), pH));
    const tgt = norm(sub(ma.Spine2, pH));
    const half = quatScaleAngle(quatFromShortestArc(cur, tgt), 0.5);
    applySwing(world, 'Spine', half);
    applySwing(fk(), 'Spine1', half);
    if (lengths) {
      // distribute the hips->chest length over the three spine translations
      const cSpan = dist(jointPos(avatar, fk(), 'Spine2'), pH);
      const f = clampLen(dist(ma.Spine2, pH) / (cSpan || 1));
      for (const s of ['Spine', 'Spine1', 'Spine2']) lengths[s] = clampLen((lengths[s] ?? 1) * f);
    }
  }

  // chest twist: roll the upper spine about its own axis so the rig's shoulder
  // line (LeftArm->RightArm) matches the markers'. Swing aims can't express
  // twist, and the clavicle steps below would otherwise contort to fake it.
  // Only expressible with Mirror L/R off — mirrored markers are symmetric and
  // solve to ~zero twist, so this is a no-op for the default workflow.
  {
    const world = fk();
    const axis = norm(sub(jointPos(avatar, world, 'Spine2'), jointPos(avatar, world, 'Spine')));
    const flat = (v) => {
      const d = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
      return [v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]];
    };
    const cur = flat(sub(jointPos(avatar, world, 'RightArm'), jointPos(avatar, world, 'LeftArm')));
    const tgt = flat(sub(ma.RightArm, ma.LeftArm));
    const lc = Math.hypot(cur[0], cur[1], cur[2]), lt = Math.hypot(tgt[0], tgt[1], tgt[2]);
    if (lc > 1e-6 && lt > 1e-6) {
      const half = quatScaleAngle(quatFromShortestArc(
        [cur[0] / lc, cur[1] / lc, cur[2] / lc],
        [tgt[0] / lt, tgt[1] / lt, tgt[2] / lt]), 0.5);
      applySwing(world, 'Spine1', half);
      applySwing(fk(), 'Spine2', half);
    }
  }

  for (const [jName, tName, stretchJoints] of AIM_STEPS) {
    const world = fk();
    const pJ = jointPos(avatar, world, jName);
    const cur = norm(sub(jointPos(avatar, world, tName), pJ));
    const tgt = norm(sub(ma[tName], pJ));
    applySwing(world, jName, quatFromShortestArc(cur, tgt));
    if (lengths) {
      const segLen = dist(jointPos(avatar, world, tName), pJ);
      const f = dist(ma[tName], pJ) / (segLen || 1);
      // scaling every translation in the chain by f scales the whole
      // aim-joint->marker-joint vector by exactly f (rigid chain linearity)
      for (const sj of stretchJoints ?? [tName]) lengths[sj] = clampLen((lengths[sj] ?? 1) * f);
    }
  }

  return lengths ? { scale, position, offsets, lengths } : { scale, position, offsets };
}

// world positions of the marker joints for a given fit — inverse of solveFit
// on reachable poses; used to init UI markers and for round-trip tests
export function markerPositions(avatar, fit) {
  const world = computeNodeWorldMatrices(avatar, fit.offsets || {}, fit.lengths || null);
  const out = {};
  for (const name of MARKER_JOINTS) {
    const p = jointPos(avatar, world, name);
    out[name] = [
      p[0] * fit.scale + fit.position[0],
      p[1] * fit.scale + fit.position[1],
      p[2] * fit.scale + fit.position[2]
    ];
  }
  return out;
}
