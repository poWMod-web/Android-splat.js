// introcam.js — the space's intro flight, built from the solved camera path.
//
// A cutscene on arrival.space IS a Sequence serialised into a `.path` file and
// hung on the space as an entity (glbUrl = the .path, autoPlay on); the camera
// track lives under the well-known entity id `main-camera`. The benchmark
// scenes have carried such an intro since August — made by hand, once. This
// builds the same artefact from the run's own cameras, on every upload.
//
// Both conventions below were MEASURED against that working intro
// (ugc.arrival.space/splatjs/models/lab360_intro.path, fitted 2026-09-16)
// rather than assumed:
//   position  world = Rz(180) · C. A .sog user model is placed at the origin
//             with rotation z=180 — createUserModel's default for the
//             extension — so the capture frame turns with it. Fitted residual
//             0.13 in a scene of radius 12.9; the next-best candidate
//             transform was 1.12 out, so the flip is not a coincidence.
//   rotation  q = quat(Rz(180) · Rᵀ · diag(1, -1, -1)): the solver's camera
//             looks down +Z with +Y down, PlayCanvas's down -Z with +Y up.
//             Fitted median error 0.20° over the 57 authored keyframes.

import { api, uploadFile } from './arrival.js';
import { camCentre, quatFromR, quatToR, qslerp } from './viewport.js';

const FPS = 24;
// Key budget. The player interpolates between keys and does its own smoothing,
// so density past a point buys nothing and costs it work — 379 keys on Truck
// made the cutscene struggle. The hand-made intro that reads well carries 57
// keys over 34 s, so this aims at that cadence and spends the budget where the
// path turns rather than spreading it evenly.
const TURN_PER_KEY = 20;   // a key per ~20° of turning, on top of the base rate
const KEYS_PER_SEC = 1.5;
const MIN_KEYS = 16;
const MAX_KEYS = 90;
// Two keys on the same frame are not interpolated — the flight jumps there.
// Keys are placed by TURNING while their clock follows DISTANCE, so a tight
// corner advances the arc by almost nothing and rounds several keys onto one
// frame: 79 of Truck's 379 keys collided that way. Every key keeps this gap.
const MIN_FRAME_GAP = 3;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** The nodes the intro flies: co-located rig poses collapsed (a panorama
 *  contributes six orientations at one spot), then the longest unbroken
 *  stretch of the capture — the same rules the viewer's tour follows, so the
 *  space opens with the flight the creator already watched. */
function pathNodes(cams) {
  const reg = (cams || []).filter((c) => c && c.R && c.t);
  if (reg.length < 2) return [];
  const centres = reg.map(camCentre);
  // the median STEP of the capture, and a rig's six views of one spot are not
  // steps: counting those zeros drags the median to nothing and every real
  // stride then reads as a break (112 panoramas fell to a 5-node path)
  const gaps = [];
  for (let i = 1; i < centres.length; i++) {
    const d = dist(centres[i], centres[i - 1]);
    if (d > 1e-9) gaps.push(d);
  }
  gaps.sort((a, b) => a - b);
  const med = gaps[gaps.length >> 1] || 1e-3;

  const keep = [];
  for (let i = 0; i < reg.length; i++) {
    const last = keep[keep.length - 1];
    if (!last || dist(centres[i], camCentre(last)) > 0.15 * med) keep.push(reg[i]);
  }
  if (keep.length < 2) return [];

  const segs = [[keep[0]]];
  for (let i = 1; i < keep.length; i++) {
    if (dist(camCentre(keep[i]), camCentre(keep[i - 1])) > 4 * med) segs.push([]);
    segs[segs.length - 1].push(keep[i]);
  }
  const best = segs.reduce((a, b) => (b.length > a.length ? b : a), segs[0]);
  return best.length < 2 ? [] : best;   // every pose: the spline is cheap, detail is not
}

/** Rotation matrix (row-major 9) -> quaternion {x,y,z,w}. */
function quatOf(m) {
  const tr = m[0] + m[4] + m[8];
  let s;
  if (tr > 0) {
    s = Math.sqrt(tr + 1) * 2;
    return { x: (m[7] - m[5]) / s, y: (m[2] - m[6]) / s, z: (m[3] - m[1]) / s, w: 0.25 * s };
  }
  if (m[0] > m[4] && m[0] > m[8]) {
    s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    return { x: 0.25 * s, y: (m[1] + m[3]) / s, z: (m[2] + m[6]) / s, w: (m[7] - m[5]) / s };
  }
  if (m[4] > m[8]) {
    s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    return { x: (m[1] + m[3]) / s, y: 0.25 * s, z: (m[5] + m[7]) / s, w: (m[2] - m[6]) / s };
  }
  s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
  return { x: (m[2] + m[6]) / s, y: (m[5] + m[7]) / s, z: 0.25 * s, w: (m[3] - m[1]) / s };
}

// What a .sog user model gets from createUserModel: the origin, turned half a
// turn about Z, unscaled. A space whose model was placed or levelled by hand
// carries something else, and the flight has to follow it exactly — the camera
// belongs to the splat, not to the room.
const DEFAULT_MODEL = { position: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 180 }, scale: 1 };

/** PlayCanvas's own euler->quaternion (Quat.setFromEulerAngles), so a model
 *  placed in the editor and this flight agree on what its angles mean. */
function eulerR({ x, y, z }) {
  const h = 0.5 * Math.PI / 180;
  const sx = Math.sin(x * h), cx = Math.cos(x * h);
  const sy = Math.sin(y * h), cy = Math.cos(y * h);
  const sz = Math.sin(z * h), cz = Math.cos(z * h);
  return quatToR([
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ]);
}

const mul3 = (A, B) => {
  const m = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    m[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  }
  return m;
};

/** A point on the flight, in the SPACE's world:
 *  world = P + R(model) · (scale · X), and the camera's basis turned the same
 *  way (the solver looks down +Z with +Y down, PlayCanvas down -Z with +Y up). */
function worldPose({ centre, q }, model) {
  const Rm = eulerR(model.rotation);
  const sc = model.scale ?? 1;
  const c = [centre[0] * sc, centre[1] * sc, centre[2] * sc];
  const position = {
    x: model.position.x + Rm[0] * c[0] + Rm[1] * c[1] + Rm[2] * c[2],
    y: model.position.y + Rm[3] * c[0] + Rm[4] * c[1] + Rm[5] * c[2],
    z: model.position.z + Rm[6] * c[0] + Rm[7] * c[1] + Rm[8] * c[2],
  };
  const R = quatToR(q);
  const rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  const flip = [1, 0, 0, 0, -1, 0, 0, 0, -1];
  return { position, rotation: quatOf(mul3(Rm, mul3(rt, flip))) };
}

const track = (type, keyframes) => ({ type, keyframes });

/** The flight itself, smoothed exactly the way the viewer's tour smooths it —
 *  a handheld capture is jittery and a spline through jitter is jittery too —
 *  then sampled at EQUAL ARC LENGTH so the platform's own interpolation runs
 *  at constant speed between the keys. Exporting the raw poses (as this did
 *  first) hands the player a polyline of camera shake. */
function flight(nodes, keys) {
  const n = nodes.length;
  // positions: triangular smoothing over ±3 neighbours
  const raw = nodes.map(camCentre);
  const pts = raw.map((_, i) => {
    const acc = [0, 0, 0];
    let w = 0;
    for (let k = -3; k <= 3; k++) {
      const j = clamp(i + k, 0, n - 1);
      const wt = 4 - Math.abs(k);
      for (let c = 0; c < 3; c++) acc[c] += raw[j][c] * wt;
      w += wt;
    }
    return [acc[0] / w, acc[1] / w, acc[2] / w];
  });
  // rotations: sign-aligned, then two passes towards the neighbours' midpoint
  const qs = nodes.map((c) => quatFromR(c.R));
  for (let i = 1; i < n; i++) {
    if (qs[i - 1][0] * qs[i][0] + qs[i - 1][1] * qs[i][1] + qs[i - 1][2] * qs[i][2] + qs[i - 1][3] * qs[i][3] < 0) {
      qs[i] = qs[i].map((v) => -v);
    }
  }
  let sq = qs;
  for (let pass = 0; pass < 2; pass++) {
    sq = sq.map((q, i) => (i === 0 || i === sq.length - 1) ? q : qslerp(q, qslerp(sq[i - 1], sq[i + 1], 0.5), 0.5));
  }
  // Catmull-Rom through the smoothed points, resampled into an arc-length table
  const P = (k) => pts[clamp(k, 0, n - 1)];
  const cr = (i, f) => {
    const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
    const t2 = f * f, t3 = t2 * f;
    return [0, 1, 2].map((k) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * f +
      (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
      (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3));
  };
  const samples = [], us = [], cum = [0];
  const SUB = 8;
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < SUB; k++) { samples.push(cr(i, k / SUB)); us.push(i + k / SUB); }
  }
  samples.push(cr(n - 2, 1)); us.push(n - 1);
  for (let k = 1; k < samples.length; k++) {
    const a = samples[k - 1], b = samples[k];
    cum.push(cum[k - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  const total = cum[cum.length - 1] || 1e-6;

  // how much the curve TURNS, per step and in total: a walk that doubles back
  // spends its corner in a few centimetres, and keys have to be there for it
  const turnAt = [0];
  for (let i = 1; i < samples.length - 1; i++) {
    const a = [0, 1, 2].map((c) => samples[i][c] - samples[i - 1][c]);
    const b = [0, 1, 2].map((c) => samples[i + 1][c] - samples[i][c]);
    const la = Math.hypot(...a), lb = Math.hypot(...b);
    const dot = (la < 1e-9 || lb < 1e-9) ? 1 : (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
    turnAt.push(Math.acos(clamp(dot, -1, 1)) * 180 / Math.PI);
  }
  turnAt.push(0);
  const totalTurn = turnAt.reduce((s, v) => s + v, 0);
  if (keys === 0) return { total, totalTurn };

  // keys are spent where they buy something: a straight metre needs one, a
  // reversal needs several. Cost = distance (in base-key units) + turning (in
  // TURN_PER_KEY units); the keys sit at equal COST, their frame numbers at
  // equal ARC LENGTH, so the corner is rounded and the speed stays constant.
  const baseKeys = Math.max(2, keys / 2);   // half the budget to distance, the rest to turning
  const perKey = total / baseKeys;
  const cost = [0];
  for (let k = 1; k < samples.length; k++) {
    cost.push(cost[k - 1] + (cum[k] - cum[k - 1]) / perKey + (turnAt[k] || 0) / TURN_PER_KEY);
  }
  const totalCost = cost[cost.length - 1] || 1e-9;

  const out = [];
  for (let key = 0; key < keys; key++) {
    const want = totalCost * (key / (keys - 1));
    let k = 1;
    while (k < cost.length - 1 && cost[k] < want) k++;
    const span = cost[k] - cost[k - 1] || 1e-9;
    const f = clamp((want - cost[k - 1]) / span, 0, 1);
    const a = samples[k - 1], b = samples[k];
    const centre = [0, 1, 2].map((c) => a[c] + (b[c] - a[c]) * f);
    const u = us[k - 1] + (us[k] - us[k - 1]) * f;
    const i0 = clamp(Math.floor(u), 0, n - 1), i1 = clamp(i0 + 1, 0, n - 1);
    // where this key sits along the path, 0..1 — the clock follows the metres
    const at = (cum[k - 1] + (cum[k] - cum[k - 1]) * f) / total;
    out.push({ centre, q: qslerp(sq[i0], sq[i1], u - i0), at });
  }
  return out;
}

/** The Sequence an intro cutscene is, or null when the capture has no path to
 *  fly (fewer than two distinct camera positions). */
export function buildIntroSequence(cams, { model = DEFAULT_MODEL } = {}) {
  const nodes = pathNodes(cams);
  if (!nodes.length) return null;
  // the viewer's own pacing: one pass, never longer than half a minute
  const seconds = clamp(1.4 * nodes.length, 10, 30);
  const { totalTurn } = flight(nodes, 0);
  const span = Math.round(seconds * FPS);
  const keys = clamp(Math.ceil(seconds * KEYS_PER_SEC + totalTurn / TURN_PER_KEY),
    MIN_KEYS, Math.min(MAX_KEYS, Math.floor(span / MIN_FRAME_GAP) + 1));
  const path = flight(nodes, keys);
  const pos = [], rot = [], scl = [];
  let last = -MIN_FRAME_GAP;
  path.forEach((step, i) => {
    const { position, rotation } = worldPose(step, model);
    // the clock follows the metres, but never closer than the player can read;
    // a corner simply takes a little longer, which is how a corner should feel
    const frameNumber = Math.max(last + MIN_FRAME_GAP, Math.round(step.at * span));
    last = frameNumber;
    pos.push({ id: `keyframe-p${i}`, frameNumber, keyframeData: position });
    rot.push({ id: `keyframe-r${i}`, frameNumber, keyframeData: rotation });
    scl.push({ id: `keyframe-s${i}`, frameNumber, keyframeData: { x: 1, y: 1, z: 1 } });
  });
  return {
    id: 'splatjsintro',
    data: {
      fps: FPS,
      entities: {
        'main-camera': {
          position: track('vec3', pos),
          rotation: track('quat', rot),
          scale: track('vec3', scl),
          fov: track('number', []),
          brightness: track('number', []),
          contrast: track('number', []),
          saturation: track('number', []),
        },
      },
      markers: [],
    },
  };
}

/** Give a freshly created space its intro flight. Best-effort by contract: the
 *  caller's upload is finished and correct without it, so every failure here is
 *  reported and swallowed. Returns { entityId, keys, seconds } or null. */
export async function attachIntroCam(spaceId, cams, { token, slug = 'splat', onStatus = () => {} } = {}) {
  const seq = buildIntroSequence(cams);
  if (!seq) return null;
  const keys = seq.data.entities['main-camera'].position.keyframes.length;
  onStatus('Building the intro flight …');
  const blob = new Blob([JSON.stringify(seq)], { type: 'application/json' });
  const { resourceKey } = await uploadFile(blob, `${slug}_intro.path`, {
    token, contentType: 'application/json', onStatus,
  });
  // the id the platform's own migration uses, so a space never grows two
  const entityId = `user-model-introcam-${spaceId}`;
  await api(`/spaces/${encodeURIComponent(spaceId)}/entities`, token, {
    resource_key: resourceKey,
    entity_id: entityId,
    entity_data: { displayName: 'Intro flight', autoPlay: true, loop: false },
  });
  return { entityId, keys, seconds: +((keys - 1) * KEY_EVERY / FPS).toFixed(1) };
}
