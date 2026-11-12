// autofit-core.js — from triangulated body landmarks to a rig fit, pure.
//
// The browser (Splat.js avatar mode) and the CLI (tools/autofit.mjs) share
// this; nothing here touches files. Frames: landmarks and markers arrive in
// the splat's PLY frame (y down); the rig lives in the rigger's frame (y up),
// which is the PLY frame flipped 180° about X — the same flip the runtime
// applies to the splat wrapper.
import { solveFit, markerPositions, MARKER_JOINTS } from './align-solver.js';

/** MediaPipe pose landmark indices the fit uses. */
export const POSE_LM = {
  nose: 0, l_ear: 7, r_ear: 8, l_sho: 11, r_sho: 12, l_elb: 13, r_elb: 14,
  l_wri: 15, r_wri: 16, l_hip: 23, r_hip: 24, l_knee: 25, r_knee: 26,
  l_ank: 27, r_ank: 28, l_heel: 29, r_heel: 30, l_toe: 31, r_toe: 32,
};

const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const lerp = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];

/** Triangulated landmarks ({name: [x,y,z]} in the PLY frame) -> the rigger's
 *  fifteen markers. Torso and head markers sit by proportion, not on a
 *  landmark: MediaPipe has no neck or chest point, and bones with the same
 *  name sit at different heights in the two rigs (an ear-midpoint Head
 *  stretched the neck 1.53x, 2026-09-12). */
export function landmarksToMarkers(pts) {
  const m = {};
  if (pts.l_hip && pts.r_hip && pts.l_sho && pts.r_sho) {
    const hips = mid(pts.l_hip, pts.r_hip), sho = mid(pts.l_sho, pts.r_sho);
    m.Hips = hips;
    m.Spine2 = lerp(hips, sho, 0.72);
    if (pts.l_ear && pts.r_ear) m.Head = lerp(sho, mid(pts.l_ear, pts.r_ear), 0.59);
  }
  const pairs = {
    LeftArm: 'l_sho', LeftForeArm: 'l_elb', LeftHand: 'l_wri', RightArm: 'r_sho', RightForeArm: 'r_elb', RightHand: 'r_wri',
    LeftLeg: 'l_knee', LeftFoot: 'l_ank', LeftToe_End: 'l_toe', RightLeg: 'r_knee', RightFoot: 'r_ank', RightToe_End: 'r_toe',
  };
  for (const [j, lm] of Object.entries(pairs)) if (pts[lm]) m[j] = pts[lm].slice();
  const feet = ['l_heel', 'r_heel', 'l_toe', 'r_toe', 'l_ank', 'r_ank'].filter((k) => pts[k]).map((k) => pts[k][1]);
  return { markers: m, floorY: feet.length ? Math.max(...feet) : null, missing: MARKER_JOINTS.filter((j) => !m[j]) };
}

/** Markers (PLY frame) -> fit JSON (what the rigger opens and export-binding
 *  binds with). opts.asym keeps left/right as measured (body-model markers);
 *  the default mirrors the limbs about the torso line (a swaying capture). */
export function markersToFit(avatar, rawMarkers, opts = {}) {
  const missing = MARKER_JOINTS.filter((j) => !rawMarkers[j]);
  if (missing.length) throw new Error(`markers missing: ${missing.join(' ')}`);
  const flip = ([x, y, z]) => [x, -y, -z];
  let m = Object.fromEntries(Object.entries(rawMarkers).map(([k, v]) => [k, flip(v)]));
  // yaw so the shoulder line lies on -X (the rig's convention)
  const d = [m.RightArm[0] - m.LeftArm[0], m.RightArm[2] - m.LeftArm[2]];
  const yaw = Math.atan2(d[1], d[0]) - Math.PI;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const rotY = ([x, y, z]) => [x * cy + z * sy, y, -x * sy + z * cy];
  m = Object.fromEntries(Object.entries(m).map(([k, v]) => [k, rotY(v)]));
  {
    const r = [m.RightArm[0] - m.LeftArm[0], m.RightArm[2] - m.LeftArm[2]];
    const n = Math.hypot(r[0], r[1]);
    if (r[0] / n > -0.99) throw new Error('yaw did not put the shoulder line on -X');
  }
  if (!opts.asym) {
    const xm = (m.Hips[0] + m.Spine2[0] + m.Head[0]) / 3;
    const mirror = ([x, y, z]) => [2 * xm - x, y, z];
    for (const [L, R] of [['LeftArm', 'RightArm'], ['LeftForeArm', 'RightForeArm'], ['LeftHand', 'RightHand'],
      ['LeftLeg', 'RightLeg'], ['LeftFoot', 'RightFoot'], ['LeftToe_End', 'RightToe_End']]) {
      const mr = mirror(m[R]);
      const l = [(m[L][0] + mr[0]) / 2, (m[L][1] + mr[1]) / 2, (m[L][2] + mr[2]) / 2];
      m[L] = l; m[R] = mirror(l);
    }
    for (const c of ['Hips', 'Spine2', 'Head']) m[c][0] = xm;
  }
  const fit = solveFit(avatar, m, { stretch: true });
  const footYs = ['LeftToe_End', 'RightToe_End', 'LeftFoot', 'RightFoot'].map((j) => m[j][1]);
  const toeRest = 0.036;   // rig Toe_End rest height (metres, at scale 1)
  fit.rootY = Math.min(...footYs) - toeRest * fit.scale;
  const s = fit.scale, p = fit.position;
  fit.markers = Object.fromEntries(Object.entries(m).map(([k, v]) => [k, [(v[0] - p[0]) / s, (v[1] - p[1]) / s, (v[2] - p[2]) / s]]));
  const yawDeg = +(yaw * 180 / Math.PI).toFixed(3);
  if (Math.abs(yawDeg) > 0.01) fit.splatRotation = [0, yawDeg, 0];
  fit.version = fit.lengths ? 2 : 1;
  const landed = markerPositions(avatar, fit);
  const res = {};
  for (const j of MARKER_JOINTS) res[j] = +Math.hypot(landed[j][0] - m[j][0], landed[j][1] - m[j][1], landed[j][2] - m[j][2]).toFixed(4);
  fit.residualsM = res;
  return fit;
}
