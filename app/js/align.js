// Selected-frame pose repair. Pair geometry contributes rotation only (the
// essential matrix has no translation scale); existing neighbouring world
// centres anchor translation. Callers validate and offer candidates before apply.
import { detectAndDescribeMS } from '../../src/sfm/features.js';
import { matchDescriptors } from '../../src/sfm/matching.js';
import {
  ransacE, decomposeE, selectPose, makeRng, m3mul, m3t,
  triangulateN, p3pBearings, refinePose, reprojError,
} from '../../src/sfm/geometry.js';

async function described(url, maxDim = 900) {
  const b = await createImageBitmap(await (await fetch(url)).blob());
  const k = Math.min(1, maxDim / Math.max(b.width, b.height));
  const w = Math.max(2, Math.round(b.width * k)), h = Math.max(2, Math.round(b.height * k));
  const cv = new OffscreenCanvas(w, h), x = cv.getContext('2d', { willReadFrequently: true });
  x.drawImage(b, 0, 0, w, h); b.close?.();
  const d = x.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] * .299 + d[i * 4 + 1] * .587 + d[i * 4 + 2] * .114) / 255;
  return { feat: detectAndDescribeMS(g, w, h, 1800), w, h };
}

function relative(ia, ib, ca, cb, seed) {
  const fa = ia.feat, fb = ib.feat;
  const raw = matchDescriptors(fa.desc, fa.n, fb.desc, fb.n), x1 = [], x2 = [];
  const sa = ia.w / ca.w, sb = ib.w / cb.w;
  for (let k = 0; k < raw.length; k += 2) {
    x1.push([(fa.x[raw[k]] - ca.cx * sa) / (ca.f * sa), (fa.y[raw[k]] - ca.cy * sa) / (ca.f * sa)]);
    x2.push([(fb.x[raw[k + 1]] - cb.cx * sb) / (cb.f * sb), (fb.y[raw[k + 1]] - cb.cy * sb) / (cb.f * sb)]);
  }
  if (x1.length < 30) return null;
  const e = ransacE(x1, x2, 2.5 / ((ca.f * sa + cb.f * sb) / 2), makeRng(seed), 900);
  if (!e || e.inliers.length < 25 || e.inliers.length < x1.length * .3) return null;
  const p = selectPose(decomposeE(e.U, e.V), x1, x2, e.inliers);
  return p && p.goodFrac >= .6
    ? { R: p.R, inliers: e.inliers.length, matches: x1.length, angle: p.medianAngle }
    : null;
}

/** Independent rotation candidates from up to two frames on either side.
 * Features are decoded/described once per image, then reused for all pairs. */
export async function adjacentPoseCandidates(photos, cams, i) {
  const cur = cams[i];
  if (!cur?.f) return [];
  const neighbours = [-2, -1, 1, 2]
    .map((offset) => ({ offset, j: i + offset }))
    .filter(({ j }) => j >= 0 && j < cams.length && cams[j]?.R && photos[j]?.url);
  const indices = [i, ...neighbours.map((n) => n.j)];
  const loaded = new Map(await Promise.all(indices.map(async (j) => [j, await described(photos[j].url)])));
  const out = [];
  for (const { offset, j } of neighbours) {
    const before = offset < 0;
    const r = before
      ? relative(loaded.get(j), loaded.get(i), cams[j], cur, 1701 + i * 11 + j)
      : relative(loaded.get(i), loaded.get(j), cur, cams[j], 2903 + i * 11 + j);
    if (!r) continue;
    out.push({ ...r,
      Rworld: before ? m3mul(r.R, cams[j].R) : m3mul(m3t(r.R), cams[j].R),
      side: before ? 'previous' : 'next', offset, frame: j });
  }
  return out;
}

/** Convert a point stored as fractions of the displayed photograph into the
 * calibrated normalized coordinates used by the COLMAP-style geometry code. */
const normalizedPoint = (cam, p) => [
  (p.x * cam.w - cam.cx) / cam.f,
  (p.y * cam.h - cam.cy) / cam.f,
];

function nearestCloudPoint(cam, xn, cloud, maxPx = 22) {
  if (!cloud || cloud.length < 3) return null;
  let best = null, bestD = Infinity;
  const maxD = maxPx / Math.max(1, cam.f);
  // The reconstruction cloud can be large. Its ordering is already spatially
  // representative, so cap this association pass at roughly 60k samples.
  const step = Math.max(3, Math.ceil(cloud.length / (60000 * 3)) * 3);
  for (let i = 0; i + 2 < cloud.length; i += step) {
    const X = cloud[i], Y = cloud[i + 1], Z = cloud[i + 2];
    const z = cam.R[6] * X + cam.R[7] * Y + cam.R[8] * Z + cam.t[2];
    if (z <= 1e-6) continue;
    const u = (cam.R[0] * X + cam.R[1] * Y + cam.R[2] * Z + cam.t[0]) / z;
    const v = (cam.R[3] * X + cam.R[4] * Y + cam.R[5] * Z + cam.t[1]) / z;
    const d = Math.hypot(u - xn[0], v - xn[1]);
    if (d < bestD) { bestD = d; best = [X, Y, Z]; }
  }
  return bestD <= maxD ? best : null;
}

/** Solve a manually identified image against registered cameras.
 *
 * marks is Map<frameIndex, [{x,y}, ...]>; array positions are the landmark
 * numbers. Known reference poses triangulate each landmark in the existing
 * reconstruction frame. With one reference image, a nearby sparse point can
 * supply the same 3D observation. P3P supplies hypotheses (so five landmarks
 * are sufficient), then all correspondences receive COLMAP-style nonlinear
 * pose refinement.
 */
export function manualPoseFromCorrespondences({ cams, target, marks, cloud = null }) {
  const targetCam = cams[target], targetMarks = marks.get(target) || [];
  if (!targetCam?.f || targetMarks.length < 5) throw new Error('Place at least 5 numbered points on the image being aligned.');
  const refs = [...marks.entries()].filter(([i, pts]) => i !== target && pts?.length);
  if (!refs.length) throw new Error('Place the same numbered points on at least one registered reference image.');
  for (const [i, pts] of refs) {
    if (!cams[i]?.R || !cams[i]?.f) throw new Error(`Frame ${i + 1} has no registered camera pose.`);
    if (pts.length < 5) throw new Error(`Frame ${i + 1} needs at least 5 numbered points.`);
  }

  const obj = [], img = [], labels = [];
  for (let n = 0; n < targetMarks.length; n++) {
    const tp = targetMarks[n];
    if (!tp) continue;
    const knownCams = [], knownObs = [];
    for (const [i, pts] of refs) {
      if (!pts[n]) continue;
      knownCams.push(cams[i]); knownObs.push(normalizedPoint(cams[i], pts[n]));
    }
    let X = knownCams.length >= 2 ? triangulateN(knownCams, knownObs) : null;
    if (!X && knownCams.length === 1) X = nearestCloudPoint(knownCams[0], knownObs[0], cloud);
    if (!X || !X.every(Number.isFinite)) continue;
    obj.push(X); img.push(normalizedPoint(targetCam, tp)); labels.push(n + 1);
  }
  if (obj.length < 5) {
    throw new Error('Only ' + obj.length + ' landmarks could be tied to the reconstruction. Add a second reference image or choose points visible in the point cloud.');
  }

  const bearings = img.map(([x, y]) => {
    const z = 1 / Math.hypot(x, y, 1); return [x * z, y * z, z];
  });
  let best = null;
  // Try every landmark triple (bounded for pathological hand-entered sets),
  // then score against all points to disambiguate P3P's multiple roots.
  let tried = 0;
  for (let a = 0; a < obj.length - 2 && tried < 160; a++) {
    for (let b = a + 1; b < obj.length - 1 && tried < 160; b++) {
      for (let c = b + 1; c < obj.length && tried++ < 160; c++) {
        for (const pose of p3pBearings([obj[a], obj[b], obj[c]], [bearings[a], bearings[b], bearings[c]])) {
          const t = [
            -(pose.R[0] * pose.C[0] + pose.R[1] * pose.C[1] + pose.R[2] * pose.C[2]),
            -(pose.R[3] * pose.C[0] + pose.R[4] * pose.C[1] + pose.R[5] * pose.C[2]),
            -(pose.R[6] * pose.C[0] + pose.R[7] * pose.C[1] + pose.R[8] * pose.C[2]),
          ];
          const errors = obj.map((X, k) => reprojError(pose.R, t, X, img[k]));
          const sorted = [...errors].sort((x, y) => x - y);
          const score = sorted.slice(0, Math.max(5, Math.ceil(sorted.length * .8))).reduce((s, x) => s + x * x, 0);
          if (!best || score < best.score) best = { R: pose.R, t, score };
        }
      }
    }
  }
  if (!best) throw new Error('The selected landmarks are geometrically degenerate. Spread them across the image and include depth variation.');
  const refined = refinePose(best.R, best.t, obj, img, 18);
  const errorsPx = obj.map((X, k) => reprojError(refined.R, refined.t, X, img[k]) * targetCam.f);
  const inliers = errorsPx.filter((e) => e <= 8).length;
  const rmsPx = Math.sqrt(errorsPx.reduce((s, e) => s + e * e, 0) / errorsPx.length);
  if (inliers < 5 || !Number.isFinite(rmsPx) || rmsPx > 18) {
    throw new Error(`Manual landmarks disagree after pose refinement (${inliers}/${obj.length} within 8 px, ${rmsPx.toFixed(1)} px RMS).`);
  }
  return { R: refined.R, t: refined.t, inliers, rmsPx, labels };
}
