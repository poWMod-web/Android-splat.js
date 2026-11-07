// landmarks.js — where her joints and her face are, from the capture itself.
//
// MediaPipe (tasks-vision, in the tab) finds 33 body points on every
// registered frame and 478 face points in a head window of it; each point is
// triangulated across the solved cameras (robust DLT), which gives
//   - the rigger's fifteen markers -> a rig fit (autofit-core, rig mesh path)
//   - one canonical 3D face, and per frame a head-stabilised camera solved
//     by PnP against it (the head moves against the body; the crop camera
//     absorbs that) plus the native-resolution window to cut for the face
//     pass. Same maths as tests/bench/face_crops.py, no Python.
// Frames: the session's reconstruction IS the splat's PLY frame.
import { landmarksToMarkers, markersToFit, POSE_LM } from '../rig/autofit-core.js';
import { parseAvatar } from '../rig/glb.js';

export const id = 'landmarks';
export const needs = ['train'];

const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const MODELS = new URL('../../models/', import.meta.url).href;
const RIG_GLB = new URL('../rig/glb_avatar.glb', import.meta.url).href;
const SIDE = 768;            // face crop window, native px
const MIN_VIEWS = 6;

// ── projective helpers (feature-scale cameras) ──────────────────────────────
export const P = (c) => {
  const R = c.R, t = c.t, f = c.f, fy = c.fy ?? c.f, cx = c.cx, cy = c.cy;
  return [
    f * R[0] + cx * R[6], f * R[1] + cx * R[7], f * R[2] + cx * R[8], f * t[0] + cx * t[2],
    fy * R[3] + cy * R[6], fy * R[4] + cy * R[7], fy * R[5] + cy * R[8], fy * t[1] + cy * t[2],
    R[6], R[7], R[8], t[2],
  ];
};
export function dlt(rows) {   // rows: [{P, u, v}] -> X = the null vector of the 2n x 4 system (smallest eigenvector of AᵀA)
  const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (const { P: p, u, v } of rows) {
    const r1 = [u * p[8] - p[0], u * p[9] - p[1], u * p[10] - p[2], u * p[11] - p[3]];
    const r2 = [v * p[8] - p[4], v * p[9] - p[5], v * p[10] - p[6], v * p[11] - p[7]];
    for (const r of [r1, r2]) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) M[i][j] += r[i] * r[j];
  }
  // column scaling for conditioning (the w column carries f·t ≈ hundreds of px)
  const sc = [0, 1, 2, 3].map((j) => Math.sqrt(M[j][j]) || 1);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) M[i][j] /= sc[i] * sc[j];
  const x = smallestEigvec4(M).map((v, j) => v / sc[j]);
  return [x[0] / x[3], x[1] / x[3], x[2] / x[3]];
}
/** Smallest eigenvector of a symmetric 4x4 (cyclic Jacobi rotations). */
function smallestEigvec4(M) {
  const A = M.map((r) => r.slice()); const V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += A[i][j] * A[i][j];
    if (off < 1e-24) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 4; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 4; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  let m = 0; for (let i = 1; i < 4; i++) if (A[i][i] < A[m][m]) m = i;
  return [V[0][m], V[1][m], V[2][m], V[3][m]];
}
const reproj = (X, p) => {
  const z = p[8] * X[0] + p[9] * X[1] + p[10] * X[2] + p[11];
  if (z <= 1e-9) return null;
  return [(p[0] * X[0] + p[1] * X[1] + p[2] * X[2] + p[3]) / z, (p[4] * X[0] + p[5] * X[1] + p[6] * X[2] + p[7]) / z];
};
export function triangulate(rows, minPx = 3) {
  let cur = rows.slice();
  if (cur.length < MIN_VIEWS) return null;
  let X = dlt(cur);
  for (let it = 0; it < 6; it++) {
    const e = cur.map((r) => { const q = reproj(X, r.P); return q ? Math.hypot(q[0] - r.u, q[1] - r.v) : 1e9; });
    const med = [...e].sort((a, b) => a - b)[e.length >> 1];
    const cut = Math.max(minPx, 1.5 * med);
    const keep = cur.filter((_, i) => e[i] <= cut);
    if (keep.length < MIN_VIEWS || keep.length === cur.length) break;
    cur = keep; X = dlt(cur);
  }
  const e = cur.map((r) => { const q = reproj(X, r.P); return q ? Math.hypot(q[0] - r.u, q[1] - r.v) : 1e9; });
  return { X, views: cur.length, err: [...e].sort((a, b) => a - b)[e.length >> 1] };
}

// ── PnP: refine a camera pose (R,t) against 3D-2D pairs, Huber-robust LM ────
function rodrigues(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const k = [r[0] / th, r[1] / th, r[2] / th], c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [c + k[0] * k[0] * C, k[0] * k[1] * C - k[2] * s, k[0] * k[2] * C + k[1] * s,
    k[1] * k[0] * C + k[2] * s, c + k[1] * k[1] * C, k[1] * k[2] * C - k[0] * s,
    k[2] * k[0] * C - k[1] * s, k[2] * k[1] * C + k[0] * s, c + k[2] * k[2] * C];
}
function rotvec(R) {
  const c = Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2)); const th = Math.acos(c);
  if (th < 1e-9) return [0, 0, 0];
  const s = 2 * Math.sin(th);
  return [(R[7] - R[5]) / s * th, (R[2] - R[6]) / s * th, (R[3] - R[1]) / s * th];
}
export function pnpRefine(obj, img, K, R0, t0, { iters = 25, huber = 4 } = {}) {
  let p = [...rotvec(R0), ...t0];
  const proj = (q, X) => {
    const R = rodrigues(q.slice(0, 3));
    const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + q[3], y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + q[4], z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + q[5];
    return [K.f * x / z + K.cx, K.fy * y / z + K.cy];
  };
  let lambda = 1e-3, w = new Float64Array(obj.length).fill(1);
  const cost = (q) => { let s = 0; for (let i = 0; i < obj.length; i++) { const u = proj(q, obj[i]); s += w[i] * ((u[0] - img[i][0]) ** 2 + (u[1] - img[i][1]) ** 2); } return s; };
  for (let it = 0; it < iters; it++) {
    // residuals + weights (Huber)
    const res = [];
    for (let i = 0; i < obj.length; i++) { const u = proj(p, obj[i]); const r = [u[0] - img[i][0], u[1] - img[i][1]]; const n = Math.hypot(r[0], r[1]); w[i] = n > huber ? huber / n : 1; res.push(r); }
    // numeric Jacobian 2N x 6
    const J = obj.map(() => [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]);
    for (let k = 0; k < 6; k++) {
      const h = k < 3 ? 1e-5 : 1e-4; const q = p.slice(); q[k] += h;
      for (let i = 0; i < obj.length; i++) { const u = proj(q, obj[i]); const u0 = proj(p, obj[i]); J[i][0][k] = (u[0] - u0[0]) / h; J[i][1][k] = (u[1] - u0[1]) / h; }
    }
    const H = Array.from({ length: 6 }, () => new Float64Array(6)), g = new Float64Array(6);
    for (let i = 0; i < obj.length; i++) for (let r = 0; r < 2; r++) for (let a = 0; a < 6; a++) { g[a] += w[i] * J[i][r][a] * res[i][r]; for (let b = 0; b < 6; b++) H[a][b] += w[i] * J[i][r][a] * J[i][r][b]; }
    const c0 = cost(p);
    for (let tries = 0; tries < 8; tries++) {
      const A = H.map((row, i) => { const r = Array.from(row); r[i] *= (1 + lambda); return [...r, -g[i]]; });
      for (let c = 0; c < 6; c++) {
        let pv = c; for (let r = c + 1; r < 6; r++) if (Math.abs(A[r][c]) > Math.abs(A[pv][c])) pv = r;
        [A[c], A[pv]] = [A[pv], A[c]]; const d = A[c][c] || 1e-30;
        for (let r = 0; r < 6; r++) if (r !== c) { const f = A[r][c] / d; for (let k = c; k < 7; k++) A[r][k] -= f * A[c][k]; }
      }
      const dp = [0, 1, 2, 3, 4, 5].map((i) => A[i][6] / (A[i][i] || 1e-30));
      const q = p.map((v, i) => v + dp[i]);
      if (cost(q) < c0) { p = q; lambda = Math.max(1e-9, lambda / 3); break; }
      lambda *= 10;
    }
  }
  const R = rodrigues(p.slice(0, 3)), t = p.slice(3);
  const errs = obj.map((X, i) => { const u = proj(p, X); return Math.hypot(u[0] - img[i][0], u[1] - img[i][1]); }).sort((a, b) => a - b);
  return { R, t, medPx: errs[errs.length >> 1], inliers: errs.filter((e) => e < 6).length };
}

// ── stage ────────────────────────────────────────────────────────────────────
export async function run(ctx, manifest, hooks) {
  const { session, frames } = ctx;
  const log = hooks.log || (() => {});
  const byName = new Map(frames.map((f) => [f.name, f]));
  const vision = await import(/* @vite-ignore */ `${VISION_CDN}/vision_bundle.mjs`);
  const files = await vision.FilesetResolver.forVisionTasks(`${VISION_CDN}/wasm`);
  const pose = await vision.PoseLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: `${MODELS}pose_landmarker_lite.task`, delegate: 'CPU' }, runningMode: 'IMAGE', numPoses: 1 });
  const face = await vision.FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: `${MODELS}face_landmarker.task`, delegate: 'CPU' }, runningMode: 'IMAGE', numFaces: 1, minFaceDetectionConfidence: 0.4 });
  const cams = session.recon.cams;
  // a collapsed solve (all cameras at one point — a rotation-only solution)
  // cannot triangulate anything; say so instead of fitting a rig to noise
  {
    const C = cams.map((c) => { const R = c.R, t = c.t; return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])]; });
    const m = [0, 1, 2].map((a) => C.reduce((s, c) => s + c[a], 0) / C.length);
    const dist = C.map((c) => Math.hypot(c[0] - m[0], c[1] - m[1], c[2] - m[2])).sort((a, b) => a - b);
    const spread = dist[Math.floor(0.5 * (dist.length - 1))];
    const radius = session.model?.radius || session.recon.sceneRadius || 1;
    log(`cameras: ${cams.length}, median distance from their mean ${spread.toFixed(2)} (scene radius ${radius.toFixed(2)})`);
    if (spread < 0.15 * radius) throw new Error('the cameras did not spread around the person — the video needs a real orbit (walk around, not just turn the phone)');
  }
  const obs = {};                 // body landmark name -> rows
  for (const k of Object.keys(POSE_LM)) obs[k] = [];
  const faceObs = new Map();      // cam index -> Float64Array(478*2) at feature scale
  const nativeSize = new Map();   // cam index -> [W, H]
  const t0 = performance.now();
  const work = new OffscreenCanvas(16, 16); const wctx = work.getContext('2d');
  for (let ci = 0; ci < cams.length; ci++) {
    const c = cams[ci]; const fr = session.frames[c.imgIdx]; const entry = byName.get(fr.name);
    if (!entry) continue;
    const bmp = await createImageBitmap(entry.source);
    nativeSize.set(ci, [bmp.width, bmp.height]);
    // pose on a bounded copy (≤ 1024 px): the lite model does not need more
    const ps = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
    work.width = Math.round(bmp.width * ps); work.height = Math.round(bmp.height * ps);
    wctx.drawImage(bmp, 0, 0, work.width, work.height);
    const pr = pose.detect(work);
    if (pr.landmarks && pr.landmarks[0]) {
      const L = pr.landmarks[0];
      for (const [name, i] of Object.entries(POSE_LM)) {
        const p = L[i]; if ((p.visibility ?? 1) < 0.5 || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) continue;
        obs[name].push({ ci, u: p.x * fr.fw, v: p.y * fr.fh });
      }
      // head window at native resolution for the face landmarker
      const head = L.slice(0, 11).filter((p) => (p.visibility ?? 1) > 0.3);
      if (head.length >= 4) {
        const hx = head.map((p) => p.x * bmp.width), hy = head.map((p) => p.y * bmp.height);
        const ext = Math.max(Math.max(...hx) - Math.min(...hx), Math.max(...hy) - Math.min(...hy));
        const win = Math.round(Math.min(1400, Math.max(384, 3.5 * ext)));
        const x0 = Math.round(Math.min(Math.max(0, (Math.max(...hx) + Math.min(...hx)) / 2 - win / 2), bmp.width - win));
        const y0 = Math.round(Math.min(Math.max(0, (Math.max(...hy) + Math.min(...hy)) / 2 - win / 2), bmp.height - win));
        work.width = win; work.height = win; wctx.drawImage(bmp, x0, y0, win, win, 0, 0, win, win);
        const fres = face.detect(work);
        if (fres.faceLandmarks && fres.faceLandmarks[0]) {
          const F = fres.faceLandmarks[0]; const a = new Float64Array(478 * 2); const sx = fr.fw / bmp.width, sy = fr.fh / bmp.height;
          for (let k = 0; k < 478; k++) { a[k * 2] = (F[k].x * win + x0) * sx; a[k * 2 + 1] = (F[k].y * win + y0) * sy; }
          faceObs.set(ci, a);
        }
      }
    }
    bmp.close();
    hooks.progress?.(ci + 1, cams.length, `finding the joints · ${ci + 1} / ${cams.length}`);
  }
  pose.close(); face.close();
  if (typeof window !== 'undefined') {   // diagnostics for the e2e harness
    const sample = (k) => (obs[k] || []).slice(0, 3);
    window.__avatarDebug = { cams: cams.slice(0, 2).map((c) => ({ imgIdx: c.imgIdx, R: Array.from(c.R), t: Array.from(c.t), f: c.f, fy: c.fy, cx: c.cx, cy: c.cy })),
      frames: cams.slice(0, 2).map((c) => { const f = session.frames[c.imgIdx]; return { name: f.name, fw: f.fw, fh: f.fh, tw: f.tw, th: f.th }; }),
      native: [...nativeSize.values()].slice(0, 2), nose: sample('nose'), l_sho: sample('l_sho'), points: (session.recon.points || []).slice(0, 2).map((p) => Array.from(p.X)),
      noseAll: obs.nose, camsAll: cams.map((c) => ({ R: Array.from(c.R), t: Array.from(c.t), f: c.f, fy: c.fy, cx: c.cx, cy: c.cy })) };
  }
  log(`landmarks: ${cams.length} frames in ${((performance.now() - t0) / 1000).toFixed(0)}s; faces in ${faceObs.size}`);

  // body points -> markers -> fit
  const Ps = cams.map(P);
  const pts = {}; const report = {};
  for (const [name, rows] of Object.entries(obs)) {
    const r = triangulate(rows.map((o) => ({ P: Ps[o.ci], u: o.u, v: o.v })));
    if (r) { pts[name] = r.X; report[name] = { views: r.views, err: +r.err.toFixed(2) }; }
  }
  log(`landmarks: ${['nose', 'l_sho', 'l_hip', 'l_ank'].map((k) => `${k} ${report[k] ? `${report[k].views} views ${report[k].err} px` : 'none'}`).join(' · ')}`);
  const { markers, floorY, missing } = landmarksToMarkers(pts);
  if (missing.length) throw new Error(`could not place ${missing.join(', ')} — is the whole body in the video?`);
  const glb = await (await fetch(RIG_GLB)).arrayBuffer();
  const avatar = parseAvatar(glb);
  const fit = markersToFit(avatar, markers, { asym: false });
  fit.source = { tool: 'avatar-mode', framesUsed: cams.length };
  const resid = Object.values(fit.residualsM); const meanRes = resid.reduce((a, b) => a + b, 0) / resid.length;
  log(`fit: scale ${fit.scale.toFixed(3)}, marker residual mean ${(meanRes / fit.scale * 100).toFixed(1)} cm`);

  // face: canonical 3D face + head-stabilised crop cameras
  const canon = []; const good = [];
  if (faceObs.size >= MIN_VIEWS) {
    for (let k = 0; k < 478; k++) {
      const rows = []; for (const [ci, a] of faceObs) rows.push({ P: Ps[ci], u: a[k * 2], v: a[k * 2 + 1] });
      const r = triangulate(rows, 1.5);
      canon.push(r ? r.X : null); if (r && r.err < 20) good.push(k);
    }
  }
  // did the head move against the room over the orbit? The nose tip triangulated
  // from the first and the last third of the face views (by capture order); the
  // room's poses are consistent with the room, so a shift here is the head's own.
  // Filip 1.7-2.5 cm (his face doubled), Tom's did not move (2026-09-14).
  let headShiftCm = null;
  if (faceObs.size >= 2 * MIN_VIEWS) {
    const order = [...faceObs.keys()].sort((a, b) => a - b); const third = Math.floor(order.length / 3);
    const seg = (idx) => { const rows = []; for (const ci of idx) { const a = faceObs.get(ci); rows.push({ P: Ps[ci], u: a[2], v: a[3] }); } return triangulate(rows, 1.5); };   // landmark 1 = nose tip
    const A = seg(order.slice(0, Math.max(MIN_VIEWS, third))), B = seg(order.slice(-Math.max(MIN_VIEWS, third)));
    if (A && B) headShiftCm = +(Math.hypot(A.X[0] - B.X[0], A.X[1] - B.X[1], A.X[2] - B.X[2]) / fit.scale * 100).toFixed(2);
    log(`face: nose tip first vs last third of the orbit ${headShiftCm != null ? headShiftCm + ' cm' : 'n/a'} (the head against the room)`);
  }
  if (typeof window !== 'undefined' && window.__avatarDebug) window.__avatarDebug.face = { points: good.map((k) => canon[k]), ids: good };   // tests: the triangulated face
  const cropCams = [];
  if (good.length > 100) {
    const obj = good.map((k) => canon[k]);
    for (const [ci, a] of faceObs) {
      const c = cams[ci]; const fr = session.frames[c.imgIdx]; const [W, H] = nativeSize.get(ci);
      const S = W / fr.fw;                                         // feature -> native
      const K = { f: c.f * S, fy: (c.fy ?? c.f) * S, cx: c.cx * S, cy: c.cy * S };
      const img = good.map((k) => [a[k * 2] * S, a[k * 2 + 1] * S]);
      const r = pnpRefine(obj, img, K, c.R, c.t);
      if (r.medPx > 4 || r.inliers < 60) continue;
      // window around the projected face, hair above it
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (const X of obj) {
        const x = r.R[0] * X[0] + r.R[1] * X[1] + r.R[2] * X[2] + r.t[0], y = r.R[3] * X[0] + r.R[4] * X[1] + r.R[5] * X[2] + r.t[1], z = r.R[6] * X[0] + r.R[7] * X[1] + r.R[8] * X[2] + r.t[2];
        const u = K.f * x / z + K.cx, v = K.fy * y / z + K.cy; x0 = Math.min(x0, u); x1 = Math.max(x1, u); y0 = Math.min(y0, v); y1 = Math.max(y1, v);
      }
      const cxf = (x0 + x1) / 2, cyf = (y0 + y1) / 2 - 0.25 * (y1 - y0);
      const wx = Math.round(Math.min(Math.max(0, cxf - SIDE / 2), W - SIDE)), wy = Math.round(Math.min(Math.max(0, cyf - SIDE / 2), H - SIDE));
      cropCams.push({ name: fr.name, R: r.R, t: r.t, f: K.f, fy: K.fy, cx: K.cx - wx, cy: K.cy - wy, x0: wx, y0: wy, side: SIDE, facePx: +(x1 - x0).toFixed(0), medPx: +r.medPx.toFixed(2) });
    }
  }
  if (faceObs.size >= MIN_VIEWS) {
    const errs = canon.map((c, k) => c ? k : -1).filter((k) => k >= 0).length;
    log(`face: ${errs} points triangulated of 478 from ${faceObs.size} views`);
  }
  log(`face: ${good.length}/478 canonical points, ${cropCams.length} head-stabilised crop cameras`);
  return {
    markers, floorY, landmarks: pts, report, fit,
    face: { points: good.map((k) => canon[k].map((v) => +v.toFixed(5))), ids: good },
    cropCams, headShiftCm,
    // the dock's line is where this lands now that the joints card is gone
    note: `${Object.keys(markers).length} markers · rig scale ${fit && fit.scale ? fit.scale.toFixed(2) : '?'} · ${cropCams.length} face views`,
  };
}

/** The markers drawn on three frames — shown while the run carries on, never
 *  waited for (the user, 2026-09-18: don't ask whether it looks right, show
 *  it). The picture stays on the card until a later stage has one of its own. */
export async function preview(ctx, manifest, hooks) {
  const st = manifest.stages.landmarks; if (!hooks.shots || !st || !st.markers) return;
  const { session, frames } = ctx; const byName = new Map(frames.map((f) => [f.name, f]));
  const cams = session.recon.cams.filter((c) => !c.crop); const picks = [0, cams.length >> 2, cams.length >> 1].map((i) => cams[i]);
  const strip = document.createElement('div'); strip.className = 'av-overlay';
  for (const c of picks) {
    const fr = session.frames[c.imgIdx]; const entry = byName.get(fr.name); if (!entry) continue;
    // drawn at 2x, shown 240 px tall — the card is 880 px wide since 2026-09-18, so
    // three frames fit side by side instead of scrolling inside 340 px
    const bmp = await createImageBitmap(entry.source); const h = 480, w = Math.round(bmp.width / bmp.height * h);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h; cv.style.height = '240px'; cv.style.width = `${w / 2}px`;
    const g = cv.getContext('2d'); g.drawImage(bmp, 0, 0, w, h); bmp.close();
    const p = P(c); const s = w / fr.fw;
    g.fillStyle = '#19e3c2'; g.strokeStyle = '#19e3c2'; g.lineWidth = 2;
    for (const [name, X] of Object.entries(st.markers)) { const q = reproj(X, p); if (!q) continue; g.beginPath(); g.arc(q[0] * s, q[1] * s, 5, 0, Math.PI * 2); g.fill(); }
    for (const [a, b] of [['Hips', 'Spine2'], ['Spine2', 'Head'], ['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'], ['RightArm', 'RightForeArm'], ['RightForeArm', 'RightHand'], ['Hips', 'LeftLeg'], ['LeftLeg', 'LeftFoot'], ['LeftFoot', 'LeftToe_End'], ['Hips', 'RightLeg'], ['RightLeg', 'RightFoot'], ['RightFoot', 'RightToe_End'], ['Spine2', 'LeftArm'], ['Spine2', 'RightArm']]) {
      const qa = reproj(st.markers[a], p), qb = reproj(st.markers[b], p); if (!qa || !qb) continue;
      g.beginPath(); g.moveTo(qa[0] * s, qa[1] * s); g.lineTo(qb[0] * s, qb[1] * s); g.stroke();
    }
    strip.appendChild(cv);
  }
  // built off-screen and handed over in one piece: the card never blinks empty
  hooks.shots(strip, `The joints on your frames — rig scale ${st.fit.scale.toFixed(2)}, ${st.cropCams.length} frames show the face`);
}
