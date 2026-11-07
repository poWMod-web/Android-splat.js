// headseed.js — a dense seed on the WHOLE head, protected from relocation.
//
// The face seed (faceseed.js) covers the face only and was pruned back to
// the trainer's own density. This one takes the body model's head — Anny
// fitted to the joints (no splat needed), its head registered to the 478
// face points as the body-fit stage does after training — samples the head
// skin uniformly, colours each sample from the camera that looks at it most
// squarely, and hands the rows to the seed with a relocation-protection
// window (trainer.protect), so the head keeps its density while the rest
// of the model finds its own. The user's request, 2026-09-14.
import { loadAnny } from './body/anny.js';
import { fitBody } from './body/fit.js';
import { headCorrespondences, deformHead } from './body/head.js';

const MODEL = new URL('../models/anny_game_engine.bin', import.meta.url).href;
const MODELS = new URL('../models/', import.meta.url).href;
const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const logit = (v) => { const c = Math.max(0.02, Math.min(0.98, v)); return Math.log(c / (1 - c)); };

/** @returns {Promise<{data: Float32Array, n: number, note: string, protectIters: number}|null>} */
export async function headSeedGaussians(session, files, lm, { count = 50000, protectIters = 8000, minHead = 0.2, log = () => {}, progress } = {}) {
  if (!lm || !lm.landmarks) { log('head seed: no landmarks — skipped'); return null; }
  progress?.(0, 1, 'fitting the body model to the joints …');
  const anny = await loadAnny(MODEL);
  const fit = await fitBody(anny, { landmarks: lm.landmarks, report: lm.report, splat: null, log, progress: (m, f) => progress?.(f, 1, m) });
  const S = fit.surface; const hb = anny.boneIndex.head;
  const headWeight = S.boneIndices.map((bi, i) => bi.reduce((s, b, k) => s + (b === hb ? S.boneWeights[i][k] : 0), 0));
  // the head onto the face points, as the body-fit stage does
  const face = lm.face && lm.face.points && lm.face.points.length > 100 ? new Map(lm.face.ids.map((id, i) => [id, lm.face.points[i]])) : null;
  const cams = (lm.cropCams || []).filter((c) => c.facePx > 0);
  if (face && cams.length) {
    try {
      const vision = await import(/* @vite-ignore */ `${VISION_CDN}/vision_bundle.mjs`);
      const vf = await vision.FilesetResolver.forVisionTasks(`${VISION_CDN}/wasm`);
      const landmarker = await vision.FaceLandmarker.createFromOptions(vf, { baseOptions: { modelAssetPath: `${MODELS}face_landmarker.task`, delegate: 'CPU' }, runningMode: 'IMAGE', numFaces: 1, minFaceDetectionConfidence: 0.2, minFacePresenceConfidence: 0.2 });
      const cam = cams.slice().sort((a, b) => b.facePx - a.facePx)[0];
      const { corr } = headCorrespondences(S, headWeight, cam, landmarker); landmarker.close();
      if (corr.size > 200) deformHead(S, headWeight, corr, face, log, fit.scale);
    } catch (e) { log(`head seed: head registration skipped (${e.message || e})`); }
  }
  // the head triangles and their area
  const V = S.vertices; const tris = S.faces.filter(([a, b, c]) => headWeight[a] > minHead && headWeight[b] > minHead && headWeight[c] > minHead);
  if (tris.length < 50) { log(`head seed: only ${tris.length} head triangles — skipped`); return null; }
  const areas = tris.map(([a, b, c]) => 0.5 * norm(cross(sub(V[b], V[a]), sub(V[c], V[a])))); const total = areas.reduce((s, v) => s + v, 0);
  const cum = []; let acc = 0; for (const a of areas) { acc += a; cum.push(acc); }
  const hc = [0, 0, 0]; let nh = 0; for (let i = 0; i < V.length; i++) if (headWeight[i] > 0.5) { hc[0] += V[i][0]; hc[1] += V[i][1]; hc[2] += V[i][2]; nh++; }
  hc[0] /= nh || 1; hc[1] /= nh || 1; hc[2] /= nh || 1;
  // cameras for colours: every registered body frame, its centre and native scale
  const byName = new Map(files.map((f) => [f.name, f]));
  const bodyCams = session.recon.cams.filter((c) => !c.crop).map((c) => { const R = c.R, t = c.t; return { c, centre: [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])], name: session.frames[c.imgIdx].name, fw: session.frames[c.imgIdx].fw }; });
  // samples
  const samples = new Array(count);
  for (let i = 0; i < count; i++) {
    const r = Math.random() * total; let ti = cum.findIndex((v) => v >= r); if (ti < 0) ti = tris.length - 1;
    const [a, b, c] = tris[ti]; const A = V[a], B = V[b], C = V[c];
    let u = Math.random(), v = Math.random(); if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const X = [A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0]), A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1]), A[2] + u * (B[2] - A[2]) + v * (C[2] - A[2])];
    let n = cross(sub(B, A), sub(C, A)); const nl = norm(n) || 1; n = [n[0] / nl, n[1] / nl, n[2] / nl];
    const out = sub(X, hc); if (n[0] * out[0] + n[1] * out[1] + n[2] * out[2] < 0) n = [-n[0], -n[1], -n[2]];   // outward: away from the head's centre
    // the camera that looks at this point most squarely
    let best = -1, bd = -0.2;
    for (let k = 0; k < bodyCams.length; k++) { const d = sub(bodyCams[k].centre, X); const dl = norm(d) || 1; const dot = (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / dl; if (dot > bd) { bd = dot; best = k; } }
    samples[i] = { X, n, cam: best, rgb: [0.55, 0.42, 0.38] };
  }
  // colours: decode each chosen frame once, read the pixel under every sample assigned to it
  const groups = new Map(); samples.forEach((s, i) => { if (s.cam >= 0) { if (!groups.has(s.cam)) groups.set(s.cam, []); groups.get(s.cam).push(i); } });
  let done = 0;
  for (const [k, idx] of groups) {
    const bc = bodyCams[k]; const file = byName.get(bc.name); if (!file) continue;
    const bmp = await createImageBitmap(file.source !== undefined ? file.source : file); const W = bmp.width, H = bmp.height; const nat = W / bc.fw;
    const cv = new OffscreenCanvas(W, H); const g = cv.getContext('2d', { willReadFrequently: true }); g.drawImage(bmp, 0, 0); bmp.close();
    const px = g.getImageData(0, 0, W, H).data; const R = bc.c.R, t = bc.c.t, f = bc.c.f * nat, fy = (bc.c.fy ?? bc.c.f) * nat, cx = bc.c.cx * nat, cy = bc.c.cy * nat;
    for (const i of idx) {
      const X = samples[i].X; const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0], y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1], z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
      if (z <= 0) continue; const uu = Math.round(f * x / z + cx), vv = Math.round(fy * y / z + cy);
      if (uu >= 0 && vv >= 0 && uu < W && vv < H) { const o = (vv * W + uu) * 4; samples[i].rgb = [px[o] / 255, px[o + 1] / 255, px[o + 2] / 255]; }
    }
    done += idx.length; progress?.(done, count, `colouring the head seed · ${done} / ${count}`);
    await new Promise((r) => setTimeout(r, 0));
  }
  // the rows: flat discs, tangent scale from the sample spacing
  const spacing = Math.sqrt(total / count); const sTan = Math.log(spacing * 0.7), sNrm = Math.log(spacing * 0.2);
  const data = new Float32Array(count * 16);
  for (let i = 0; i < count; i++) {
    const s = samples[i], n = s.n, b = i * 16;
    let q; if (n[2] < -0.9999) q = [0, 1, 0, 0]; else { const w = 1 + n[2]; const ql = Math.hypot(w, -n[1], n[0]); q = [w / ql, -n[1] / ql, n[0] / ql, 0]; }
    data[b] = s.X[0]; data[b + 1] = s.X[1]; data[b + 2] = s.X[2]; data[b + 3] = sTan; data[b + 4] = sTan; data[b + 5] = sNrm;
    data[b + 6] = q[0]; data[b + 7] = q[1]; data[b + 8] = q[2]; data[b + 9] = q[3];
    data[b + 10] = logit(s.rgb[0]); data[b + 11] = logit(s.rgb[1]); data[b + 12] = logit(s.rgb[2]); data[b + 13] = logit(0.6);
  }
  log(`head seed: ${count} discs on ${tris.length} head triangles (area ${(total / fit.scale / fit.scale * 1e4).toFixed(0)} cm², spacing ${(spacing / fit.scale * 1000).toFixed(1)} mm), coloured from ${groups.size} frames, protected for ${protectIters} iterations`);
  // the head's bounding sphere: the caller clears the sparse-cloud seed inside it, so the
  // discs are not buried under the cloud's neighbour-sized blobs (iteration 0 was fog, 2026-09-14)
  let rad = 0; for (let i = 0; i < V.length; i++) if (headWeight[i] > 0.5) rad = Math.max(rad, norm(sub(V[i], hc)));
  return { data, n: count, note: `head mesh, spacing ${(spacing / fit.scale * 1000).toFixed(1)} mm`, protectIters, head: { centre: hc, radius: rad } };
}
