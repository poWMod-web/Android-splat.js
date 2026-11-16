// facepolish.js — the face trained on its own, then transplanted.
//
// The smoothest face we have (2026-09-14) came from an experiment that never
// saw anything but the face: 20k flat discs on the triangulated face mesh,
// trained on the head crops with every pixel outside the landmark contour
// masked out. Its skin is clean because no splat has to serve hair, room or
// silhouette. On its own it has a halo of unsupervised splats around the
// face and nothing above or behind. Here it becomes a stage: train that
// model in the SAME frame as the avatar, cut it to the face mesh's
// neighbourhood, remove the avatar's own face splats there, and put the
// clean face in. Hair, ears, back and body stay the pipeline's.
import { createSession } from '../../../src/index.js';
import { faceSeedGaussians } from '../faceseed.js';
import { addPersonCrops } from '../crops.js';

export const id = 'facepolish';
export const needs = ['cut'];
const STRIDE = 16;
const ITERS = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('polishiters')) || 8000);
const HULL = 0.94;          // the loss region: the projected landmark hull scaled about its centroid
const KEEP_MM = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('polishkeep')) || 30);      // the face-only model survives this close to the face mesh
const REPLACE_MM = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('polishreplace')) || 25); // the avatar's own splats this close are replaced
const STITCH = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('polishstitch')) || 1500);   // iterations on the merged model: no growth, no relocation — the seam and the tone settle
const MIN_OP = 0.05;        // dead face splats stay out (the halo is killed in training now: random background outside the contour)

async function cutCrop(entry, cam) {
  const bmp = await createImageBitmap(entry.source !== undefined ? entry.source : entry);
  const cv = new OffscreenCanvas(cam.side, cam.side); cv.getContext('2d').drawImage(bmp, cam.x0, cam.y0, cam.side, cam.side, 0, 0, cam.side, cam.side);
  const W = bmp.width; bmp.close();
  return { source: await cv.convertToBlob({ type: 'image/jpeg', quality: 0.95 }), name: `polish_${cam.name}`, W };
}

export async function run(ctx, manifest, hooks) {
  const log = hooks.log || (() => {}); const lm = manifest.stages.landmarks; const src = ctx.session; const files = ctx.frames;
  const face = lm && lm.face; const crops = (lm && lm.cropCams) || [];
  if (!face || !face.points || face.points.length < 100 || crops.length < 6) return { skipped: true, note: 'no face crops' };
  const P = face.points; const scale = (lm.fit && lm.fit.scale) || 1;
  const useFacePose = !!(lm.report && lm.report.nose && +lm.report.nose.err > 2.5);   // the head-window rule
  hooks.progress?.(0, 5, 'seeding the face …');
  const seed = await faceSeedGaussians(src, files, lm, { count: 20000, log });
  if (!seed) return { skipped: true, note: 'no seed' };
  // the face crops: native windows from the landmarks stage, cameras on the room pose (or the face pose)
  const byName = new Map(files.map((f) => [f.name, f])); const bodyByName = new Map(src.recon.cams.filter((c) => !c.crop).map((c) => [src.frames[c.imgIdx].name, c]));
  const entries = [], meta = [];
  for (const c of crops) {
    const e = byName.get(c.name); const body = bodyByName.get(c.name); if (!e || !body) continue;
    const ce = await cutCrop(e, c); entries.push({ source: ce.source, name: ce.name }); meta.push({ c, body, nat: ce.W / src.frames[body.imgIdx].fw });
  }
  hooks.progress?.(1, 5, 'training the face on its own …');
  // the face-only model trains the MASKED way: inside the landmark contour the crop pixels,
  // outside a random background — that is what kills the halo (splats that wander out are
  // trained to transparency) while the face itself is untouched by silhouette pressure
  const F = createSession({ maxIters: ITERS, evalSplit: 0, holdout: -1, frames: { trainMaxDim: 768 }, maskTraining: true, shHorizontal: src.opts.shHorizontal !== false, device: src.opts.device,
    trainer: { shDeg: 3, maxSplats: 300000, capMult: 8 } });   // no anisoReg: the discs must stay flat (with it the face came back as round blobs, needle 1.1, and soft)
  await F.load(entries);
  const cams = [];
  for (let i = 0; i < meta.length; i++) {
    const { c, body, nat } = meta[i]; const fr = F.frames[i]; const s = fr.fw / c.side;
    const cam = useFacePose ? { imgIdx: i, R: c.R, t: c.t, f: c.f * s, fy: c.fy * s, cx: c.cx * s, cy: c.cy * s }
      : { imgIdx: i, R: body.R, t: body.t, f: body.f * nat * s, fy: (body.fy ?? body.f) * nat * s, cx: (body.cx * nat - c.x0) * s, cy: (body.cy * nat - c.y0) * s };
    cams.push(cam);
    // the loss region: the landmark hull, scaled about its centroid; everything else invalid
    const st = fr.tw / fr.fw; const pts = [];
    for (const p of P) { const x = cam.R[0] * p[0] + cam.R[1] * p[1] + cam.R[2] * p[2] + cam.t[0], y = cam.R[3] * p[0] + cam.R[4] * p[1] + cam.R[5] * p[2] + cam.t[1], z = cam.R[6] * p[0] + cam.R[7] * p[1] + cam.R[8] * p[2] + cam.t[2]; if (z > 0) pts.push([(cam.f * x / z + cam.cx) * st, (cam.fy * y / z + cam.cy) * st]); }
    pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]); const cz = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = []; for (const p of pts) { while (lo.length >= 2 && cz(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
    const up = []; for (let k = pts.length - 1; k >= 0; k--) { const p = pts[k]; while (up.length >= 2 && cz(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
    const hull = lo.slice(0, -1).concat(up.slice(0, -1)); if (hull.length < 3) continue;
    const hc = hull.reduce((a, p) => [a[0] + p[0] / hull.length, a[1] + p[1] / hull.length], [0, 0]);
    const mcv = new OffscreenCanvas(fr.tw, fr.th); const mg = mcv.getContext('2d', { willReadFrequently: true }); mg.fillStyle = '#000'; mg.fillRect(0, 0, fr.tw, fr.th); mg.fillStyle = '#fff'; mg.beginPath();
    hull.forEach((p, k) => { const x = hc[0] + (p[0] - hc[0]) * HULL, y = hc[1] + (p[1] - hc[1]) * HULL; if (k === 0) mg.moveTo(x, y); else mg.lineTo(x, y); }); mg.closePath(); mg.fill();
    const md = mg.getImageData(0, 0, fr.tw, fr.th).data; const n = fr.tw * fr.th; const alpha = new Uint8Array(n); let empty = 0;
    for (let p = 0; p < n; p++) { alpha[p] = md[p * 4] < 128 ? 0 : 255; if (!alpha[p]) empty++; }
    fr.alpha = alpha; fr.emptyFrac = empty / n; fr.maskedFrac = 0;
  }
  const radius = 0.3 * scale;
  F.useReconstruction({ ...src.recon, cams, points: [], sceneRadius: radius, frames: F.frames.map((f) => ({ name: f.name, fw: f.fw, fh: f.fh, tw: f.tw, th: f.th })) });
  await F.seedFrom({ data: seed.data, n: seed.n, sh: null, shK: null, dc: 'sigmoid' }, { iter: 0, sceneRadius: radius });
  log(`face polish: ${cams.length} face crops (${useFacePose ? 'face-PnP' : 'room'} poses), ${seed.n} discs, ${ITERS} iterations`);
  await new Promise((resolve, reject) => {
    F.on('metrics', (e) => { if (e.iter != null) hooks.progress?.(e.iter, ITERS, `polishing the face · ${e.iter} / ${ITERS}${e.psnrTrain != null ? ` · ${e.psnrTrain.toFixed(1)} dB` : ''}`); });
    F.on('event', (e) => { if (e.kind === 'train-complete') resolve(); if (e.kind === 'device-lost' || e.kind === 'error') reject(new Error(e.message || e.kind)); });
    F.start();
  });
  // transplant: the face model within KEEP_MM of the mesh replaces the avatar's splats within REPLACE_MM
  hooks.progress?.(4, 5, 'transplanting the face …');
  const faceRaw = await F.exportRawState(); const avRaw = await src.exportRawState();
  const near = (data, n, mm) => { const keep = new Uint8Array(n); const lim2 = (mm / 1000 * scale) ** 2; for (let i = 0; i < n; i++) { const b = i * STRIDE; const x = data[b], y = data[b + 1], z = data[b + 2]; for (const p of P) { const dx = x - p[0], dy = y - p[1], dz = z - p[2]; if (dx * dx + dy * dy + dz * dz < lim2) { keep[i] = 1; break; } } } return keep; };
  const fKeep = near(faceRaw.data, faceRaw.n, KEEP_MM), aNear = near(avRaw.data, avRaw.n, REPLACE_MM);
  for (let i = 0; i < faceRaw.n; i++) if (fKeep[i] && 1 / (1 + Math.exp(-faceRaw.data[i * STRIDE + 13])) < MIN_OP) fKeep[i] = 0;   // the haze stays out
  const nF = fKeep.reduce((s, v) => s + v, 0), nA = avRaw.n - aNear.reduce((s, v) => s + v, 0); const shK = avRaw.shK || 0;
  const data = new Float32Array((nA + nF) * STRIDE); const sh = shK ? new Float32Array((nA + nF) * shK * 3) : null; let j = 0;
  const put = (raw, keepFn, i) => { data.set(raw.data.subarray(i * STRIDE, (i + 1) * STRIDE), j * STRIDE); if (sh && raw.sh && raw.shK === shK) sh.set(raw.sh.subarray(i * shK * 3, (i + 1) * shK * 3), j * shK * 3); j++; };
  for (let i = 0; i < avRaw.n; i++) if (!aNear[i]) put(avRaw, null, i);
  for (let i = 0; i < faceRaw.n; i++) if (fKeep[i]) put(faceRaw, null, i);
  log(`face polish: ${nF} face splats replace ${avRaw.n - nA} of the avatar's within ${REPLACE_MM} mm of the face mesh`);
  const iter0 = src.trainer.iter;
  // the stitch trains the MASKED way (random background outside the matte): with the room
  // merely excluded the splats outside the person bloomed unsupervised (2026-09-15)
  const M = createSession({ ...src.opts, maxIters: iter0 + STITCH, evalSplit: 0, holdout: -1, maskTraining: true, trainer: { ...(src.opts.trainer || {}), growUntil: 0, relocUntil: 0 } });
  const srcEntries = ctx.sessionEntries || src.recon.cams.map((c) => byName.get(src.frames[c.imgIdx].name)).filter(Boolean);
  await M.load(srcEntries);
  const byName2 = new Map(M.frames.map((f, i) => [f.name, i])); const mcams = [];
  for (const c of src.recon.cams) { const i = byName2.get(src.frames[c.imgIdx].name); if (i == null) continue; const s = M.frames[i].fw / src.frames[c.imgIdx].fw; mcams.push({ ...c, imgIdx: i, f: c.f * s, ...(c.fy != null ? { fy: c.fy * s } : {}), cx: c.cx * s, cy: c.cy * s }); }
  M.useReconstruction({ ...src.recon, cams: mcams, points: src.recon.points });
  // the merged model is the PERSON only: the stitch must not see the room (the first stitch
  // trained a cut-out against full frames and exploded into fog) — body frames supervise
  // the matte's interior only, and the person crops come back at native resolution
  try { await addPersonCrops(M, files, { log, faceCams: lm.cropCams || null, facePoints: P, headMoved: useFacePose }); } catch (e) { log(`face polish: crops for the stitch skipped (${e.message || e})`); }
  await M.seedFrom({ data, n: nA + nF, sh, shK, dc: avRaw.dc }, { iter: iter0 });
  if (STITCH > 0) {
    // the transplanted face and the avatar around it settle together: a short run of the
    // merged model on every camera at the schedule's end (low learning rates), nothing
    // grows or moves rows — the seam's tone and edge are what change
    log(`face polish: stitching for ${STITCH} iterations`);
    await new Promise((resolve, reject) => {
      M.on('metrics', (e) => { if (e.iter != null) hooks.progress?.(e.iter - iter0, STITCH, `stitching the face · ${e.iter - iter0} / ${STITCH}`); });
      M.on('event', (e) => { if (e.kind === 'train-complete') resolve(); if (e.kind === 'device-lost' || e.kind === 'error') reject(new Error(e.message || e.kind)); });
      M.start();
    });
  }
  ctx.session = M; ctx.plyBlob = null; ctx.sogBlob = null;
  return { faceSplats: nF, replaced: avRaw.n - nA, note: `${nF.toLocaleString()} face splats in` };
}
