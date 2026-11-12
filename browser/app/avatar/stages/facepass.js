// facepass.js — sharpen the face: native-resolution crops as extra cameras,
// then continue training the isolated splat (the 09-12d recipe, in the tab).
//
// The landmarks stage left head-stabilised crop cameras (R, t, f, cx, cy at
// native scale and the window to cut). Here each window is cut out of the
// native pick (+ its matte), a SECOND session is loaded with the body frames
// and the crops, handed the trained model (raw state, no bake) and the merged
// reconstruction, and trained on for a while with the crops sampled three
// times. The app keeps its viewer on the first session; the avatar pipeline
// continues with this one (ctx.session becomes it).
import { createSession } from '../../../src/index.js';

export const id = 'facepass';
export const needs = ['landmarks'];
// OFF by default since 2026-09-14: on a room-trained model (maskTraining false) the
// user saw no sharpness gain from the pass and streaks around the head — the 1263 px
// room training already resolves the face. ?faceiters=N runs it (tests, experiments).
const EXTRA_ITERS = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('faceiters')) || 0);
const CROP_WEIGHT = 3;
// The face pass on a room-trained model grew streaks around the head (needle ratio
// median 32); the anisotropy regulariser at 0.01 takes them out (2.8) and keeps the
// sharpness — a scale floor of 1e-3 only halved them (7.7). Measured on Tom's clip,
// 2026-09-14, close-ups from the same camera. ?faceaniso= / ?faceminscale= override.
const FACE_GROW = +((typeof location !== 'undefined' && new URLSearchParams(location.search).get('facegrow')) || 0.1);   // growth per refine in the pass
const FACE_TRAINER = (() => { const q = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null; const o = { anisoReg: 0.01 }; if (q && q.get('faceaniso')) o.anisoReg = +q.get('faceaniso'); if (q && q.get('faceminscale')) o.minScale = +q.get('faceminscale'); return o; })();

async function cutCrop(entry, cam) {
  const bmp = await createImageBitmap(entry.source);
  const cv = new OffscreenCanvas(cam.side, cam.side); const g = cv.getContext('2d');
  g.drawImage(bmp, cam.x0, cam.y0, cam.side, cam.side, 0, 0, cam.side, cam.side); bmp.close();
  const source = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.95 });
  let mask = null;
  if (entry.mask) {
    // the matte is at a bounded size: scale it to native, then cut the same window
    const mb = await createImageBitmap(entry.mask);
    const W = entry._nativeW || (await createImageBitmap(entry.source)).width;
    const s = W / mb.width;
    const mc = new OffscreenCanvas(cam.side, cam.side); const mg = mc.getContext('2d');
    mg.imageSmoothingEnabled = true;
    mg.drawImage(mb, cam.x0 / s, cam.y0 / s, cam.side / s, cam.side / s, 0, 0, cam.side, cam.side); mb.close();
    mask = await mc.convertToBlob({ type: 'image/png' });
  }
  return { source, mask, name: `face_${entry.name}` };
}

export async function run(ctx, manifest, hooks) {
  const { session, frames } = ctx; const log = hooks.log || (() => {});
  const lm = manifest.stages.landmarks;
  const crops = lm.cropCams || [];
  if (!(EXTRA_ITERS > 0)) return { note: 'off — the room training resolves the face', skipped: true };
  if (crops.length < 6) return { note: `${crops.length} face views — skipped`, skipped: true };
  const byName = new Map(frames.map((f) => [f.name, f]));
  // 1. cut the windows
  const cropEntries = [];
  for (let k = 0; k < crops.length; k++) {
    const cam = crops[k]; const entry = byName.get(cam.name); if (!entry) continue;
    cropEntries.push({ ...(await cutCrop(entry, cam)), cam });
    hooks.progress?.(k + 1, crops.length, `cutting the face windows · ${k + 1} / ${crops.length}`);
  }
  // 2. a second session: body frames + crops, the trained model, merged cameras
  const iter0 = session.trainer.iter;
  const raw = await session.exportRawState();
  const t0 = session.opts?.trainer || {};
  const ses = createSession({
    maxIters: iter0 + EXTRA_ITERS, evalSplit: 0, holdout: -1,
    maxViewW: session.opts?.maxViewW, maxViewH: session.opts?.maxViewH,
    frames: { trainMaxDim: 1600 },
    // growth and relocation are gated on the RUN's horizon (grow until 75 % of
    // maxIters, relocate until the anneal end): continuing from 20k for 12k
    // left the pass 4k iterations of growth spread over the whole room — the
    // face gained 7 % splats, not the doubling the crops are there for
    // (2026-09-14). The pass gets its own window: grow and relocate for 80 %
    // of its length, with a faster growth rate.
    trainer: { ...t0, maxSplats: Math.max(t0.maxSplats || 0, 1000000), capMult: 8, lrWarmup: 1000,
      growUntil: iter0 + Math.round(0.8 * EXTRA_ITERS), relocUntil: iter0 + Math.round(0.8 * EXTRA_ITERS), growRate: FACE_GROW, ...FACE_TRAINER },
    ...(session.opts?.maskTraining === false ? { maskTraining: false } : {}),   // the room stays in the picture; the cut comes after
  });
  const bodyEntries = session.recon.cams.map((c) => byName.get(session.frames[c.imgIdx].name)).filter(Boolean);
  const entries = [...bodyEntries, ...cropEntries.map(({ source, mask, name }) => ({ source, mask, name }))];
  await ses.load(entries);
  ctx.sessionEntries = entries;   // the cut stage rebuilds a session from the same files
  const byName2 = new Map(ses.frames.map((f, i) => [f.name, i]));
  const cams = [];
  for (const c of session.recon.cams) {
    const i = byName2.get(session.frames[c.imgIdx].name); if (i == null) continue;
    const s = ses.frames[i].fw / session.frames[c.imgIdx].fw;
    cams.push({ ...c, imgIdx: i, f: c.f * s, ...(c.fy != null ? { fy: c.fy * s } : {}), cx: c.cx * s, cy: c.cy * s });
  }
  for (const e of cropEntries) {
    const i = byName2.get(e.name); if (i == null) continue;
    const s = ses.frames[i].fw / e.cam.side;                      // crop cams are at native crop scale
    const cam = { imgIdx: i, R: e.cam.R, t: e.cam.t, f: e.cam.f * s, fy: e.cam.fy * s, cx: e.cam.cx * s, cy: e.cam.cy * s };
    const w = Math.max(1, Math.min(6, Math.round(CROP_WEIGHT * e.cam.facePx / 250)));   // the close faces carry the most
    for (let k = 0; k < w; k++) cams.push(cam);
  }
  ses.useReconstruction({ ...session.recon, cams, points: session.recon.points });
  await ses.seedFrom(raw, { iter: iter0 });
  log(`face pass: ${bodyEntries.length} body frames + ${cropEntries.length} crops (${cams.length - session.recon.cams.length} crop samples), continuing from ${iter0} for ${EXTRA_ITERS}`);
  // 3. train
  await new Promise((resolve, reject) => {
    ses.on('metrics', (e) => { if (e.iter != null) hooks.progress?.(e.iter - iter0, EXTRA_ITERS, `sharpening the face · ${e.iter - iter0} / ${EXTRA_ITERS}${e.psnrTrain != null ? ` · ${e.psnrTrain.toFixed(1)} dB` : ''}`); });
    ses.on('event', (e) => { if (e.kind === 'train-complete') resolve(); if (e.kind === 'device-lost' || e.kind === 'error') reject(new Error(e.message || e.kind)); });
    ses.start();
  });
  ctx.session = ses;       // the sharper model is the one to bind and ship
  ctx.faceSession = ses;
  return { iters: EXTRA_ITERS, crops: cropEntries.length, splats: ses.trainer.n, note: `${cropEntries.length} crops · +${EXTRA_ITERS}` };
}
