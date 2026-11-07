// bodyfit.js — a fitted body model as the binding surface, in the tab.
//
// The Anny body (app/avatar/body/anny.js, a 1.5 MB PCA snapshot of the
// game_engine model) is fitted to the triangulated landmarks and pulled onto
// the splat (body/fit.js), stripped to its outer skin, and its head is
// registered to the 468 face points through correspondences found on a
// render of the head (body/head.js). What comes out is what the CLI
// pipeline produced yesterday: a surface in the PLY frame for
// export-binding-core and rig markers from the fitted skeleton, which
// autofit-core turns into the fit JSON (asymmetric — the fitted body keeps
// the person's stance).
import { loadAnny } from '../body/anny.js';
import { fitBody } from '../body/fit.js';
import { headCorrespondences, deformHead } from '../body/head.js';
import { plyCenters } from '../rig/export-binding-core.js';
import { markersToFit } from '../rig/autofit-core.js';
import { parseAvatar } from '../rig/glb.js';

export const id = 'bodyfit';
export const needs = ['landmarks'];
const MODEL = new URL('../../models/anny_game_engine.bin', import.meta.url).href;
const RIG_GLB = new URL('../rig/glb_avatar.glb', import.meta.url).href;
const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const MODELS = new URL('../../models/', import.meta.url).href;

export async function run(ctx, manifest, hooks) {
  const log = hooks.log || (() => {}); const lm = manifest.stages.landmarks;
  hooks.progress?.(0, 1, 'loading the body model …');
  const anny = await loadAnny(MODEL);
  // splat centres (the export is reused by the bind stage)
  if (!ctx.plyBlob) ctx.plyBlob = await ctx.session.exportPlyBlob();
  const centers = plyCenters(await ctx.plyBlob.arrayBuffer());
  const fit = await fitBody(anny, {
    landmarks: lm.landmarks, report: lm.report, splat: centers, log,
    progress: (m, f) => hooks.progress?.(f, 1, m),
  });
  let surface = fit.surface;
  // head on the face: correspondences from a render of the head through the frontal-most crop camera
  let head = null;
  const face = lm.face && lm.face.points && lm.face.points.length > 100 ? new Map(lm.face.ids.map((id, i) => [id, lm.face.points[i]])) : null;
  const cams = (lm.cropCams || []).filter((c) => c.facePx > 0);
  if (face && cams.length) {
    hooks.progress?.(0, 1, 'registering the head to the face …');
    try {
      const vision = await import(/* @vite-ignore */ `${VISION_CDN}/vision_bundle.mjs`);
      const files = await vision.FilesetResolver.forVisionTasks(`${VISION_CDN}/wasm`);
      const landmarker = await vision.FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: `${MODELS}face_landmarker.task`, delegate: 'CPU' }, runningMode: 'IMAGE', numFaces: 1, minFaceDetectionConfidence: 0.2, minFacePresenceConfidence: 0.2 });
      // head-bone weight per surface vertex (outer-skin order)
      const hb = anny.boneIndex.head;
      const headWeight = surface.boneIndices.map((bi, i) => bi.reduce((s, b, k) => s + (b === hb ? surface.boneWeights[i][k] : 0), 0));
      // the most frontal crop: the one whose face is widest
      const cam = cams.slice().sort((a, b) => b.facePx - a.facePx)[0];
      const { corr } = headCorrespondences(surface, headWeight, cam, landmarker);
      landmarker.close();
      log(`head: ${corr.size} correspondences on the rendered head (crop ${cam.name})`);
      if (corr.size > 200) { deformHead(surface, headWeight, corr, face, log, fit.scale); head = surface.headRegistered; }
    } catch (e) { log(`head registration skipped: ${e.message || e}`); }
  }
  // the rig fit from the fitted skeleton's markers (asymmetric: the body kept her stance)
  const glb = await (await fetch(RIG_GLB)).arrayBuffer();
  const rigFit = markersToFit(parseAvatar(glb), fit.markers, { asym: true });
  rigFit.source = { tool: 'avatar-mode body fit', framesUsed: manifest.stages.landmarks.framesUsed || null };
  ctx.surface = surface; ctx.fit = rigFit;
  const res = Object.values(fit.residualsM); const meanRes = res.reduce((a, b) => a + b, 0) / res.length;
  return { path: 'anny', scale: fit.scale, landmarkResidualCm: +(meanRes * 100).toFixed(2), head, note: `body model · ${(meanRes * 100).toFixed(1)} cm${head ? ` · face ${head.meanMm.toFixed(1)} mm` : ''}` };
}

/** The checkpoint: the fitted mesh over the face crop and over a body frame. */
/** The fitted surface drawn over two face crops and one body frame — shown
 *  while the run carries on, never waited for (2026-09-18). It replaces the
 *  joints picture on the card and stays until the run ends. */
export async function preview(ctx, manifest, hooks) {
  const S = ctx.surface; if (!hooks.shots || !S) return;
  const lm = manifest.stages.landmarks; const { session, frames } = ctx; const byName = new Map(frames.map((f) => [f.name, f]));
  const strip = document.createElement('div'); strip.className = 'av-overlay';
  // triangles spanning more than ~13 cm of depth are skipped (a poor man's back-face
  // cull) — in scene units through the fit scale, or on a 15 units/m scene it is 3 mm
  // and the face draws as holes (the user's 'leaky' overlay, 2026-09-14)
  const depthCut = 0.13 * (manifest.stages.bodyfit.scale || 1);
  const draw = async (img, cam, scale, crop) => {
    const cv = document.createElement('canvas'); const g = cv.getContext('2d');
    cv.width = img.width * scale; cv.height = img.height * scale; g.drawImage(img, 0, 0, cv.width, cv.height);
    g.strokeStyle = 'rgba(25,227,194,.55)'; g.lineWidth = 1;
    const P = S.vertices.map((v) => { const x = cam.R[0] * v[0] + cam.R[1] * v[1] + cam.R[2] * v[2] + cam.t[0], y = cam.R[3] * v[0] + cam.R[4] * v[1] + cam.R[5] * v[2] + cam.t[1], z = cam.R[6] * v[0] + cam.R[7] * v[1] + cam.R[8] * v[2] + cam.t[2]; return z > 0 ? [((cam.f * x / z + cam.cx) - (crop ? crop.x0 : 0)) * scale, (((cam.fy ?? cam.f) * y / z + cam.cy) - (crop ? crop.y0 : 0)) * scale, z] : null; });
    g.beginPath();
    for (const [a, b, c] of S.faces) { const pa = P[a], pb = P[b], pc = P[c]; if (!pa || !pb || !pc) continue; if (Math.abs(pa[2] - pb[2]) > depthCut || Math.abs(pa[2] - pc[2]) > depthCut) continue; g.moveTo(pa[0], pa[1]); g.lineTo(pb[0], pb[1]); g.lineTo(pc[0], pc[1]); g.lineTo(pa[0], pa[1]); }
    g.stroke(); cv.style.height = '240px'; cv.style.width = `${cv.width / cv.height * 240}px`; strip.appendChild(cv);
  };
  const cams = (lm.cropCams || []).slice().sort((a, b) => b.facePx - a.facePx).slice(0, 2);
  for (const cam of cams) {
    const entry = byName.get(cam.name); if (!entry) continue;
    const bmp = await createImageBitmap(entry.source); const c = new OffscreenCanvas(cam.side, cam.side); c.getContext('2d').drawImage(bmp, cam.x0, cam.y0, cam.side, cam.side, 0, 0, cam.side, cam.side); bmp.close();
    await draw(c, { ...cam }, 480 / cam.side, { x0: 0, y0: 0 });
  }
  const bodyCams = session.recon.cams.filter((c) => !c.crop); const bc = bodyCams[bodyCams.length >> 2]; const fr = session.frames[bc.imgIdx]; const entry = byName.get(fr.name);
  if (entry) { const bmp = await createImageBitmap(entry.source); const s = fr.fw / bmp.width; const c = new OffscreenCanvas(fr.fw, fr.fh); c.getContext('2d').drawImage(bmp, 0, 0, fr.fw, fr.fh); bmp.close(); await draw(c, bc, 480 / fr.fh, null); void s; }
  const st = manifest.stages.bodyfit;
  hooks.shots(strip, `The body model on the capture — landmarks ${st.landmarkResidualCm} cm${st.head ? `, face ${st.head.meanMm.toFixed(1)} mm` : ''}; this surface carries the skin weights and the lighting normals`);
}
