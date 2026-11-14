// Selected-frame pose repair. Pair geometry contributes rotation only (the
// essential matrix has no translation scale); existing neighbouring world
// centres anchor translation. Callers must validate/confirm before applying.
import { detectAndDescribeMS } from '../../src/sfm/features.js';
import { matchDescriptors } from '../../src/sfm/matching.js';
import { ransacE, decomposeE, selectPose, makeRng, m3mul, m3t } from '../../src/sfm/geometry.js';

async function gray(url, maxDim = 900) {
  const b = await createImageBitmap(await (await fetch(url)).blob());
  const k = Math.min(1, maxDim / Math.max(b.width, b.height));
  const w = Math.max(2, Math.round(b.width * k)), h = Math.max(2, Math.round(b.height * k));
  const cv = new OffscreenCanvas(w, h), x = cv.getContext('2d', { willReadFrequently: true });
  x.drawImage(b, 0, 0, w, h); b.close?.();
  const d = x.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] * .299 + d[i * 4 + 1] * .587 + d[i * 4 + 2] * .114) / 255;
  return { g, w, h };
}

async function relative(a, b, ca, cb, seed) {
  const [ia, ib] = await Promise.all([gray(a), gray(b)]);
  const [fa, fb] = [detectAndDescribeMS(ia.g, ia.w, ia.h, 1800), detectAndDescribeMS(ib.g, ib.w, ib.h, 1800)];
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
  return p && p.goodFrac >= .6 ? { R: p.R, inliers: e.inliers.length, matches: x1.length, angle: p.medianAngle } : null;
}

/** Rotation candidates anchored independently from previous and next poses. */
export async function adjacentPoseCandidates(photos, cams, i) {
  const cur = cams[i], out = [];
  if (!cur || !cur.f) return out;
  if (i > 0 && cams[i - 1]?.R) {
    const r = await relative(photos[i - 1].url, photos[i].url, cams[i - 1], cur, 1701 + i);
    if (r) out.push({ ...r, Rworld: m3mul(r.R, cams[i - 1].R), side: 'previous' });
  }
  if (i + 1 < cams.length && cams[i + 1]?.R) {
    const r = await relative(photos[i].url, photos[i + 1].url, cur, cams[i + 1], 2903 + i);
    if (r) out.push({ ...r, Rworld: m3mul(m3t(r.R), cams[i + 1].R), side: 'next' });
  }
  return out;
}
