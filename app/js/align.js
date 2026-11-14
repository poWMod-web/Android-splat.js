// Selected-frame pose repair. Pair geometry contributes rotation only (the
// essential matrix has no translation scale); existing neighbouring world
// centres anchor translation. Callers validate and offer candidates before apply.
import { detectAndDescribeMS } from '../../src/sfm/features.js';
import { matchDescriptors } from '../../src/sfm/matching.js';
import { ransacE, decomposeE, selectPose, makeRng, m3mul, m3t } from '../../src/sfm/geometry.js';

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
