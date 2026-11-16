// bind.js — the SBA1 sidecar, in the tab (export-binding-core, the same code
// the rigger's CLI runs). Splat centres come from the session's own export;
// the fit from the landmarks stage; the surface from the body fit when there
// is one, the rig mesh otherwise.
import { buildBinding, plyCenters } from '../rig/export-binding-core.js';

export const id = 'bind';
export const needs = ['landmarks'];
const RIG_GLB = new URL('../rig/glb_avatar.glb', import.meta.url).href;

export async function run(ctx, manifest, hooks) {
  const log = hooks.log || (() => {});
  hooks.progress?.(0, 3, 'exporting the splat …');
  const ply = await ctx.session.exportPlyBlob();
  ctx.plyBlob = ply;
  hooks.progress?.(1, 3, 'binding to the rig …');
  const centers = plyCenters(await ply.arrayBuffer());
  const fit = ctx.fit || manifest.stages.landmarks.fit;
  const avatarGlb = await (await fetch(RIG_GLB)).arrayBuffer();
  const { bytes, stats } = buildBinding({ avatarGlb, centers, fit, surface: ctx.surface || null, log });
  ctx.bindingBytes = bytes; ctx.fitJson = fit; ctx.rigGlb = avatarGlb;
  // leakage ruler: for each leg's splats (by dominant joint), the weight carried by the other leg
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + new DataView(bytes.buffer).getUint32(4, true))).replace(/\0+$/, ''));
  const n = meta.numSplats, nb = meta.numBones, off = 8 + new DataView(bytes.buffer).getUint32(4, true) + nb * 64;
  const idx = bytes.subarray(off, off + n * 4), wgt = bytes.subarray(off + n * 4, off + n * 8);
  const J = meta.jointNames; const left = new Set(['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'].map((j) => J.indexOf(j)));
  const right = new Set(['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'].map((j) => J.indexOf(j)));
  let lTot = 0, lLeak = 0, rTot = 0, rLeak = 0;
  for (let i = 0; i < n; i++) {
    const dom = idx[i * 4]; const inL = left.has(dom), inR = right.has(dom); if (!inL && !inR) continue;
    for (let k = 0; k < 4; k++) { const w = wgt[i * 4 + k] / 255; if (inL) { lTot += w; if (right.has(idx[i * 4 + k])) lLeak += w; } else { rTot += w; if (left.has(idx[i * 4 + k])) rLeak += w; } }
  }
  const leakage = [lTot ? lLeak / lTot : 0, rTot ? rLeak / rTot : 0];
  hooks.progress?.(3, 3, 'bound');
  return { ...stats, leakage, note: `${stats.meanNearCm} cm · leak ${(Math.max(...leakage) * 100).toFixed(1)}%` };
}
