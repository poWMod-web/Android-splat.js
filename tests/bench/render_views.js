// Headless novel-view renderer for bench models: loads a posted .ply + its
// recon JSON, renders the trained model from chosen TRAINING camera poses, and
// posts the PNGs back to /scratch. Same pose as a real photograph, so a render
// can be put straight beside the frame it is supposed to reproduce.
//   ?ply=lisa_30000.ply&recon=lisa_30000_recon.json&views=0,40,80&w=540&out=shot
// ?bg=1,1,1 renders on white; white minus black gives exact per-pixel coverage.
import { createSession } from '/src/index.js';
import { parsePlyGaussians } from '/app/js/session_io.js';

const Q = new URLSearchParams(location.search);
const logEl = document.getElementById('log');
const say = (m) => { logEl.textContent += m + '\n'; console.log('[RENDER]', m); };
const post = (name, body) => fetch(`/scratch/${name}`, { method: 'POST', body });

try {
  const recon = await (await fetch(`/scratch/${Q.get('recon')}`)).json();
  const plyBytes = new Uint8Array(await (await fetch(`/scratch/${Q.get('ply')}`)).arrayBuffer());
  const g = parsePlyGaussians(plyBytes);
  say(`${g.n} splats, ${recon.cams.length} cams, radius ${recon.sceneRadius}`);

  // ?dilate=0.3&mipcomp=0 emulates a classic viewer (PlayCanvas: +0.3 px dilation, no compensation)
  const ses = createSession({ trainer: { ...(Q.get('dilate') ? { dilate: +Q.get('dilate') } : {}), ...(Q.get('mipcomp') === '0' ? { mipComp: false } : {}) } });
  ses.useReconstruction(recon);                // cameras for the opacity unbake
  await ses.seedFrom(g, { viewOnly: true, sceneRadius: recon.sceneRadius, unbake: !Q.get('baked') });   // render what the trainer rendered (?baked=1: the export as external viewers show it)

  const W = +(Q.get('w') || 540);
  const cv = document.getElementById('cv');
  const views = (Q.get('views') || '0').split(',').map(Number);
  const byIdx = new Map(recon.cams.map((c) => [c.imgIdx, c]));
  const frames = recon.frames;

  for (const vi of views) {
    const c = byIdx.get(vi) || recon.cams[vi % recon.cams.length];
    const fr = frames[c.imgIdx];
    const s = W / fr.fw;                       // recon f is at FEATURE scale
    const H = Math.round(fr.fh * s);
    cv.width = W; cv.height = H;
    ses.view.attach(cv);
    const bg = Q.get('bg') ? Q.get('bg').split(',').map(Number) : null;
    ses.view.setCamera({ R: c.R, t: c.t, f: c.f * s, fy: (c.fy ?? c.f) * s,
                         cx: (c.cx ?? fr.fw / 2) * s, cy: (c.cy ?? fr.fh / 2) * s, w: W, h: H, ...(bg ? { bg } : {}) });
    ses.view.renderNow();
    await new Promise((r) => requestAnimationFrame(r));
    ses.view.renderNow();                       // second pass: sorted, settled
    const blob = await new Promise((r) => cv.toBlob(r, 'image/png'));
    await post(`${Q.get('out')}_${String(vi).padStart(3, '0')}.png`, blob);
    say(`view ${vi} (${fr.name}) ${W}x${H} -> ${(blob.size / 1024).toFixed(0)}kB`);
  }
  await post(`${Q.get('out')}_done.json`, JSON.stringify({ ok: true, views, n: g.n }));
  say('DONE');
} catch (e) {
  say('ERROR ' + e.message);
  await post(`${Q.get('out')}_done.json`, JSON.stringify({ error: e.message }));
}
