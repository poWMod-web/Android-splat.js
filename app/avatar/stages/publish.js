// publish.js — the final stage either hands an avatar back to an embedded
// rigger or offers a local package download. It does not sign in or upload.

export const id = 'publish';
export const needs = ['bind'];

const fmtMB = (b) => `${(b / 1e6).toFixed(b > 1e7 ? 0 : 1)} MB`;

/** the zip of everything, on #av-dl — the way out of both endings */
function wireDownload(body, { name, ply, bin, fit, glb, sog, thumb }) {
  const el = body.querySelector('#av-dl');
  if (!el) return;
  el.onclick = async () => {
    const { zipStore } = await import('../../js/zip.js');
    const files = [
      { name: `${name}.ply`, data: new Uint8Array(await ply.arrayBuffer()) },
      { name: `${name}_binding.bin`, data: bin },
      { name: `${name}_fit.json`, data: new TextEncoder().encode(JSON.stringify(fit, null, 1)) },
      { name: 'rig.glb', data: new Uint8Array(glb) },
    ];
    if (sog) files.push({ name: `${name}.sog`, data: new Uint8Array(await sog.arrayBuffer()) });
    if (thumb) files.push({ name: thumb.type === 'image/webp' ? 'thumb.webp' : 'thumb.png', data: new Uint8Array(await thumb.arrayBuffer()) });
    const zip = zipStore(files);
    const a = document.createElement('a'); a.href = URL.createObjectURL(zip); a.download = `${name}_avatar.zip`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  };
}

export async function run(ctx, manifest, hooks) {
  const body = hooks.body?.(); const log = hooks.log || (() => {});
  const ply = ctx.plyBlob, bin = ctx.bindingBytes, fit = ctx.fitJson, glb = ctx.rigGlb;
  if (!ply || !bin || !fit || !glb) throw new Error('nothing to publish (bind first)');
  const name = (manifest.source?.video || 'avatar').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 40) || 'avatar';
  let thumb = null;
  try { thumb = ctx.thumb ? await ctx.thumb() : null; } catch { thumb = null; }
  // the splat ships as SOG: the client streams a .sog directly and the driver
  // remaps the binding by splat centres (SOG reorders and requantises), so
  // the 20-80 MB PLY becomes a few MB with no runtime change. The PLY stays
  // in the package for the rigger and for re-fits.
  let sog = ctx.sogBlob || null;
  if (!sog && !ctx.handoff) {   // the rigger takes the PLY and compresses later
    try {
      const { plyToSog } = await import('../../js/sog.js');
      sog = await plyToSog(new Uint8Array(await ply.arrayBuffer()), { onProgress: ({ label, frac }) => hooks.progress?.(frac ?? 0, 1, `compressing the splat${label ? ' · ' + label : ''}`) });
      ctx.sogBlob = sog;
      log(`sog: ${fmtMB(ply.size)} ply -> ${fmtMB(sog.size)} sog`);
    } catch (e) { log(`sog failed (${e.message || e}) — shipping the PLY`); sog = null; }
  }
  // ── the hand-off: post the package to whoever opened this tool ────────────
  if (ctx.handoff) {
    const buf = async (b) => (b instanceof Blob ? await b.arrayBuffer() : b.buffer ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) : b);
    const assets = [
      { assetType: 'splat', name: `${name}.ply`, buffer: await buf(ply) },
      { assetType: 'binding', name: `${name}_binding.bin`, buffer: await buf(bin) },
      { assetType: 'fit', name: `${name}_fit.json`, buffer: await buf(new TextEncoder().encode(JSON.stringify(fit))) },
    ];
    if (thumb) assets.push({ assetType: 'thumbnail', name: `${name}.png`, buffer: await buf(thumb) });
    hooks.progress?.(0.5, 1, 'handing the avatar over …');
    await ctx.handoff({ name, assets });
    log(`handed over: ${assets.map((a) => a.assetType).join(', ')} (${fmtMB(assets.reduce((t, a) => t + a.buffer.byteLength, 0))})`);
    if (body) {
      body.innerHTML = `<div class="upcard-row"><p class="fine">Your avatar has gone back to the rigger — inspect it, give it its animations and put it on your account there.</p>
        <span style="display:flex;gap:8px"><button class="btn btn-outline" id="av-dl">Download package</button></span></div>
        <p class="fine" id="av-status"></p>`;
      wireDownload(body, { name, ply, bin, fit, glb, sog, thumb });
    }
    return { handedOff: true, format: 'ply', note: 'handed to the rigger' };
  }

  // Standalone builds no longer sign in or publish to an account. The
  // completed avatar remains fully exportable as a local package.
  if (body) {
    body.innerHTML = `<div class="upcard-row"><p class="fine">Avatar package ready · splat ${sog ? fmtMB(sog.size) : fmtMB(ply.size)} · binding ${fmtMB(bin.byteLength)} · rig ${fmtMB(glb.byteLength)}.</p>
      <button class="btn btn-accent" id="av-dl">Download package</button></div>
      <p class="fine" id="av-status">Everything stays on this device.</p>`;
    wireDownload(body, { name, ply, bin, fit, glb, sog, thumb });
  }
  log('avatar package ready for local download');
  return { format: sog ? 'sog' : 'ply', note: 'local download package' };
}
