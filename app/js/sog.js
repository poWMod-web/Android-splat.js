// sog.js — in-browser SOG compression via @playcanvas/splat-transform
// (vendored ESM bundle in app/vendor: bundle + its webp wasm + its encode
// worker; ~5MB, loaded on demand the first time an export needs it).
//
// SOG is the web-delivery format: Morton-ordered, quantized, k-means SH
// palettes, WebP-packed — typically 10-20x smaller than the .ply.

let stP = null;
export const loadST = () => stP ??= import('../vendor/splat-transform.bundle.mjs').then((st) => {
  st.WorkerQueue.workerUrl = new URL('../vendor/st-worker.mjs', import.meta.url).href;
  // the workers' own fallback resolves ../lib/webp.wasm (package layout);
  // point everyone at the vendored copy instead
  st.WebPCodec.wasmUrl = new URL('../vendor/webp.wasm', import.meta.url).href;
  return st;
});

/** Compressor GPU device with a deadline. The bundle awaits createDevice with
 *  no guard of its own, and a wedged GPU process (seen after multi-hour
 *  training runs) leaves that await pending FOREVER — the visitor stares at a
 *  frozen "Compressing" card for half an hour. Fail loud instead; the caller
 *  turns it into an honest message and the model stays exportable as .ply. */
function makeStDevice(st, keep) {
  const cv = document.createElement('canvas');
  return Promise.race([
    st.createGraphicsDevice(cv, { deviceTypes: ['webgpu'] }).then((d) => { keep(d); return d; }),
    new Promise((_, rej) => setTimeout(() => rej(new Error(
      'the GPU compressor did not answer. Restarting the browser usually clears this — ' +
      'the model is kept on this device under Yours')), 15000)),
  ]);
}

/**
 * PLY bytes -> bundled .sog Blob.
 * onProgress({ label, frac }): the writer's own stage names (gather, cluster,
 * encode …); frac is 0..1 within the current stage, or null for unbarred
 * stages.
 */
export async function plyToSog(plyBytes, { iterations = 10, onProgress = () => {} } = {}) {
  const st = await loadST();
  st.logger.setRenderer({
    handle(e) {
      if (e.kind === 'barStart') onProgress({ label: e.name, frac: 0 });
      else if (e.kind === 'barTick') onProgress({ label: e.name, frac: e.total ? e.current / e.total : 0 });
      else if (e.kind === 'barEnd') onProgress({ label: e.name, frac: 1 });
      else if (e.kind === 'scopeStart') onProgress({ label: e.name, frac: null });
    },
  });
  const rfs = new st.MemoryReadFileSystem();
  rfs.set('model.ply', plyBytes instanceof Uint8Array ? plyBytes : new Uint8Array(plyBytes));
  const [source] = await st.readFile({ filename: 'model.ply', inputFormat: 'ply', fileSystem: rfs });
  const pool = st.createChunkDataPool();
  const out = new st.MemoryFileSystem();
  // GPU device for the SH clustering — the CPU fallback pins the main thread
  // for MINUTES on big models (measured: 566k splats did not finish in 8)
  let device = null;
  const createDevice = () => makeStDevice(st, (d) => { device = d; });
  try {
    await st.writeSource(
      { filename: 'model.sog', outputFormat: 'sog-bundle', source, pool, options: { iterations }, createDevice }, out);
  } finally {
    source.close?.();
    device?.destroy?.();
  }
  const bytes = out.results.get('model.sog');
  if (!bytes) throw new Error('SOG writer produced no output');
  return new Blob([bytes], { type: 'application/octet-stream' });
}

/**
 * Trained detail levels -> a Streamed SOG (multi-LOD, spatially chunked).
 * plyLevels: Uint8Array[] with the FINEST level first (LOD 0), then coarser.
 * Returns [name, Uint8Array] entries of the streamed-SOG file set (lod-meta
 * + chunk files) — zip them for a download.
 */
export async function plysToLodEntries(plyLevels, { iterations = 10, onProgress = () => {} } = {}) {
  const st = await loadST();
  st.logger.setRenderer({
    handle(e) {
      if (e.kind === 'barTick') onProgress({ label: e.name, frac: e.total ? e.current / e.total : 0 });
      else if (e.kind === 'scopeStart') onProgress({ label: e.name, frac: null });
    },
  });
  const rfs = new st.MemoryReadFileSystem();
  const sources = [];
  for (let i = 0; i < plyLevels.length; i++) {
    rfs.set(`l${i}.ply`, plyLevels[i]);
    const [src] = await st.readFile({ filename: `l${i}.ply`, inputFormat: 'ply', fileSystem: rfs });
    sources.push(src);
  }
  const main = st.stackLods(sources);
  const out = new st.MemoryFileSystem();
  let device = null;
  const createDevice = () => makeStDevice(st, (d) => { device = d; });
  try {
    await st.writeLodSource({
      filename: 'lod-meta.json', mainSource: main, envSource: null,
      iterations, createDevice, chunkCount: 512, chunkExtent: 16,
    }, out);
  } finally {
    sources.forEach((s2) => s2.close?.());
    device?.destroy?.();
  }
  return [...out.results.entries()];
}
