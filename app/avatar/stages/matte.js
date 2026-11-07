// matte.js — the person's alpha for every picked frame, in the browser.
//
// RobustVideoMatting (MobileNetV3, ONNX) through onnxruntime-web: recurrent,
// so the frames go through in capture order and the four state tensors are
// carried from one to the next. The source is fed at a bounded size (long
// side MATTE_MAX) with RVM's own downsample_ratio; the alpha comes back at
// that size and the trainer resamples it to the training grid (an alpha at
// 1280 px against a 1600 px target is plenty — the loss's guard ring and the
// random background forgive the last pixel).
//
// Output per frame: a grayscale PNG (white = subject), attached to the
// capture file entry as `mask`, which is exactly what the session's per-file
// mask input takes.

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MODEL_URL = new URL('../../models/rvm_mobilenetv3_fp32.onnx', import.meta.url).href;
const MATTE_MAX = 1280;      // long side fed to the net
const INTERNAL = 512;        // RVM's internal long side (its downsample_ratio is derived per frame)

let ortPromise = null;
function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ `${ORT_CDN}ort.webgpu.min.mjs`).then((m) => {
      const ort = m.default || m;
      ort.env.wasm.wasmPaths = ORT_CDN;
      return ort;
    });
  }
  return ortPromise;
}

async function createSession(ort, log) {
  const providers = ['webgpu', 'wasm'];
  for (const ep of providers) {
    try {
      const s = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      log(`matte: onnxruntime ${ep}`);
      return s;
    } catch (e) {
      log(`matte: ${ep} unavailable (${e.message || e})`);
    }
  }
  throw new Error('no onnxruntime execution provider');
}

/** @param {Array<{source: Blob, name: string}>} frames  capture order
 *  @param {{onProgress?: (done:number,total:number,frame?:object)=>void, log?: (m:string)=>void, signal?: AbortSignal}} o
 *  @returns {Promise<{coverage: number, model: string}>}  and each frame gets .mask (PNG Blob) */
export async function run(frames, o = {}) {
  const log = o.log || (() => {});
  const onProgress = o.onProgress || (() => {});
  const ort = await loadOrt();
  let sess = await createSession(ort, log);
  let maxSide = MATTE_MAX;
  const zero = () => new ort.Tensor('float32', new Float32Array(1), [1, 1, 1, 1]);
  let rec = [zero(), zero(), zero(), zero()];
  const cover = [];
  let cv = null, ctx = null, mcv = null, mctx = null;
  const tStart = performance.now();
  // the WebGPU provider may accept the model and still refuse an op at run
  // time (onnxruntime-web 1.20: AveragePool with ceil_mode); the first frame
  // is the probe — on failure the run continues on wasm at a smaller size
  const runOrFallback = async (feeds) => {
    try { return await sess.run(feeds); } catch (e) {
      if (sess._fellBack) throw e;
      log(`matte: ${e.message || e} — falling back to wasm`);
      try { sess.release?.(); } catch { /* fine */ }
      sess = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      sess._fellBack = true;
      throw new Error('__retry');
    }
  };
  for (let i = 0; i < frames.length; i++) {
    if (o.signal && o.signal.aborted) throw new Error('cancelled');
    const f = frames[i];
    const bmp = await createImageBitmap(f.source);
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const W = Math.round(bmp.width * s), H = Math.round(bmp.height * s);
    if (!cv || cv.width !== W || cv.height !== H) {
      cv = new OffscreenCanvas(W, H); ctx = cv.getContext('2d', { willReadFrequently: true });
      mcv = new OffscreenCanvas(W, H); mctx = mcv.getContext('2d');
    }
    ctx.drawImage(bmp, 0, 0, W, H); bmp.close();
    const px = ctx.getImageData(0, 0, W, H).data;
    const src = new Float32Array(3 * W * H);
    const n = W * H;
    for (let p = 0; p < n; p++) {
      src[p] = px[p * 4] / 255; src[n + p] = px[p * 4 + 1] / 255; src[2 * n + p] = px[p * 4 + 2] / 255;
    }
    let out;
    const ratio = new ort.Tensor('float32', new Float32Array([Math.min(1, INTERNAL / Math.max(W, H))]), [1]);
    try {
      out = await runOrFallback({
        src: new ort.Tensor('float32', src, [1, 3, H, W]),
        r1i: rec[0], r2i: rec[1], r3i: rec[2], r4i: rec[3], downsample_ratio: ratio,
      });
    } catch (e) {
      if (e.message !== '__retry') throw e;
      rec = [zero(), zero(), zero(), zero()]; cv = null;   // state tensors belonged to the old provider
      i--; continue;                                        // same frame again, on wasm
    }
    rec = [out.r1o, out.r2o, out.r3o, out.r4o];
    if (i % 20 === 19) log(`matte: ${i + 1}/${frames.length} · ${((performance.now() - tStart) / (i + 1) / 1000).toFixed(2)} s/frame at ${W}×${H}`);
    const pha = out.pha.data;   // [1,1,H,W]
    const img = mctx.createImageData(W, H);
    let hard = 0;
    for (let p = 0; p < n; p++) {
      const a = Math.max(0, Math.min(255, Math.round(pha[p] * 255)));
      img.data[p * 4] = a; img.data[p * 4 + 1] = a; img.data[p * 4 + 2] = a; img.data[p * 4 + 3] = 255;
      if (a > 230) hard++;
    }
    mctx.putImageData(img, 0, 0);
    f.mask = await mcv.convertToBlob({ type: 'image/png' });
    cover.push(hard / n);
    onProgress(i + 1, frames.length, f);   // the frame just cut, for a live preview
  }
  try { sess.release?.(); } catch { /* fine */ }
  cover.sort((a, b) => a - b);
  const coverage = cover.length ? cover[cover.length >> 1] : 0;
  log(`matte: ${frames.length} frames, subject covers ${(coverage * 100).toFixed(1)}% (median)`);
  return { coverage, model: 'rvm-mobilenetv3' };
}
