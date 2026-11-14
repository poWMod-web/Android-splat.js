// session.js — layer 2: the one object a UI talks to.
//
//   const s = createSession();
//   s.on('stage',   e => ...)   // { stage, done, total, detail }
//   s.on('metrics', e => ...)   // { iter, splats, itersPerSec, psnrTrain, psnrHold }
//   s.on('event',   e => ...)   // { kind: 'refine' | 'train-complete' | 'device-lost' |
//                                 'worker-error' | 'train-error' | 'view-error' |
//                                 'train-stall', ... }
//   (a no-present view-error also carries diagnosis: 'presentation' |
//   'raster' | 'capture-failed' | 'unknown' — which link the probe blames)
//   s.on('log',     m => ...)   // prose, for consoles
//
//   await s.load(files);        // File[] / Blob[] -> Frames
//   await s.solve();            // SfM: cameras + sparse points
//   await s.seed();             // Gaussians + WebGPU trainer
//   s.start(); s.pause();       // the training loop (policy lives HERE)
//
//   s.view.attach(canvas);      // render target
//   s.view.lookThrough(i);      // camera = training frame i
//   s.view.setCamera(pose);     // free camera
//   await s.exportPlyBlob();    // standard 3DGS .ply (opacity comp baked)
//
// All numeric policy that used to live in the demo page is here: iteration
// batching, auto-stop, refine cadence, blur exclusion, holdout choice, PSNR
// conversion, and the hidden-tab watchdog.

import { FastRobustZLEMA, SKIP_FIRST } from './zlema.js';
import { decodeFrames } from './io/frames.js';
import { reconFromKnownCameras } from './io/cameras.js';
import { focalPxFrom35 } from './io/exif.js';
import { isEquirect, sliceEquirect, faceSizeFor, probeImageSize, FACE_ROTS } from './io/pano.js';
import { runSfM } from './sfm/sfm.js';
import { initGaussians } from './gs/init.js';
import { buildVisualHull, makeHullTest, makeSplatTest } from './gs/hull.js';
import { GSTrainer } from './gs/trainer.js';
import { RemoteTrainer } from './gs/remote-trainer.js';
import { createGpu } from './gpu/context.js';
import { gaussiansToPly, bakeOpacityCompensation, unbakeOpacityCompensation } from './io/ply.js';
import { simplifyGaussians } from './gs/simplify.js';

/** One macrotask breath — lets a pending frame present go out before a GPU
 *  readback drains the queue behind it. */
const tick = () => new Promise((r) => setTimeout(r, 0));

/** world-space position of a camera pose {R, t} */
export function camPosition({ R, t }) {
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

/** Undistort training images in place (bilinear resample) so the pinhole
 *  trainer matches BA's distortion-corrected geometry. Returns true when a
 *  resample actually happened (|k| above the noise floor). Out-of-frame
 *  samples get a -1 sentinel the trainer excludes from the loss. */
export function undistortFrames(frames, recon) {
  const { k1, k2 } = recon;
  // below ~0.01 the estimated distortion is noise (synthetic GT k1=0 comes
  // back as ~0.005) and resampling would only soften the targets
  if (Math.abs(k1) < 0.01 && Math.abs(k2) < 0.01) return false;
  for (const im of frames) {
    const f = recon.fFeat * (im.tw / im.fw);
    const cx = im.tw / 2, cy = im.th / 2;
    const src = im.rgb;
    const dst = new Float32Array(src.length);
    for (let y = 0; y < im.th; y++) {
      for (let x = 0; x < im.tw; x++) {
        const xp = (x + 0.5 - cx) / f, yp = (y + 0.5 - cy) / f;
        const r2 = xp * xp + yp * yp;
        const D = 1 + k1 * r2 + k2 * r2 * r2;
        const rx = f * xp * D + cx - 0.5;
        const ry = f * yp * D + cy - 0.5;
        const o = (y * im.tw + x) * 3;
        if (rx < 0 || ry < 0 || rx > im.tw - 1.001 || ry > im.th - 1.001) {
          dst[o] = -1; dst[o + 1] = -1; dst[o + 2] = -1;
          continue;
        }
        const x0 = rx | 0, y0 = ry | 0;
        const fx = rx - x0, fy = ry - y0;
        const i00 = (y0 * im.tw + x0) * 3, i01 = i00 + 3;
        const i10 = i00 + im.tw * 3, i11 = i10 + 3;
        for (let c = 0; c < 3; c++) {
          dst[o + c] =
            src[i00 + c] * (1 - fx) * (1 - fy) + src[i01 + c] * fx * (1 - fy) +
            src[i10 + c] * (1 - fx) * fy + src[i11 + c] * fx * fy;
        }
      }
    }
    if (im.alpha) {
      // the subject alpha rides along (nearest), or it drifts off the photo
      const adst = new Uint8Array(im.alpha.length);
      for (let y = 0; y < im.th; y++) {
        for (let x = 0; x < im.tw; x++) {
          const xp = (x + 0.5 - cx) / f, yp = (y + 0.5 - cy) / f;
          const r2 = xp * xp + yp * yp;
          const D = 1 + k1 * r2 + k2 * r2 * r2;
          const rx = Math.round(f * xp * D + cx - 0.5), ry = Math.round(f * yp * D + cy - 0.5);
          adst[y * im.tw + x] = (rx < 0 || ry < 0 || rx >= im.tw || ry >= im.th) ? 0 : im.alpha[ry * im.tw + rx];
        }
      }
      im.alpha = adst;
    }
    im.rgb = dst;
  }
  return true;
}

/**
 * @typedef {object} SessionOptions
 * @property {GPUDevice} [device]        host-owned WebGPU device to share
 * @property {number} [maxIters=60000]   auto-stop; trainer schedules scale to it
 * @property {number} [itersPerFrame=15] training batch per animation frame
 * @property {number} [gpuUsage=1]  GPU-usage fraction (the slider): full
 *   batches with duty-cycle idle between them — 0.5 trains half the wall
 *   time; 1 (or unset) is full speed
 * @property {boolean} [autoDuty=true]  duty governor: derate the effective
 *   duty below the slider when rAF / event-loop health slips or the GPU's
 *   own step cost sags, recover when healthy. The slider stays the thermal
 *   ceiling ("how hot"); false pins the loop to the slider exactly. Frozen
 *   while the tab is hidden.
 * @property {object} [governor]  governor thresholds
 *   ({ lagJank, lagOk, rafJank, rafOk, gpuJank, gpuOk, down, up, floor };
 *   see GOV_DEFAULTS)
 * @property {number} [initTarget=60000] initial Gaussian count (grows during training)
 * @property {number|'auto'|null} [holdout='auto']  frame excluded from training
 *   and scored as the novel-view metric; 'auto' picks a sharp mid-sequence frame
 * @property {number} [evalHoldEvery=4000]  holdout PSNR cadence (iterations)
 * @property {number} [evalSplit=0]  the standard benchmark protocol: every Nth
 *   frame of the input order is excluded from training and scored together via
 *   evalTestPsnr() (papers use N=8); 0 = off
 * @property {string[]} [evalFrames]  explicit test set by frame name (overrides evalSplit;
 *   a fixed ruler when the frame set itself is what varies, e.g. video extraction variants)
 * @property {object} [sfm]              SfmOptions passed to solve()
 * @property {object} [trainer]          extra GSTrainer options (shDeg, ...)
 * @property {object} [frames]           FrameOptions passed to load()
 * @property {'auto'|boolean} [trainWorker='auto']  run the GPU training core
 *   in a dedicated worker (encoding, submits and readbacks leave the main
 *   thread responsive); 'auto' falls back to main-thread training when
 *   workers or WebGPU-in-workers are unavailable, false forces local, true
 *   demands the worker. A host-owned `device` always forces local: devices
 *   cannot cross threads.
 * @property {() => Worker} [createTrainWorker]  inject the training worker
 *   (tests; default spawns src/gs/trainworker.js as a module worker)
 * @property {number} [gpuChunkMs=400]  the phone budget (ms of GPU work):
 *   bounds one batch's queued work AND the total outstanding across the
 *   fence ring — over it, the loop cuts submits until the queue drains
 */

/** Duty cycle of the GPU-usage throttle: the fraction of wall time the
 *  training loop runs. Unset/garbage = 1 (full speed, legacy behavior);
 *  0 or below = a crawl (one full batch per idle window, still moving). */
export function gpuDuty(usage) {
  if (usage == null) return 1;
  const u = Number(usage);
  if (!Number.isFinite(u) || u >= 1) return 1;
  if (u <= 0) return 0.001;
  return u;
}

/** Idle ms after a busyMs batch so busy/(busy+idle) equals the requested
 *  GPU usage. Capped: one slow frame (refine, first batch) never parks
 *  the loop for longer than the cap window. */
export function gpuIdleMs(busyMs, usage) {
  const d = gpuDuty(usage);
  if (d >= 1) return 0;
  return Math.min(2000, Math.max(0, busyMs) * (1 - d) / d);
}

/** Burst bound for one batch's GPU work: the chunk cap scaled by duty. A
 *  full-size burst at low duty would run through the whole "idle" window
 *  as backlog and the system would never get the GPU; at full duty this
 *  is exactly the old bound (provable no-op). */
export function gpuBurstMs(chunkMs, usage) {
  return chunkMs * gpuDuty(usage);
}

/** Whether the page is hidden (no document, e.g. Node/tests, counts as
 *  visible). Plan item (7) gates everything the user can't see on this. */
function isHidden() {
  return typeof document !== 'undefined' && !!document.hidden;
}

/** Lit-pixel sample of an RGBA readback: { lit, n, frac }. Up to ~2k
 *  pixels sampled; lit = any channel above the near-black floor. Shared
 *  by the no-present probe (canvas-free capture) and the app's
 *  present-health watchdog (placeholder-canvas readback) so "renders
 *  content" means the same thing in both. */
export function litSample(rgba, { threshold = 12, maxSamples = 2048 } = {}) {
  if (!rgba || rgba.length < 4) return { lit: 0, n: 0, frac: 0 };
  const step = Math.max(4, Math.floor(rgba.length / 4 / maxSamples) * 4);
  let lit = 0, n = 0;
  for (let o = 0; o < rgba.length; o += step) {
    n++;
    if (rgba[o] > threshold || rgba[o + 1] > threshold || rgba[o + 2] > threshold) lit++;
  }
  return { lit, n, frac: n ? lit / n : 0 };
}

/** Duty-governor thresholds: jank above which to derate, health below which
 *  to recover, the multiplicative down-step, additive up-step, and floor. */
export const GOV_DEFAULTS = {
  lagJank: 50, lagOk: 20, rafJank: 34, rafOk: 20,
  gpuJank: 1.25, gpuOk: 1.1, down: 0.7, up: 0.1, floor: 0.1,
};

/** One governor step (pure, for tests): derate fast on jank, recover slow
 *  on health, hysteresis between the thresholds. Returns { auto, moved }
 *  with moved -1/0/+1 for derated/held/recovered. */
export function governDuty(auto, { lagMs = 0, rafMs = 16.7, gpuRatio = 1 } = {}, opts = {}) {
  const g = { ...GOV_DEFAULTS, ...opts };
  if (lagMs > g.lagJank || rafMs > g.rafJank || gpuRatio > g.gpuJank) {
    return { auto: Math.max(g.floor, auto * g.down), moved: -1 };
  }
  if (lagMs < g.lagOk && rafMs < g.rafOk && gpuRatio < g.gpuOk && auto < 1) {
    return { auto: Math.min(1, auto + g.up), moved: 1 };
  }
  return { auto, moved: 0 };
}

class Emitter {
  constructor() { this.map = new Map(); }
  on(type, fn) {
    if (!this.map.has(type)) this.map.set(type, new Set());
    this.map.get(type).add(fn);
    return () => this.map.get(type).delete(fn);
  }
  emit(type, e) {
    const s = this.map.get(type);
    if (s) for (const fn of s) fn(e);
  }
}

/** Robust plateau test for score-driven refinement. A capacity-starved model can
 * oscillate around a flat ceiling, so raw variance is a POSITIVE saturation
 * signal rather than a veto. The filtered curve catches ordinary stalls; a
 * rising raw MAD catches the observed noisy-under-capacity plateau before the
 * filter has become perfectly flat. Recovery is gated separately below. */
export function scorePlateaued(history, nowIter, {
  window = 1200, slope = 0.1, noisySlope = 0.35, minPoints = 4,
  after = -Infinity, rawHistory = null, noiseRatio = 1.35, noiseFloor = 0.08,
} = {}) {
  if (!Array.isArray(history) || !history.length) return false;
  const floor = Math.max(nowIter - window, after);
  const take = (rows) => (Array.isArray(rows) ? rows : [])
    .filter(([i, p]) => Number.isFinite(i) && Number.isFinite(p) && i >= floor && i <= nowIter)
    .sort((a, b) => a[0] - b[0]);
  const pts = take(history);
  if (pts.length < minPoints || pts.at(-1)[0] - pts[0][0] < window * 0.75) return false;
  const medianValues = (a) => { const v = a.map((p) => p[1]).sort((x, y) => x - y); return v[v.length >> 1]; };
  const q = Math.max(2, Math.floor(pts.length / 3));
  const early = pts.slice(0, q), late = pts.slice(-q);
  const dt = late.reduce((n, p) => n + p[0], 0) / q - early.reduce((n, p) => n + p[0], 0) / q;
  if (!(dt > 0)) return false;
  const robustSlope = (medianValues(late) - medianValues(early)) * 1000 / dt;
  if (robustSlope < slope) return true;

  const raw = take(rawHistory);
  if (raw.length < 6 || raw.at(-1)[0] - raw[0][0] < window * 0.75) return false;
  const mid = Math.floor(raw.length / 2);
  const mad = (rows) => {
    const m = medianValues(rows);
    return medianValues(rows.map(([i, p]) => [i, Math.abs(p - m)]));
  };
  const earlyNoise = mad(raw.slice(0, mid)), lateNoise = mad(raw.slice(mid));
  const noiseRising = lateNoise >= noiseFloor &&
    lateNoise >= Math.max(earlyNoise * noiseRatio, earlyNoise + 0.03);
  return noiseRising && robustSlope < noisySlope;
}



export class Session {
  /** @param {SessionOptions} [opts] */
  constructor(opts = {}) {
    this.opts = opts;
    this.frames = [];
    this.recon = null;      // solve() result: { cams, points, k1, k2, ... }
    this.model = null;      // seed() result: { data, n, center, radius }
    this.trainer = null;
    this.gpu = null;
    this._worker = null;    // the training worker (remote mode; null = local)
    this._remote = false;   // whether this.trainer is a RemoteTrainer proxy
    this._workerGen = 0;    // bumped per remote boot; canvas transfers die with the worker
    this.holdout = -1;
    this.testCams = [];
    this.training = false;
    this.lossHistory = [];  // [iter, psnrTrain]
    // the graph's curve, kept incrementally: same filter, same parameters,
    // same cold-start hold as the score plot (zlema.js) — the plateau slope
    // below reads THIS, so the trigger trips on the line the user sees
    this.zlemaHistory = [];   // [iter, filteredPsnrTrain]
    this._zlema = new FastRobustZLEMA();
    // Intervention cycle: plateau -> expected score drop -> recovery to the
    // pre-intervention reference -> a fresh plateau. This blocks repeated
    // growth while the model is still rebuilding quality.
    this._refineRecoveryTarget = null;
    this._fences = [];      // in-flight batch fences (see _frameLoop)
    this._govAuto = 1;      // duty governor: auto duty, min()'d with the slider
    this._govLag = 0;       // event-loop lag EWMA (ms)
    this._govRaf = 16.7;    // rAF inter-arrival EWMA (ms)
    this._govMsStep = 0;    // GPU ms/step EWMA (item 3: the ceiling's own signal)
    this._govMsBase = 0;    // best smoothed ms/step at this splat count (0 = unanchored)
    this._govN = -1;        // splat count the baseline is anchored to
    this._gpuMsMisses = 0;  // consecutive batches without a timer reading
    this._cutMs = 0;        // (4) estimated outstanding GPU work (ms)
    this._cutT = null;      // wall time the estimate is current as of
    this._cutPred = 0;      // last submit's predicted ms (corrected by measure)
    this._cutPending = false; // last submit uncorrected by its gpuMs yet
    this._lastGpuMs = null; // most recent batch's measured GPU-busy ms
    this._govRafLast = 0;
    this._govRafRunning = false;
    this._govBucket = 1;    // last logged derate bucket (log on change only)
    // Per-frame loop timings, recorded by default — the cost is a handful of
    // performance.now() calls and one small array row per submitted batch.
    // opts.perf: false disables.
    this.perf = opts.perf === false ? null : { frames: [], marks: [] };
    this._em = new Emitter();
    this._debug = null;     // solver internals (feats/tracks) for UI beats
    this.view = new SessionView(this);
  }

  /** Hydrate a restored score history: raw rows stay raw (dots, exports),
   *  and the filtered curve replays through a fresh filter — the same
   *  fresh pass the graph runs per draw, so a reopened run's trigger
   *  history matches its plotted line from the first slope test. */
  setLossHistory(rows) {
    this.lossHistory = rows.map((r) => Array.from(r || []));
    this._zlema = new FastRobustZLEMA();
    this.zlemaHistory = this._zlema.series(this.lossHistory.slice(SKIP_FIRST));
  }

  on(type, fn) { return this._em.on(type, fn); }
  _log(m) { this._em.emit('log', m); }
  _stage(e) { this._em.emit('stage', e); }

  /** Decode photographs into Frames. files: File[]/Blob[]/{source,name}[]
   *  360 equirectangular panos (2:1 aspect) are detected here and sliced
   *  into six-face cubemap rigs — the solver then treats each pano as ONE
   *  camera pose and the trainer sees ordinary pinhole faces. */
  async load(files) {
    this._stage({ stage: 'decode', done: 0, total: files.length });
    const { prepared, rigs, focalNative, faceSize } = await this._slicePanos(files);
    this.frames = await decodeFrames(prepared, {
      ...this.opts.frames, log: (m) => this._log(m),
    });
    this._stage({ stage: 'decode', done: this.frames.length, total: files.length });
    if (this.frames.length < 2) throw new Error('need at least 2 decodable images');
    if (rigs) {
      // the sliced focal is exact but lives in face-native pixels; the
      // solver works at feature scale
      this.rigInfo = rigs;
      this.rigFocalPx = focalNative * this.frames[0].fw / faceSize;
      this._log(`camera rigs active: ${new Set(rigs.filter(Boolean).map((r) => r.id)).size} panos, ` +
        `focal ${this.rigFocalPx.toFixed(1)}px (known from the slice)`);
    }
    return this.frames;
  }

  /** Detect equirect panos in the input and slice them into rig faces.
   *  Detection reads image HEADERS only (probeImageSize) — a normal photo
   *  set pays nothing here; panos are decoded once, sliced, released. */
  async _slicePanos(files) {
    // Calibrated XR images must retain their captured frustum and pixels.
    if (this.opts.frames?.disablePano) return { prepared: files, rigs: null };
    const meta = [];
    let panos = 0, sized = 0, maxW = 0;
    for (const f of files) {
      const src = f && f.source !== undefined ? f.source : f;
      const name = (f && f.name) || (src && src.name) || `image_${meta.length}`;
      const dims = await probeImageSize(src);
      const pano = !!dims && isEquirect(dims.w, dims.h);
      if (dims) sized++;
      if (pano) { panos++; maxW = Math.max(maxW, dims.w); }
      meta.push({ src, name, pano });
    }
    if (!panos) return { prepared: files, rigs: null };
    if (panos < sized) {
      throw new Error('mixed input: 360 panoramas and regular photos in one set are not ' +
        'supported yet — drop either panos or photos, not both');
    }
    // one shared face size + fov for the whole set (the solver's focal is shared)
    const size = faceSizeFor(maxW);
    this._log(`360 input: slicing ${panos} equirectangular panos into ${size}px cubemap rigs`);
    const prepared = [], rigs = [];
    let f = 0, rigId = 0;
    for (const b of meta) {
      const bmp = await createImageBitmap(b.src);
      const sl = sliceEquirect(bmp, size);
      bmp.close?.();
      f = sl.f;
      const base = b.name.replace(/\.[^.]+$/, '');
      sl.faces.forEach((cv, k) => {
        prepared.push({ source: cv, name: `${base}_f${k}` });
        rigs.push({ id: rigId, R: FACE_ROTS[k] });
      });
      rigId++;
      this._stage({ stage: 'decode', done: rigId, total: meta.length });
      await new Promise((r) => setTimeout(r, 0));
    }
    return { prepared, rigs, focalNative: f, faceSize: size };
  }

  /** Use frames prepared elsewhere (e.g. processSource on fetched bitmaps). */
  useFrames(frames) { this.frames = frames; return frames; }

  /** Structure from motion: camera poses + sparse points from the frames. */
  async solve(extra = {}) {
    const opts = { ...this.opts.sfm, ...extra };
    if (this.rigInfo && !opts.rigs) {
      opts.rigs = this.rigInfo;
      opts.focalPx = opts.focalPx ?? this.rigFocalPx;
      // sliced faces are bilinear resamples of the pano: the upsampled SIFT octave
      // (desktop default -1 since 2026-09-04) finds "features" in the interpolation.
      // MEASURED bar360 30k 2026-09-06: octave -1 19.90-19.95 vs octave 0 20.64-20.69
      // (aspect and the 8000 budget are neutral) — rigs never search below octave 0.
      if ((opts.siftFirstOctave ?? 0) < 0) opts.siftFirstOctave = 0;
      // and their f, k1 = k2 = 0, square pixel are exact by construction: BA refining
      // them only fits noise. MEASURED bar360 30k: lock +0.29 (old solve 20.69 -> 20.98),
      // +0.22 on the octave -1 solve. sfm.lockIntrinsics: false opts out.
      opts.lockIntrinsics = opts.lockIntrinsics ?? true;
    }
    // EXIF focal prior: when most photos carry the same 35 mm-equivalent
    // focal (one phone, one lens), the solver starts from it and skips the
    // four-candidate focal search (sfm.js keeps the search as the fallback).
    // sfm.exifFocal: false opts out; a rig's exact focal takes precedence.
    if (opts.exifFocal !== false && !opts.focalPx && !opts.focalPrior) {
      const f35s = this.frames.map((f) => f.exif && f.exif.f35).filter((v) => v > 0);
      if (f35s.length >= Math.max(2, 0.6 * this.frames.length)) {
        const sorted = [...f35s].sort((a, b) => a - b);
        const med = sorted[sorted.length >> 1];
        const spread = sorted[sorted.length - 1] / sorted[0];
        if (spread <= 1.1) {
          const fr = this.frames[0];
          opts.focalPrior = focalPxFrom35(med, fr.fw, fr.fh);
          this._log(`EXIF focal ${med} mm (35 mm-equivalent) on ${f35s.length}/${this.frames.length} photos → ` +
            `prior ${opts.focalPrior.toFixed(1)} px at the ${fr.fw} px feature frame`);
          // a VIDEO's 35 mm value names the lens, not the recording: stabilisation and
          // the video sensor crop lengthen the effective focal by up to ~1.5x. The prior
          // is tried first (accepted at >= 60 % registration); the fallback search then
          // runs on a narrow grid of crop factors around it instead of the whole range,
          // which on a person orbit picked three different focals at three feature
          // resolutions (2026-09-15)
          if (this.frames.some((f) => f.exif && f.exif.video)) {
            // a VIDEO's 35 mm value names the lens, not the recording: taken as the
            // focal it is accepted at >= 60 % registration and bent (Tom: 0.37x, 62/65);
            // narrowed to crop factors 1.0-1.55 the search still picked 0.46x (62/65,
            // ghosting) where the free search on every frame finds 0.55x with 65/65
            // and a clean model (2026-09-15). So for video the lens is logged and the
            // focal search keeps its own range; the every-frame search carries it.
            opts.focalPrior = 0;
            this._log(`video lens ${med} mm (35 mm-equivalent): noted, the focal search keeps its full range (a person orbit's effective focal is the lens x an unknown crop)`);
          }
        } else {
          this._log(`EXIF focal varies across the set (${sorted[0]}–${sorted[sorted.length - 1]} mm) — focal search kept`);
        }
      }
    }
    // the GPU matcher shares the session device (created here rather than at
    // seed) — it carries the raised buffer limits big feature sets need.
    // Remote training skips it: the matcher then uses a transient own device
    // (destroyed after matching) and the worker owns the only live device.
    if (!this.gpu && !this._shouldOffload()) {
      this.gpu = await createGpu({ device: this.opts.device });
      this.gpu.onLost = (info) => this._deviceLost(info);
    }
    this.recon = await runSfM(
      this.frames,
      (m) => this._log(m),
      (imgIdx, x, y) => this.frames[imgIdx].sampleColor(x, y),
      {
        gpu: this.gpu,
        ...opts,
        onEvent: (e) => { this._stage(e); if (opts.onEvent) opts.onEvent(e); },
        debug: (d) => { this._debug = d; if (opts.debug) opts.debug(d); },
      });
    if (undistortFrames(this.frames, this.recon)) {
      this._log(`undistorted training images (k1 ${this.recon.k1.toFixed(4)}, k2 ${this.recon.k2.toFixed(4)})`);
    }
    if (this.opts.lowMem) {
      // the feature-scale grayscale is solve-only (SIFT + LK); on phones
      // 50 frames of it is ~140MB standing between us and the trainer
      for (const f of this.frames) f.gray = null;
    }
    this._stage({ stage: 'solved', done: 1, total: 1, detail: {
      cams: this.recon.cams.length, points: this.recon.points.length,
      rms: this.recon.rmsBA,
    } });
    return this.recon;
  }

  /** Train on cameras known from elsewhere instead of solving: a renderer's ground truth, a
   *  rig, an AR session. `cameras` from parseCameras() (transforms.json or a ground-truth list),
   *  in image pixels and matched to the loaded frames by file name; opts.points seeds the model
   *  ([{ X, rgb }], else the viewed volume is filled at random).
   *  Pre-matted renders: load them with opts.frames.alphaMask. They are often small enough to
   *  train at native size; the frames' memory budget may still pick a lower resolution than the
   *  renders (opts.frames.trainMaxDim decides). The photo blur gate is off for these sets (a
   *  thin object seen edge-on scores "blurry"); opts.excludeBlurry turns it back on. */
  useKnownCameras(cameras, opts = {}) {
    if (!this.frames) throw new Error('load() first');
    const recon = this.useReconstruction(reconFromKnownCameras(this.frames, cameras, opts));
    this._log(`known cameras: ${recon.cams.length} of ${this.frames.length} frames, ` +
      `${recon.points.length} ${opts.points && opts.points.length ? 'given' : 'random'} seed points, no solve`);
    return recon;
  }

  /** Load a reconstruction obtained elsewhere ({ cams, points, ... }). */
  useReconstruction(recon) {
    this.recon = { k1: 0, k2: 0, fScale: 1, medErr: 0, rmsBA: null, ...recon };
    // stored/external recons (a resumed run, a GT solve) usually lack fFeat.
    // undistortFrames divides by it — undefined turned EVERY target pixel
    // NaN→invalid, so a continued run trained against emptiness and faded
    // itself to transparency. All cams share the solve's feature-scale
    // focal, so the first one serves.
    if (this.recon.fFeat == null && this.recon.cams && this.recon.cams.length) {
      this.recon.fFeat = this.recon.cams[0].f;
    }
    return this.recon;
  }

  /** Drop sparse-cloud points that the subject masks put in empty space.
   *
   *  Without this, a masked run SEEDS THE THING IT IS TRYING TO REMOVE: the
   *  sparse cloud is mostly room, and a room Gaussian that reaches opacity ~1
   *  can never come back down, because dalpha/d(logit opacity) = o(1-o)(...)
   *  vanishes as o -> 1. Neither the coverage loss nor opacityReg can rescue a
   *  saturated splat; both only govern what grows. So the cure is not to
   *  create it. (Measured 2026-09-11: coverage loss alone, covW 3, left the
   *  room fully opaque at 45% dead splats.)
   *
   *  The masks are already in the frames (frame.alpha), so this needs no
   *  extra input. A point is kept when
   *  it lands on the subject in at least `keep` of the views that see it.
   */
  maskPoints(keep = 0.5) {
    const frames = this.frames;
    if (!frames.some((f) => f.alpha && f.emptyFrac > 0)) return 0;
    const pts = this.recon.points;
    const seen = new Int32Array(pts.length);
    const subj = new Int32Array(pts.length);
    for (const c of this.recon.cams) {
      const im = frames[c.imgIdx];
      const sx = im.tw / im.fw, sy = im.th / im.fh;
      const { R, t } = c;
      for (let i = 0; i < pts.length; i++) {
        const X = pts[i].X;
        const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
        if (z <= 1e-6) continue;
        const u = (c.f * (R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0]) / z + c.cx) * sx;
        const v = ((c.fy ?? c.f) * (R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1]) / z + c.cy) * sy;
        if (u < 0 || v < 0 || u >= im.tw || v >= im.th) continue;
        seen[i]++;
        if (im.alpha[(v | 0) * im.tw + (u | 0)] > 64) subj[i]++;   // not known-empty
      }
    }
    const before = pts.length;
    this.recon.points = pts.filter((_, i) => seen[i] > 0 && subj[i] / seen[i] >= keep);
    const dropped = before - this.recon.points.length;
    this._log(`subject mask: seeding from ${this.recon.points.length} of ${before} points `
      + `(${(100 * dropped / before).toFixed(0)}% of the sparse cloud was background)`);
    // a solve that registered the room but not the person (a wrong focal
    // winner: Lisa's 4K orbit put 108 of 13741 points on her once, seeded 14,
    // and trained 2,814 splats for 20k iterations, 2026-09-14) must stop
    // here — nothing downstream can recover a subject that was never seeded
    if (before >= 2000 && this.recon.points.length < Math.max(150, 0.01 * before))
      throw new Error(`the solve missed the person: only ${this.recon.points.length} of ${before} sparse points land on the subject — try again, or shoot a slower orbit at arm's length`);
    return dropped;
  }

  /** Carve the subject hull from the masks and hand it to the trainer, so
   *  refinement treats anything in provably empty space as dead capacity.
   *  No-op when the frames carry no masks. */
  _buildHull(opts = {}) {
    if (opts.hull === false) return null;
    const t0 = Date.now();
    const hull = buildVisualHull(this.recon, this.frames, opts.hullOpts);
    if (!hull) return null;
    this.hull = hull;
    this.hullTest = makeHullTest(hull);
    this.splatTest = makeSplatTest(hull, opts.hullOpts);
    // seed from inside the hull only: the silhouette test alone cannot remove
    // a point that sits along the viewing ray through the subject, and those
    // stragglers are what blow the scene bounds out
    const before = this.recon.points.length;
    const inside = this.recon.points.filter((p) => this.hullTest(p.X[0], p.X[1], p.X[2]));
    this._log(`subject hull: ${hull.dim.join('x')} voxels of ${hull.cell.toFixed(3)} (box ${hull.dim.map((d) => (d * hull.cell).toFixed(2)).join('x')}), ${(hull.fill * 100).toFixed(1)}% solid, `
      + `carved in ${((Date.now() - t0) / 1000).toFixed(1)}s; seeding from `
      + `${inside.length} of ${before} points inside it`);
    // a hull that disagrees with the masks (a bounding box blown out by ray
    // stragglers, a matte that does not line up) would seed NOTHING and
    // kill every splat that grows — the run trained 0 Gaussians in the app,
    // 2026-09-13. Then the masks alone must do: no hull, all mask-filtered points.
    if (inside.length < Math.max(8, 0.05 * before) || hull.fill < 0.002) {
      this._log(`subject hull rejected (${inside.length} of ${before} points, ${(hull.fill * 100).toFixed(2)}% solid) — training on the masks alone`);
      this.hull = null; this.hullTest = null; this.splatTest = null;
      return null;
    }
    this.recon.points = inside;
    return hull;
  }

  /** Seed Gaussians from the sparse cloud and set up the WebGPU trainer. */
  async seed(extra = {}) {
    if (!this.recon) throw new Error('solve() first');
    this._stage({ stage: 'seed', done: 0, total: 1 });
    const allPoints = this.recon.points;
    // opts.maskTraining === false: the mattes stay on the frames (the hull and
    // the cut after training read them) but training does not see them — no
    // seed filter, no hull kill, no random background, alpha 255 in the
    // targets. The masked recipe drove the needle look on a person (needle
    // ratio 344 vs 15 for the same clip trained as a room, 2026-09-14b).
    const masked = this.opts.maskTraining !== false;
    if (masked && extra.maskPoints !== false) this.maskPoints(extra.maskKeep ?? 0.5);
    if (masked) this._buildHull(extra);
    if (this.recon.points.length < 8) {
      // never seed from nothing: the whole cloud beats an empty model
      this._log(`subject filters left ${this.recon.points.length} points — seeding from the whole cloud instead`);
      this.recon.points = allPoints;
    }
    // default seed scales with the solve's point count: the flat 60k
    // default seed-bound capacity (cap = seed x capMult) on point-rich
    // scenes — garden measured +0.4 dB from lifting it. Explicit
    // initTarget (phones pass one) always wins.
    const target = extra.initTarget || this.opts.initTarget ||
      Math.min(250000, Math.max(60000, this.recon.points.length * 8));
    // The 24 cap suits a scene cloud, where points are plentiful and heavy
    // cloning only duplicates. A masked SUBJECT cloud is the opposite case —
    // a person in dark clothing gives few hundred points — and capping there
    // starves the model (measured: 468 points -> 11.7k seed -> 15.7k splats
    // at 6k cycles, 2 dB down). Lift the cap when the cloud is small so the
    // seed still reaches the target.
    const cloneCap = this.recon.points.length < 5000 ? 200 : 24;
    const clones = Math.min(cloneCap, Math.max(2, Math.round(target / this.recon.points.length) - 1));
    const v2 = this.opts.trainer && this.opts.trainer.engine === 'v2';
    this.model = initGaussians(this.recon.points, clones, undefined,
      v2 ? { dc: 'sh', randRot: true } : {});
    // extra.appendGaussians: ready-made rows (stride 16, sigmoid-DC convention) joined to
    // the cloud seed as they are — flat, oriented, sized (a face mesh seed, avatar mode)
    if (extra.appendGaussians && extra.appendGaussians.n > 0) {
      const g = extra.appendGaussians; const m = this.model; const merged = new Float32Array((m.n + g.n) * 16);
      merged.set(m.data.subarray(0, m.n * 16), 0); merged.set(g.data.subarray(0, g.n * 16), m.n * 16);
      this.model = { ...m, data: merged, n: m.n + g.n };
      this._appendedSeed = { from: m.n, to: m.n + g.n, protectIters: g.protectIters || 0, freezePos: !!g.freezePos };
      this._log(`+ ${g.n} seed Gaussians appended (${g.note || 'ready-made'})${g.protectIters ? `, protected from relocation for ${g.protectIters} iterations` : ''}`);
    }
    this._log(`initialized ${this.model.n} Gaussians (scene radius ${this.model.radius.toFixed(2)})`);

    // opts.shHorizontal: SH evaluated on the view direction's horizontal part only —
    // the up axis is the cameras' dominant up (minus each R's second row)
    const shUp = this.opts.shHorizontal ? this._camerasUp() : null;
    if (shUp) this._log(`SH on the horizontal view direction only (up ${shUp.map((v) => v.toFixed(2)).join(', ')})`);
    const trainerOpts = {
      maxIters: this.opts.maxIters ?? 60000,
      ...(shUp ? { shUp } : {}),
      // a masked set trains its empty pixels against a random background by
      // default (gs/shaders.js randBg) — the thing that keeps splats out of
      // the cleared area at full photometric strength
      ...(masked && this.frames && this.frames.some((f) => f.emptyFrac > 0) ? { randomBg: true } : {}),
      ...this.opts.trainer, ...extra.trainer,
    };
    // training-resolution intrinsics (recon poses live at feature scale)
    const cams = this.recon.cams.map((c) => {
      const im = this.frames[c.imgIdx];
      const s = im.tw / im.fw;
      return { ...c, f: c.f * s, ...(c.fy != null ? { fy: c.fy * s } : {}),
        // the principal point is normally the frame centre; a crop camera (a
        // native-resolution window cut out of a larger image, see
        // tests/bench/face_crops.py) carries its own, at feature scale
        cx: c.cx != null ? c.cx * s : im.tw / 2, cy: c.cy != null ? c.cy * s : im.th / 2, w: im.tw, h: im.th };
    });
    // output buffers must fit the largest view the host will ever render —
    // interactive canvases are usually LARGER than the training resolution
    const maxW = Math.max(this.opts.maxViewW ?? 2560, ...cams.map((c) => c.w));
    const maxH = Math.max(this.opts.maxViewH ?? 1440, ...cams.map((c) => c.h));
    // lowMem released the frame pixels after the first boot (and the worker
    // owns the seed) — a second seed (recover) has nothing to boot from.
    // Say so plainly instead of crashing inside setup on null rgb.
    // (recover's catch shows the device-lost screen, so this text is for
    // the log — and it stays "reload": the old trainer died with the
    // device, so unlike a refused prune there is nothing to resume.)
    if (this._seedPixelsReleased()) {
      throw new Error('seed data was released after the first boot (lowMem) — reload to train again');
    }
    await this._bootTrainer(trainerOpts, {
      gaussians: this.model, cams, frames: this.frames, maxW, maxH, radius: this.model.radius,
      hull: (masked && this.splatTest) ? this.hull : null, hullOpts: extra.hullOpts,
      hullKillFn: (masked && this.splatTest) || null,
      protect: (this._appendedSeed && this._appendedSeed.protectIters > 0)
        ? { from: this._appendedSeed.from, to: this._appendedSeed.to, until: this._appendedSeed.protectIters } : null,
      freezePos: (this._appendedSeed && this._appendedSeed.freezePos)
        ? { from: this._appendedSeed.from, to: this._appendedSeed.to } : null,
    });
    if (this.opts.lowMem) {
      // targets now live on the GPU (packed RGBA8); the float32 CPU copies
      // are 3x that size and nothing reads them after setup
      for (const f of this.frames) { f.rgb = null; f.sampleColor = () => [0.5, 0.5, 0.5]; }
    }

    await this._applyTrainingSplit(extra);

    if (this._appendedSeed && this._appendedSeed.freezePos) this._log(`seed rows ${this._appendedSeed.from}-${this._appendedSeed.to} keep their positions`);
    this._stage({ stage: 'seed', done: 1, total: 1, detail: { splats: this.model.n } });
    return this.model;
  }

  /** the cameras' dominant up axis (unit, world) */
  _camerasUp() {
    const u = [0, 0, 0];
    for (const c of this.recon.cams) { u[0] -= c.R[3]; u[1] -= c.R[4]; u[2] -= c.R[5]; }
    const l = Math.hypot(u[0], u[1], u[2]) || 1; return [u[0] / l, u[1] / l, u[2] / l];
  }

  /** trainer.setup with the frame alphas hidden when the session trains unmasked
   *  (opts.maskTraining === false): the packed targets then carry alpha 255. */
  _setupTrainer(trainer, cams, frames, maxW, maxH, radius) {
    const hide = this.opts.maskTraining === false && frames && frames.some((f) => f.alpha);
    const saved = hide ? frames.map((f) => f.alpha) : null;
    if (hide) for (const f of frames) f.alpha = null;
    try { trainer.setup(this.model, cams, frames, maxW, maxH, radius); }
    finally { if (hide) frames.forEach((f, i) => { f.alpha = saved[i]; }); }
  }

  /** Whether the GPU training core should live in a dedicated worker.
   *  'auto' (default) attempts it and falls back to local training on any
   *  failure; false forces local; true demands the worker. A host-owned
   *  device (opts.device) can never cross threads, so it always forces local
   *  — loudly under true, silently under auto. */
  _shouldOffload() {
    const mode = this.opts.trainWorker ?? 'auto';
    if (mode === false || mode === 'never') return false;
    if (this.opts.device) {
      if (mode === true) throw new Error('trainWorker: true with opts.device — a host-owned device cannot move to a worker');
      return false;
    }
    if (typeof Worker === 'undefined') {
      if (mode === true) throw new Error('trainWorker: true but Web Workers are unavailable');
      return false;
    }
    return true;
  }

  /** Create the trainer and run setup, local or remote per _shouldOffload.
   *  boot: { gaussians, cams, frames, maxW, maxH, radius, hull?, hullOpts?,
   *  hullKillFn? (local), protect?, freezePos?, sh?/shN? (resume), holdout?,
   *  excluded?, iter? }. */
  /** Prune/grow tandem: pre-boost growRate so the per-refine cull doesn't
   *  bend the growth curve — every refine culls p and grows g', with g'
   *  solving (1+g') * (1-p) = (1+g) exactly. Runs here, the single funnel
   *  into local/remote/worker boots, so every path trains the same
   *  trajectory; ses.opts.trainer stays unboosted, so continuations
   *  recompute from the base rate instead of compounding. Engine v2
   *  grows candidates, not n — the same relative uplift there is
   *  approximate. */
  _withTandemBoost(trainerOpts) {
    const sp = trainerOpts.softPrune;
    if (!sp || !(sp.ratio > 0 && sp.ratio < 1)) return trainerOpts;
    const g0 = trainerOpts.growRate ?? 0.15;
    const g1 = (1 + g0) / (1 - sp.ratio) - 1;
    if (!(g1 > g0)) return trainerOpts;
    let note = '';
    let g = g1;
    if (g > 0.6) { g = 0.6; note = ' [clamped at 0.6]'; }
    const out = { ...trainerOpts, growRate: +g.toFixed(4) };
    if (trainerOpts.engine === 'v2') {
      out.growFrac = +((trainerOpts.growFrac ?? 0.1) * (g / g0)).toFixed(4);
    }
    this._log(`prune/grow tandem: growRate ${g0} -> ${out.growRate} holds the no-prune curve against -${Math.round(sp.ratio * 100)}% every refine${note}`);
    return out;
  }

  async _bootTrainer(trainerOpts, boot) {
    trainerOpts = this._withTandemBoost(trainerOpts);
    if (!trainerOpts.softPrune && trainerOpts.useStats === undefined) {
      // soft-prune-off runs still cull vote-confirmed dead splats, scored
      // off the same rendered-mass window — without the accumulation the
      // gather reads all-zero and the dead cull silently never fires
      trainerOpts = { ...trainerOpts, useStats: true };
    }
    if (this._shouldOffload()) {
      try {
        await this._bootRemoteTrainer(trainerOpts, boot);
        return;
      } catch (e) {
        if ((this.opts.trainWorker ?? 'auto') === true) throw e;
        if (e && e.workerOwned) throw e;   // the seed moved — fallback would boot from detached buffers
        this._log(`worker training unavailable (${(e && e.message) || e}) — training on the main thread`);
        // the failed boot published nothing: the previous core (a reseed's)
        // stays live until the fallback below swaps its own in (2026-11)
      }
    }
    const twMode = this.opts.trainWorker ?? 'auto';
    if (twMode === false || twMode === 'never') this._log('training on the main thread (trainWorker disabled)');
    if (!this.gpu) this.gpu = await createGpu({ device: this.opts.device });
    this.gpu.onLost = (info) => this._deviceLost(info);
    // boot into a local and swap on success: publishing this.trainer BEFORE
    // setup() meant any setup throw left a half-booted trainer (camMeta set,
    // buffers missing) that passed the frame guard and then failed every
    // call cryptically — the prune-failure cascade (2026-11)
    const next = await GSTrainer.create({ ...trainerOpts, gpu: this.gpu });
    const gi = this.gpu.info || {};
    this._log(`GPU: ${gi.vendor || 'unknown'} ${gi.architecture || ''} — ` +
      `${next.tileGrad ? 'tile-shared' : 'direct'} gradient accumulation`);
    this._setupTrainer(next, boot.cams, boot.frames, boot.maxW, boot.maxH, boot.radius);
    if (boot.hullKillFn) next.hullKill = boot.hullKillFn;
    if (boot.protect) next.protect = { ...boot.protect };
    if (boot.freezePos) next.setFreezePos(boot.freezePos.from, boot.freezePos.to);
    if (boot.sh && next.bufSH) {
      // a checkpoint with more splats than the cap truncated the params
      // but pasted all shN rows into a cap-sized SH buffer — an upload
      // larger than its buffer. Clamp to the same rows the params kept.
      const shRows = Math.min(boot.shN ?? next.cap, next.cap);
      if (shRows < boot.shN) {
        this._log(`SH restore truncated to ${shRows} of ${boot.shN} rows (splat cap)`);
      }
      next.device.queue.writeBuffer(next.bufSH, 0,
        boot.sh.buffer, boot.sh.byteOffset, shRows * next.shK * 3 * 4);
    }
    next.excluded = new Set(boot.excluded || []);
    next.holdout = boot.holdout ?? -1;
    if (boot.iter != null) { next.iter = boot.iter; next.adamT0 = boot.iter; }
    if (this._worker) this._terminateWorker();   // the replacement is in: retire the previous core
    this.trainer = next;
    this._remote = false;
  }

  /** Spawn the training worker and upload the seed + targets through the
   *  trainer protocol. Images are projected to cloneable payloads (frames
   *  carry functions); lowMem transfers ownership instead of copying. */
  async _bootRemoteTrainer(trainerOpts, boot) {
    const factory = this.opts.createTrainWorker ||
      (() => new Worker(new URL('./gs/trainworker.js', import.meta.url), { type: 'module' }));
    let worker;
    try {
      worker = await factory();
    } catch (e) {
      throw new Error(`training worker did not start (${(e && e.message) || e})`);
    }
    const remote = new RemoteTrainer(worker, trainerOpts);
    remote.onEvent('log', (m) => this._log(m));
    // lowMem moves (transfers) the seed and targets instead of copying: from
    // the setup postMessage on, the main-side buffers are detached. A worker
    // failure past that point is fatal — a local fallback would boot from
    // dead memory (its first upload throws "Number of bytes to write is too
    // large", 2026-11) — and the ORIGINAL error is the diagnosable one.
    const moved = !!this.opts.lowMem;
    try {
      const images = boot.frames.map((f) => ({ tw: f.tw, th: f.th, rgb: f.rgb, alpha: f.alpha ?? null }));
      if (this.opts.maskTraining === false && images.some((im) => im.alpha)) {
        for (const im of images) im.alpha = null;   // mirror _setupTrainer
      }
      const mirror = await remote.setupRemote({
        gaussians: boot.gaussians, cams: boot.cams, images,
        maxViewW: boot.maxW, maxViewH: boot.maxH, radius: boot.radius,
        trainerOpts, hull: boot.hull ?? null, hullOpts: boot.hullOpts,
        protect: boot.protect ?? null, freezePos: boot.freezePos ?? null,
        holdout: boot.holdout ?? -1, excluded: [...(boot.excluded || [])],
        iter: boot.iter ?? null,
        transferOwnership: !!this.opts.lowMem,
      });
      // matching is over: release a main-side device left from solve() rather
      // than carry two live devices (solve already skips creating one when
      // offloading; this covers option-timing edges)
      if (this.gpu && this.gpu.owned) {
        try { this.gpu.dispose(); } catch { /* already gone */ }
        this.gpu = null;
      }
      // the replacement is fully booted: retire the previous core (a reseed
      // that orphaned it carried two live GPU devices) and publish. Fault
      // events wire up HERE, so a dying replacement never reports against
      // the previous core it failed to replace (2026-11).
      if (this._worker) this._terminateWorker();
      remote.onEvent('device-lost', (info) => this._deviceLost(info));
      remote.onEvent('worker-error', (err) => this._workerFailed(err));
      this._worker = worker;
      this.trainer = remote;
      this._remote = true;
      this._workerGen++;
      this.gpu = { info: mirror.gpuInfo || {}, owned: false, dispose() {} };
      const gi = mirror.gpuInfo || {};
      this._log(`GPU: ${gi.vendor || 'unknown'} ${gi.architecture || ''} — ` +
        `${mirror.tileGrad ? 'tile-shared' : 'direct'} gradient accumulation (worker)`);
    } catch (e) {
      // the failed boot published nothing: terminate ONLY the replacement
      // and fail its RPCs — the previous trainer stays live (a failed prune
      // reseed keeps training on the unpruned model) and the view stays
      // attached to it (2026-11)
      try { remote.abort(e instanceof Error ? e : new Error(String((e && e.message) || e))); } catch { /* gone */ }
      try { if (typeof worker.terminate === 'function') worker.terminate(); } catch { /* gone */ }
      if (moved) {
        const fatal = new Error(`training worker failed (${(e && e.message) || e}) — ` +
          `the seed already moved to the worker, so training cannot fall back to the main thread; reload to retry`);
        fatal.cause = e;
        fatal.workerOwned = true;
        throw fatal;
      }
      throw e;
    }
  }

  _terminateWorker() {
    // the dead worker answers nothing: fail the in-flight RPCs (else they
    // hang) and drop the view chain's tail (else every post-recovery view
    // command queues behind a promise that never settles — a black view
    // after every recover, 2026-11)
    if (this.trainer && this._remote && typeof this.trainer.abort === 'function') {
      try { this.trainer.abort(new Error('training worker terminated')); } catch { /* gone */ }
    }
    if (this._worker) {
      try { if (typeof this._worker.terminate === 'function') this._worker.terminate(); } catch { /* gone */ }
      this._worker = null;
    }
    this._remote = false;
    if (this.view) {
      this.view._chain = null;
      this.view._remoteInFlight = false;   // aborted sends never settle: drop the gate with the chain
      // the abort above rejects in-flight view commands by design — not a
      // view failure. (Cleared by the next success or resync, so a genuinely
      // broken re-attach still surfaces.)
      this.view._viewFailed = true;
      this.view._falseSince = null;
    }
  }

  /** Commit the training split computed by _applyTrainingSplit. */
  async _commitSplit(excluded) {
    if (this._remote) await this.trainer.setConfig({ excluded: [...excluded], holdout: this.holdout });
    else { this.trainer.excluded = excluded; this.trainer.holdout = this.holdout; }
  }

  /** After a run with photometric pose optimisation (trainer opts.camOpt): log how far
   *  the cameras moved and write the moved poses back into recon.cams, so everything
   *  after training (hull, cut, landmarks, body fit) sees the frame the model was
   *  trained in. camMeta and recon.cams share their order. */
  async _camDriftDone() {
    try {
      const tr = this.trainer; if (!tr || !tr.opts.camOpt || !this.recon || !this.recon.cams) return;
      let dr, poses;
      if (this._remote) {
        ({ report: dr, poses } = await tr.camDrift());
      } else {
        dr = tr.camDrift();
        if (!dr) return;
        poses = tr.camMeta;
      }
      if (!dr) return;
      let moved = 0;
      for (let i = 0; i < poses.length && i < this.recon.cams.length; i++) {
        const m = poses[i], c = this.recon.cams[i]; if (!m || !c) continue;
        c.R = Array.from(m.R); c.t = Array.from(m.t); moved++;
      }
      this._log(`pose optimisation: ${dr.n} cameras moved — rotation median ${dr.rotDegMedian.toFixed(3)}° max ${dr.rotDegMax.toFixed(2)}°, centre median ${dr.trnMedian.toFixed(4)} max ${dr.trnMax.toFixed(3)} scene units; ${moved} poses written back to the reconstruction`);
    } catch (e) { this._log(`pose drift report failed: ${e.message || e}`); }
  }

  /** Which cameras train and which are scored: blur exclusions, the chart
   *  holdout and the evaluation split. Shared by seed() and a continuation
   *  (seedFrom with frames), so a resumed run keeps the same test set. */
  async _applyTrainingSplit(extra = {}) {
    // blur-aware training: the blurriest frames stay registered (their poses
    // hold the chain together) but are excluded from the loss so the model
    // doesn't learn their motion blur
    // opts.lossCams(cam): which registered cameras carry the loss at all — the
    // others keep their poses (hull, cut, landmarks) but never train. Avatar
    // crops-only training (2026-09-15): the person's native windows train, the
    // room frames do not.
    const inLoss = typeof this.opts.lossCams === 'function' && this.recon && this.recon.cams
      ? this.trainer.camMeta.map((m, i) => !!this.opts.lossCams(this.recon.cams[i], i)) : null;
    const sh = this.trainer.camMeta.map((m) => this.frames[m.imgIdx].sharpness);
    // the blur median is taken per camera GROUP (the frames that solved the scene vs
    // the avatar's crop windows, cams flagged crop): the groups decode at different
    // scales and their sharpness is not comparable — one median over the mix threw
    // out 87 of 519 cameras on a 4K orbit (2026-09-15), 1 of 208 without crops
    const group = this.trainer.camMeta.map((m, i) => (this.recon && this.recon.cams && this.recon.cams[i] && this.recon.cams[i].crop) ? 1 : 0);
    const medOf = (g) => { const v = sh.filter((x, i) => group[i] === g && (!inLoss || inLoss[i])).sort((a, b) => a - b); return v.length ? v[v.length >> 1] : 0; };
    const med = [medOf(0), medOf(1)];
    const excluded = new Set();
    const blurry = [0, 0];
    // known cameras (useKnownCameras) are mostly renders: their low "sharpness" is a thin
    // object seen edge-on, not motion blur — 16 of 180 sharp renders went out (2026-10-21)
    const blurGate = this.opts.excludeBlurry ?? !(this.recon && this.recon.known);
    this.trainer.camMeta.forEach((m, i) => {
      if (inLoss && !inLoss[i]) { excluded.add(i); return; }
      if (blurGate && sh[i] < med[group[i]] * 0.45) { excluded.add(i); blurry[group[i]]++; }
    });
    if (inLoss) this._log(`training loss on ${inLoss.filter(Boolean).length} of ${inLoss.length} cameras (opts.lossCams; the rest keep their poses)`);
    if (blurry[0] + blurry[1]) {
      const nCrop = group.filter((g) => g === 1).length;
      this._log(`excluding ${blurry[0] + blurry[1]} blurry cameras from the training loss ` +
        `(sharpness < 45% of the group median; poses kept)` + (nCrop ? ` — ${blurry[0]} of ${group.length - nCrop} frames, ${blurry[1]} of ${nCrop} crop windows` : ''));
    }

    // holdout: one sharp mid-sequence frame excluded from training and scored
    // as the honest novel-view metric
    const want = extra.holdout ?? this.opts.holdout ?? 'auto';
    if (want === 'auto') {
      let hi = this.trainer.camMeta.length >> 1;
      for (let k = 0; k < this.trainer.camMeta.length && excluded.has(hi); k++) {
        hi = (hi + 1) % this.trainer.camMeta.length;
      }
      this.holdout = hi;
    } else {
      this.holdout = (want == null || want < 0) ? -1 : want;
    }

    // benchmark protocol: every Nth frame joins the test set — excluded from
    // the loss (poses kept, same as the blur exclusions) and scored together
    // by evalTestPsnr(). The single chart holdout follows a mid-set test
    // frame so live progress tracks the same distribution.
    const split = extra.evalSplit ?? this.opts.evalSplit ?? 0;
    this.testCams = [];
    const evalNames = extra.evalFrames ?? this.opts.evalFrames;   // explicit test set by frame name (a fixed ruler across extractions of one video)
    if (evalNames && evalNames.length) {
      const want = new Set(evalNames);
      this.trainer.camMeta.forEach((m, i) => {
        const f = this.frames[m.imgIdx];
        if (f && want.has(f.name)) { excluded.add(i); this.testCams.push(i); }
      });
      if (this.testCams.length && this.holdout < 0) {
        this.holdout = this.testCams[this.testCams.length >> 1];
      }
      this._log(`evaluation set: ${this.testCams.length} of ${this.trainer.camMeta.length} cameras (named) held out of training`);
    } else if (split >= 2) {
      this.trainer.camMeta.forEach((m, i) => {
        if (m.imgIdx % split === 0) { excluded.add(i); this.testCams.push(i); }
      });
      if (this.testCams.length && this.holdout < 0) {
        this.holdout = this.testCams[this.testCams.length >> 1];
      }
      this._log(`evaluation split: ${this.testCams.length} of ${this.trainer.camMeta.length} ` +
        `cameras (every ${split}th) held out of training`);
    }
    await this._commitSplit(excluded);
  }

  /**
   * Attach an ALREADY-TRAINED model instead of seeding from the sparse cloud
   * — the loading half of exportPlyBlob / a saved session.
   *
   * gaussians: { data: Float32Array (n*16, the trainer's raw parameter
   * layout), n, sh?: Float32Array (n*shK*3), shK? }.
   *
   * View-only (no frames needed): pass opts.viewOnly and the trainer is set
   * up with zero training cameras — renders, tours and exports work, training
   * does not. For a full resume call useFrames()/load() first and the normal
   * camera wiring happens exactly like seed(); pass opts.iter to continue the
   * schedules where the saved run stopped.
   */
  /** True when a reseed has no pixels to re-upload: lowMem released the
   *  frame buffers after the first boot. View-only boots need no pixels.
   *  One predicate for seed(), seedFrom() and prune's pre-check, so the
   *  three cannot drift apart. */
  _seedPixelsReleased(viewOnly = false) {
    return !viewOnly && !!this.frames && this.frames.some((f) => !f.rgb);
  }

  async seedFrom(gaussians, opts = {}) {
    if (!this.recon && !opts.viewOnly) throw new Error('useReconstruction() first');
    // fail BEFORE touching this.model: a refused reseed leaves the session
    // bit-identical, so the prune catch resumes the previous trainer onto
    // the same model it paused (a failed prune never bricks the run)
    if (this._seedPixelsReleased(opts.viewOnly)) {
      throw new Error('seed data was released after the first boot (lowMem) — the current run is intact, but it cannot reseed');
    }
    this._stage({ stage: 'seed', done: 0, total: 1 });
    const radius = opts.sceneRadius ?? this.recon?.sceneRadius ?? 10;
    this.model = { data: gaussians.data, n: gaussians.n, radius, dc: gaussians.dc };

    const shUp2 = this.opts.shHorizontal && this.recon?.cams?.length ? this._camerasUp() : null;
    const trainerOpts = {
      maxIters: this.opts.maxIters ?? 60000,
      ...(shUp2 ? { shUp: shUp2 } : {}),
      // same masked-set default as seed(): without it a continuation trained
      // its empty pixels against BLACK (2026-09-12: 200 steps from a
      // converged person grew opaque dark needles out of the subject)
      ...(this.opts.maskTraining !== false && this.frames && this.frames.some((f) => f.emptyFrac > 0) ? { randomBg: true } : {}),
      ...this.opts.trainer, ...opts.trainer,
    };
    if (gaussians.shK != null) trainerOpts.shDeg = { 0: 0, 3: 1, 8: 2, 15: 3 }[gaussians.shK] ?? trainerOpts.shDeg;

    let cams = [];
    if (!opts.viewOnly) {
      cams = this.recon.cams.map((c) => {
        const im = this.frames[c.imgIdx];
        const s = im.tw / im.fw;
        return { ...c, f: c.f * s, ...(c.fy != null ? { fy: c.fy * s } : {}),
        // the principal point is normally the frame centre; a crop camera (a
        // native-resolution window cut out of a larger image, see
        // tests/bench/face_crops.py) carries its own, at feature scale
        cx: c.cx != null ? c.cx * s : im.tw / 2, cy: c.cy != null ? c.cy * s : im.th / 2, w: im.tw, h: im.th };
      });
    }
    const maxW = Math.max(this.opts.maxViewW ?? 2560, ...cams.map((c) => c.w));
    const maxH = Math.max(this.opts.maxViewH ?? 1440, ...cams.map((c) => c.h));
    if (opts.unbake && trainerOpts.mipComp !== false && this.recon?.cams?.length) {
      // a bare .ply/.sog export carries opacities with the Mip compensation
      // BAKED (exportPlyBlob); this trainer compensates at render time, so
      // undo the bake with the exporter's own f (training scale of the first
      // camera), camera positions and dilate. A state.bin is raw: never unbake it.
      const c0 = this.recon.cams[0];
      const fr0 = (this.frames || this.recon.frames || [])[c0.imgIdx];
      const f0 = cams.length ? cams[0].f : c0.f * (fr0 && fr0.tw && fr0.fw ? fr0.tw / fr0.fw : 1);
      const pos = Float32Array.from(this.recon.cams.flatMap(camPosition));
      // dilate derives from trainer opts alone (GSTrainer: opts.dilate ?? 0.1),
      // so the unbake needs no trainer instance — local or remote
      const dilate = trainerOpts.dilate ?? 0.1;
      this.model.data = unbakeOpacityCompensation(this.model.data, this.model.n, f0, pos, dilate);
      this._log(`unbaked the export's opacity compensation (f ${f0.toFixed(1)}, ${this.recon.cams.length} cams, dilate ${dilate})`);
    }
    await this._bootTrainer(trainerOpts, {
      gaussians: {
        data: this.model.data, n: gaussians.n, dc: gaussians.dc,
        ...(gaussians.sh ? { sh: gaussians.sh, shK: gaussians.shK } : {}),
      },
      cams, frames: this.frames || [], maxW, maxH, radius,
      sh: gaussians.sh || null, shN: gaussians.n,
      holdout: -1, excluded: [], iter: opts.iter || null,
    });
    this.holdout = -1;
    this.testCams = [];
    // a continuation trains like the run it continues: the same blur
    // exclusions and evaluation split (a resume that trained on the held-out
    // views scored them as train views)
    if (!opts.viewOnly && this.frames && cams.length) await this._applyTrainingSplit(opts);
    // a resumed run must not refine on its first frame: the refine timer is
    // not part of the saved state, and lastRefine 0 fired a growth refine
    // (+5 % splats) right after every resume (e2e resume spec, 2026-09-09)
    this.trainer.lastRefine = this.trainer.iter;
    this._log(`restored ${gaussians.n} Gaussians` +
      (opts.viewOnly ? ' (view only)' : ` at iteration ${opts.iter || 0}`));
    this._stage({ stage: 'seed', done: 1, total: 1, detail: { splats: gaussians.n } });
    return this.model;
  }

  // ---- the training loop (policy that used to live in the demo page) ----

  start() {
    if (!this.trainer) throw new Error('seed() first');
    if (this.training) return;
    this.training = true;
    // the throughput window must not span the pause gap — otherwise the
    // first resumed frame reads the idle seconds as GPU slowness and the
    // batch adapter comes back shrunk instead of at full speed
    this._thrT = null;
    this._stage({ stage: 'train', done: this.trainer.iter, total: this._maxIters() });
    this._ensureScheduler();
    if (!this._govRafRunning) {
      this._govRafRunning = true;
      this._sched.raf(() => this._govRafLoop());
    }
    this._scheduleFrame();
  }

  pause() { this.training = false; }

  /** Stop scheduling and wait for the currently submitted frame/readbacks to
   * settle. GPU buffer mapping is exclusive: prune, checkpoint, snapshot and
   * grow must cross this barrier before they start their own readback. Returns
   * whether training was running so callers can restore that state. */
  async pauseAndDrain() {
    const wasTraining = !!this.training;
    this.training = false;
    this._framePending = false;
    const active = this._frameWork;
    if (active) await active;
    // The finishing frame may have requested a view-only follow-up.
    this._framePending = false;
    if (!this._remote && this.trainer?.device?.queue) {
      await this.trainer.device.queue.onSubmittedWorkDone();
    }
    return wasTraining;
  }

  /** Raise the training horizon by `moreIters` and resume. The trainer's
   *  schedules (pos-lr decay, growth stop) are horizon-relative and stretch
   *  with it — this is a real continued run, not idling at floor lr. */
  async continueFor(moreIters) {
    if (!this.trainer) throw new Error('seed() first');
    this.opts.maxIters = (this.opts.maxIters ?? 60000) + moreIters;
    if (this._remote) await this.trainer.setConfig({ maxIters: this.opts.maxIters });
    else { this.trainer.opts.maxIters = this.opts.maxIters; this.trainer.horizon = this.opts.maxIters; }
    this.start();
    return this.opts.maxIters;
  }

  /** Prune-and-continue: compact the model to `ratio` of its splats with
   *  the training-free merge-down, reseed the trainer on the compacted
   *  model where the schedules stopped (in place when lowMem released
   *  the pixels — no pixels, no fresh boot), stretch the horizon, and resume.
   *  The holdout is probed before and after (skipped without one); a drop
   *  past 0.5 dB warns but never blocks — training recovers it. Resets
   *  the cap prompt, so the next ceiling hit asks again. A failed reseed
   *  leaves the previous trainer live (atomic swap) and a run that was
   *  training resumes onto it — a failed prune never bricks the run. */
  async pruneAndContinue(ratio = 0.5, { onStatus = null, onProgress = null } = {}) {
    const trainer = this.trainer;
    if (!trainer) throw new Error('seed() first');
    // Pause OUTSIDE the readback lock. An active frame can itself be waiting
    // for that lock; taking it first would deadlock, while skipping this
    // barrier caused GPUBuffer "outstanding map pending" in the service.
    const wasTraining = await this.pauseAndDrain();
    return this._locked(async () => {
      // pure prune: cull and reseed only — the run's state and horizon are
      // the visitor's, not the button's, so a paused run stays paused and
      // the horizon never stretches behind their back
      const say = (m) => { this._log(m); if (onStatus) onStatus(m); };
      // the loop restarts through start(), not the bare flag: the in-flight
      // frame already exited without rescheduling, and start() no-ops while
      // training — so the flag drops first to force the reschedule. A paused
      // run stays paused (the cap-hit card's prune included).
      const resume = () => { if (wasTraining) { this.training = false; this.start(); } };
      try {
        // pixel-less (lowMem) + local trainer: the merge-down applies IN
        // PLACE below — the button works mid-run on phones with no reseed
        // and no pixels. Pixel-less + worker has no row-upload RPC, so it
        // refuses here, before the merge-down burns minutes of phone CPU.
        const noPixels = this._seedPixelsReleased();
        if (noPixels && this._remote) {
          throw new Error('seed data was released after the first boot (lowMem) — the current run is intact, but it cannot reseed');
        }
        const before = this.holdout >= 0 ? await trainer.evalCamPsnr(this.holdout) : null;
        const raw = await trainer.readGaussians();
        say(`pruning ${raw.n.toLocaleString()} splats to ${Math.round(ratio * 100)}% …`);
        const simp = await simplifyGaussians(
          { data: raw.data, n: raw.n, sh: raw.sh, shK: raw.shK, dc: raw.dc ?? trainer.dcMode },
          { ratio, onStatus: (m) => say(String(m).replace(/^simplify: /, 'prune: ')), onProgress });
        const iter = trainer.iter;
        if (noPixels) await trainer.applyPrunedModel({ data: simp.data, n: simp.n, sh: simp.sh, shK: simp.shK });
        else await this.seedFrom(
          { data: simp.data, n: simp.n, sh: simp.sh, shK: simp.shK, dc: simp.dc },
          { iter });
        let after = null, delta = null;
        if (this.holdout >= 0 && before != null) {
          after = await this.trainer.evalCamPsnr(this.holdout);
          delta = after - before;
        }
        resume();
        this._capAsked = false;
        if (delta != null && delta < -0.5) {
          this._log(`WARNING: prune cost ${(-delta).toFixed(2)} dB ` +
            `(holdout ${before.toFixed(2)} → ${after.toFixed(2)}) — continuing anyway; training should recover it`);
        } else if (delta != null) {
          say(`prune holdout ${before.toFixed(2)} → ${after.toFixed(2)} dB`);
        } else {
          say(`pruned ${raw.n.toLocaleString()} → ${simp.n.toLocaleString()} splats (no holdout — flying blind)`);
        }
        const out = { from: raw.n, to: simp.n, before, after, delta, maxIters: this.opts.maxIters, resumed: wasTraining };
        this._em.emit('event', { kind: 'prune-continue', iter, ...out });
        return out;
      } catch (e) {
        // a failed prune must not swallow the prompt — the ceiling is
        // still full and the user can retry (or keep training at the cap).
        // The reseed swaps atomically, so the previous trainer is intact:
        // a run that was training resumes onto the unpruned model instead
        // of wedging paused on a half-booted core (2026-11).
        this._capAsked = false;
        try { resume(); } catch (re) { this._log(`prune cleanup: resume failed (${(re && re.message) || re})`); }
        if (e && e.workerOwned) {
          // the replacement's seed moved, but the previous core is intact
          // (and training again, above) — say that instead of "reload"
          const f = new Error(`prune reseed failed (${(e.cause && e.cause.message) || 'worker boot failed'}) — the unpruned model is intact`);
          f.cause = e;
          throw f;
        }
        throw e;
      }
    });
  }

  /** Replace one training pose at a paused model boundary. Recon, trainer
   * metadata and camera uniforms move together so checkpoints reopen exactly. */
  async updateCameraPose(ci, R, t) {
    const trainer = this.trainer;
    if (!trainer || !trainer.camMeta?.[ci]) throw new Error('camera is not trainable');
    const wasTraining = await this.pauseAndDrain();
    try {
      await this._locked(async () => {
        if (this._remote) await trainer.updateCameraPose(ci, R, t);
        else {
          const meta = trainer.camMeta[ci]; meta.R = Array.from(R); meta.t = Array.from(t);
          trainer.camUniforms[ci] = trainer._camUniform(meta, 1, meta.offset, ci);
        }
        const imgIdx = trainer.camMeta[ci].imgIdx;
        const rc = this.recon?.cams?.find((c) => c.imgIdx === imgIdx);
        if (rc) { rc.R = Array.from(R); rc.t = Array.from(t); }
      });
    } finally { if (wasTraining) this.start(); }
  }

  /** Manual growth pass, any time: one immediate refine (relocate the dead
   *  + grow into the highest-error donors) outside the schedule — the
   *  Prune/Grow button's other half. Pure growth: one refine, no cull —
   *  lastRefine untouched. At a full cap it asks the ceiling question
   *  instead of growing into nothing. */
  async growNow() {
    const trainer = this.trainer;
    if (!trainer) throw new Error('seed() first');
    const wasTraining = await this.pauseAndDrain();
    try {
      const r = await this._locked(() => trainer.refine());
      this._log(`grow-now @${trainer.iter}: relocated ${r.moved || 0}, grew +${r.grown || 0} -> ${r.n} splats`);
      this._em.emit('event', { kind: 'grow-now', iter: trainer.iter, ...r });
      if (r.capped && !this._capAsked) {
        this._capAsked = true;
        this.training = false;
        this._log(`splat cap reached at ${trainer.n} — paused: keep training at the cap, or prune and continue`);
        this._em.emit('event', { kind: 'cap-hit', iter: trainer.iter, splats: trainer.n });
      } else if (wasTraining) this.start();
      return r;
    } catch (e) {
      if (wasTraining) this.start();
      throw e;
    }
  }

  /** End the run early with the model as it stands: final metrics, then the
   *  same train-complete event an auto-stop emits. */
  async finish() {
    if (!this.trainer) return;
    this.training = false;
    this._log(`training finished early at ${this.trainer.iter} iterations`);
    await this._emitMetrics(true);
    await this._camDriftDone();
    this._em.emit('event', { kind: 'train-complete', iter: this.trainer.iter, splats: this.trainer.n });
  }

  _maxIters() { return this.opts.maxIters ?? 60000; }

  /** The GPU vanished under us (iOS reclaims WebGPU devices from backgrounded
   *  tabs; drivers reset). Training stops — the splats lived on the dead
   *  device — but frames and reconstruction are CPU-side, so recover() can
   *  rebuild and train again without redoing the solve. */
  _deviceLost(info) {
    if (this._lost) return;
    this._lost = true;
    this.training = false;
    this._fences = [];
    this._log(`GPU device lost (${(info && info.reason) || 'unknown'}) — ` +
      `${(info && info.message) || 'reclaimed by the system'}`);
    this._em.emit('event', { kind: 'device-lost', iter: this.trainer ? this.trainer.iter : 0 });
  }

  get deviceLost() { return !!this._lost; }

  /** The training worker faulted (uncaught exception in the worker realm).
   *  Like device loss the loop cannot continue — but unlike device loss the
   *  main-side device (if any) is fine, so this is its own event and the app
   *  reboots the worker through recover(). Pre-boot failures (this._remote
   *  still false) already propagate through seed()'s throw and stay quiet
   *  here, or a successful local fallback would flash a phantom error. */
  _workerFailed(err) {
    if (!this._remote || !this.trainer) return;
    if (this._lost) return;   // device-lost owns the narrative then
    this.training = false;
    const msg = (err && err.message) || String(err || 'worker error');
    this._log(`training worker failed (${msg})`);
    this._em.emit('event', { kind: 'worker-error', message: msg,
      iter: this.trainer ? this.trainer.iter : 0 });
  }

  /** A scheduled frame threw: the loop is dead (the throw skipped the
   *  reschedule at the end of _frameLoop) and training must NOT stay true —
   *  start() no-ops while it is, which wedged runs as "never starts, no
   *  error" (2026-11). Device loss and worker faults already emitted their
   *  own events and stopped the loop; anything else stops it HERE, loudly. */
  _frameFailed(e) {
    const msg = (e && e.message) || String(e);
    this._log(`frame loop: ${msg}`);
    if (this._lost || !this.training) return;
    this.training = false;
    this._em.emit('event', { kind: 'train-error', message: msg,
      iter: this.trainer ? this.trainer.iter : 0 });
  }

  /** Rebuild after device loss: a fresh device and trainer from the CPU-side
   *  reconstruction. Training restarts at iteration 0. */
  async recover() {
    if (!this.recon) throw new Error('nothing to recover — no reconstruction');
    this.training = false;
    this._lost = false;
    this._fences = [];
    this._ips = null;
    this._thrT = null;
    this._statsGap = 2000;
    this._lastStats = performance.now();
    this._metricsLock = null;    // readbacks on the dead device never resolve
    this._instantChecked = false;   // the fresh device re-proves its fences
    this._instantFences = false;
    this._instantSuspect = false;
    this._instantHits = 0;
    this._readOverrun = false;
    this._framePending = false;
    this.lossHistory = [];
    this.zlemaHistory = [];
    this._zlema = new FastRobustZLEMA();
    this._lastHold = null;
    this._lastHoldEval = 0;
    this._itersAtStats = 0;
    this._terminateWorker();
    this.trainer = null;
    if (this.gpu && this.gpu.owned) { try { this.gpu.dispose(); } catch { /* already gone */ } }
    this.gpu = null;
    return this.seed();
  }

  _ensureScheduler() {
    if (this._sched) return;
    // Hidden tabs AND occluded windows get their rAF throttled; worker
    // messages are not. The worker tick drives the loop whenever a scheduled
    // frame is >150ms late.
    let tickWorker = null;
    try {
      const src = 'setInterval(() => postMessage(0), 33);';
      tickWorker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    } catch { /* no Worker: rAF only */ }
    this._framePending = false;
    this._frameScheduledAt = 0;
    const runFrame = () => {
      if (!this._framePending) return;
      this._framePending = false;
      // stall watchdog: a frame that never completes (wedged worker,
      // hung readback, runaway backlog) must surface instead of sitting
      // at "Starting" forever — loud, repeating, and non-fatal (the
      // frame still owns the loop if it ever lands)
      const stallMs = this.opts.stallMs ?? 60000;
      const id = (this._frameSeq = (this._frameSeq || 0) + 1);
      const arm = () => {
        clearTimeout(this._stallT);
        this._stallT = setTimeout(() => {
          if (!this.training || this._frameSeq !== id) return;
          const tr = this.trainer;
          this._log(`training stall: a frame has run ${(stallMs / 1000) | 0}s+ without completing ` +
            `(iteration ${tr ? tr.iter : '?'})`);
          this._em.emit('event', { kind: 'train-stall', frameMs: stallMs, iter: tr ? tr.iter : 0 });
          arm();   // re-arm while the same frame stays stuck
        }, stallMs);
      };
      arm();
      // a readback rejecting mid-flight (device loss) must not become an
      // unhandled rejection — _frameFailed stops the loop and explains it
      const work = this._frameLoop();
      this._frameWork = work;
      work.then(
        () => { if (this._frameSeq === id) clearTimeout(this._stallT); },
        (e) => { if (this._frameSeq === id) clearTimeout(this._stallT); this._frameFailed(e); })
        .finally(() => { if (this._frameWork === work) this._frameWork = null; });
    };
    if (tickWorker) {
      tickWorker.onmessage = () => {
        if (this._framePending && this.training &&
            performance.now() - this._frameScheduledAt > 150) runFrame();
      };
    }
    // Background driver: a hidden page loses rAF entirely and its worker
    // timers throttle (1/s hidden, 1/min after five minutes) — the loop
    // would crawl, then freeze, on swipe-away. Message tasks are NOT
    // timer-throttled, so a hidden page pumps its frames through a channel
    // instead; the tick worker stays as the backstop for visible pages.
    let pump = null;
    if (typeof MessageChannel === 'function') {
      try {
        const ch = new MessageChannel();
        ch.port1.onmessage = () => runFrame();
        // Node-only: a listening port holds the event loop open — the pump
        // must never keep a process (or a test) alive by itself
        try { ch.port1.unref?.(); ch.port2.unref?.(); } catch { /* browsers have no unref */ }
        pump = () => ch.port2.postMessage(0);
      } catch { /* no channel: rAF plus the worker backstop, as before */ }
    }
    this._sched = {
      runFrame, tickWorker, pump,
      raf: typeof requestAnimationFrame === 'function'
        ? (fn) => requestAnimationFrame(fn) : (fn) => setTimeout(fn, 16),
    };
    this._frameCount = 0;
    this._lastStats = performance.now();
    this._itersAtStats = 0;
    this._lastHoldEval = 0;
  }

  _scheduleFrame() {
    this._framePending = true;
    this._frameScheduledAt = performance.now();
    // hidden (swiped-away, background tab, screen off): rAF never fires and
    // the worker tick throttles — pump through the channel instead. Decided
    // per frame, so hiding or showing mid-run just switches drivers.
    if (isHidden() && this._sched.pump) this._sched.pump();
    else this._sched.raf(this._sched.runFrame);
  }

  /** Effective GPU usage: the governor's auto duty capped by the slider.
   *  The slider is the thermal ceiling ("how hot"); auto only ever derates
   *  below it. Disabled (opts.autoDuty false) or hidden: the slider as-is. */
  _effUsage() {
    const slider = gpuDuty(this.opts.gpuUsage);
    if (this.opts.autoDuty === false || isHidden()) return slider;
    return Math.min(slider, this._govAuto);
  }

  /** One event-loop lag sample per frame: a setTimeout(0) measures how long
   *  the main thread takes to turn (encode bursts, GC, refine planning). */
  _govProbe() {
    const p0 = performance.now();
    setTimeout(() => this._govNoteLag(performance.now() - p0), 0);
  }

  _govNoteLag(ms) {
    this._govLag = this._govLag * 0.7 + Math.min(ms, 500) * 0.3;
  }

  /** rAF sampler tick (own loop, not the training cadence): inter-arrival
   *  time is what the user feels as animation smoothness. */
  _govNoteRaf(now) {
    const dt = now - (this._govRafLast || now);
    this._govRafLast = now;
    if (dt > 0 && dt < 1000) this._govRaf = this._govRaf * 0.8 + Math.min(dt, 250) * 0.2;
  }

  /** Plan item (3): feed the batch GPU timer into the governor. ms/step
   *  at a fixed splat count is unambiguous — growth only lands at refine
   *  boundaries (the baseline re-anchors there), so a sustained rise
   *  between refines is the GPU slowing under its own heat or contention:
   *  the ceiling shrinks; a return toward the best lets it grow back. A
   *  null gpuMs (timerless trainers) leaves the state untouched — ratio 1,
   *  the batch-A law. Called with the pre-refine count the steps ran with. */
  _govNoteGpu(gpuMs, batch, n) {
    if (!Number.isFinite(gpuMs) || gpuMs <= 0 || !(batch > 0)) return;
    const sample = gpuMs / batch;
    if (n !== this._govN) {
      this._govMsStep = sample;
      this._govMsBase = sample;
      this._govN = n;
      return;
    }
    this._govMsStep = this._govMsStep * 0.7 + sample * 0.3;
    if (this._govMsStep < this._govMsBase) this._govMsBase = this._govMsStep;
  }

  /** Current GPU slowdown vs its best this refine-window (1 = healthy). */
  _govGpuRatio() {
    return this._govMsBase > 0 ? this._govMsStep / this._govMsBase : 1;
  }

  /** Instant-fence detector (timerless loops only — a batch timer makes
   *  fences irrelevant). Some phone drivers resolve onSubmittedWorkDone
   *  before the work completes: dt then measures CPU encode, the batch
   *  ramps to 64 on fake speed, and the queue grows unboundedly while the
   *  notification sits at "Starting" (2026-11, ~10-image hangs). A gated
   *  fence settling in <5ms with real work submitted is meaningless; three
   *  in a row engages drain pacing (every frame readback-drains, so wall
   *  dt turns honest and the adapter self-regulates). Any healthy fence
   *  (>=5ms) or any timed batch concludes the check the other way.
   *  fenceMs null = nothing gated this frame (no signal either way). */
  _noteFence(fenceMs, batch, gpuMs) {
    if (this._instantChecked) return;
    if (gpuMs != null) { this._instantChecked = true; return; }
    if (fenceMs == null) return;
    if (fenceMs >= 5) {
      this._instantChecked = true;
      this._instantHits = 0;
      return;
    }
    if (!(batch >= 4)) return;   // trivial work settles fast legitimately
    this._instantHits = (this._instantHits || 0) + 1;
    this._instantSuspect = true;
    if (this._instantHits >= 3) {
      this._instantChecked = true;
      this._instantFences = true;
      this._log('fences resolve before the GPU drains (instant fences) — ' +
        'pacing each frame on a readback drain instead of queue depth');
    }
  }

  /** Ideal next batch for the measured cadence. Growth needs the honest
   *  timer: the first null batches (the timer delivers one batch lagged)
   *  hold-or-shrink on wall dt, because stale fences fake speed and would
   *  spike the batch into its clamp; permanently timerless trainers keep
   *  the legacy wall-dt law after the grace batches. */
  _batchIdeal(batch, paceMs, gpuMs) {
    if (gpuMs == null) this._gpuMsMisses = (this._gpuMsMisses || 0) + 1;
    else this._gpuMsMisses = 0;
    const ideal = batch * (120 / paceMs);
    return gpuMs == null && this._gpuMsMisses <= 2 ? Math.min(batch, ideal) : ideal;
  }

  /** Plan item (4): submit cut when the queued GPU work exceeds the phone
   *  budget (opts.gpuChunkMs). The fence ring caps outstanding SUBMITS, but
   *  on a phone each batch can hold hundreds of ms of GPU work — this caps
   *  the outstanding WORK instead: predict the batch from the governor's
   *  ms/step, correct the last submit with its measured gpuMs, and drain
   *  with the wall clock (stale fences can't serialize the loop — the cut
   *  only bites when the GPU is genuinely behind). Returns the ms to idle
   *  instead of submitting, 0 to submit. Without the batch timer there is
   *  no estimate and the loop runs uncut; the wall drain guarantees
   *  progress (worst case one batch per budget window), so no fail-open. */
  _submitCutMs(batch) {
    const now = performance.now();
    if (this._cutT == null) this._cutT = now;
    this._cutMs = Math.max(0, this._cutMs - (now - this._cutT));
    this._cutT = now;
    if (this._cutPending) {
      if (Number.isFinite(this._lastGpuMs)) {
        this._cutMs = Math.max(0, this._cutMs + (this._lastGpuMs - this._cutPred));
      }
      this._cutPending = false;
    }
    const pred = this._govMsStep * batch;
    const excess = this._cutMs + pred - (this.opts.gpuChunkMs ?? 400);
    // no timer signal yet (or timerless): run uncut but keep accounting, so
    // the first measurement corrects the estimate retroactively. And the
    // floor batch always submits: a minimum batch over budget must still
    // run (slowly, honestly) — cutting it idles forever, since the cut
    // branch never re-estimates (2026-11: large-res runs hung exactly this
    // way — completing frames, responsive UI, frozen metrics, and a stall
    // watchdog with nothing to catch).
    if (this._govN < 0 || batch <= 4 || excess <= 0) {
      this._cutMs += pred;
      this._cutPred = pred;
      this._cutPending = true;
      return 0;
    }
    return Math.min(100, excess);
  }

  _govRafLoop() {
    if (!this.training) { this._govRafRunning = false; return; }
    this._govNoteRaf(performance.now());
    this._sched.raf(() => this._govRafLoop());
  }

  /** Run the controller over the EWMAs (stats-window cadence); frozen while
   *  hidden, where there is no responsiveness to protect. Logs only when the
   *  derate crosses a bucket boundary. */
  _govUpdate() {
    if (this.opts.autoDuty === false || isHidden()) return;
    const g = { ...GOV_DEFAULTS, ...this.opts.governor };
    const ratio = this._govGpuRatio();
    const { auto, moved } = governDuty(this._govAuto,
      { lagMs: this._govLag, rafMs: this._govRaf, gpuRatio: ratio }, this.opts.governor);
    this._govAuto = auto;
    if (moved !== 0) {
      const bucket = auto >= 1 ? 1 : auto > 0.5 ? 0.5 : auto > 0.25 ? 0.25 : 0.1;
      if (bucket !== this._govBucket) {
        this._govBucket = bucket;
        const gpuTrip = ratio > g.gpuJank;
        const sysTrip = this._govLag > g.lagJank || this._govRaf > g.rafJank;
        this._log(auto >= 1
          ? 'system healthy — GPU duty back to the slider'
          : `${gpuTrip && !sysTrip ? 'GPU throttling' : 'display struggling'} (event-loop lag ` +
            `${this._govLag.toFixed(0)}ms, frame ${this._govRaf.toFixed(0)}ms` +
            (gpuTrip ? `, GPU step cost ${ratio.toFixed(2)}x its best` : '') +
            `) — GPU duty auto-derated to ${Math.round(auto * 100)}% of the slider`);
      }
    }
  }

  async _frameLoop() {
    const trainer = this.trainer;
    if (!trainer || !trainer.camMeta || this._lost) return;
    // ready === false is a half-booted trainer (setup threw after camMeta,
    // before the buffers): fail LOUD here instead of throwing the GPU's
    // cryptic TypeErrors mid-frame (2026-11). Proxies and mocks leave ready
    // undefined, which passes.
    if (trainer.ready === false) {
      throw new Error('trainer setup did not complete — the seed/reseed failed partway; reload or recover to reboot cleanly');
    }
    this._frameCount++;

    if (this.training && trainer.iter >= this._maxIters()) {
      this.training = false;
      this._log(`training complete at ${trainer.iter} iterations`);
      await this._emitMetrics(true);
      // emitted AFTER the final readback: listeners typically call metrics()
      // right away, which must not interleave with ours on the staging buffer
      await this._camDriftDone();
      this._em.emit('event', { kind: 'train-complete', iter: trainer.iter, splats: trainer.n });
    }

    if (this.training) {
      // Batch enough iterations per frame to keep the GPU busy. The batch
      // adapts to the device: ~120ms of GPU work per frame keeps a desktop
      // GPU saturated with few submits AND keeps metrics/UI alive on a phone
      // where the same 15 iterations could take seconds.
      let batch = this.opts.itersPerFrame ??
        Math.max(4, Math.min(64, Math.round(this._batch ?? 15)));
      // timerless + pacing suspect: hold small batches until the fence
      // check concludes (a healthy fence or a timed batch lifts this
      // within a few frames; instant fences keep it, plus drain pacing)
      if (this._lastGpuMs == null && !this._instantChecked &&
          (this._instantSuspect || this._readOverrun)) {
        batch = Math.min(batch, 8);
      }
      // (4) over budget: skip the submit, keep the view alive, idle a beat
      // for the queue to drain — then re-evaluate next frame
      const cutMs = this._submitCutMs(batch);
      if (cutMs > 0) {
        if (this.perf) {
          this.perf.marks.push({ t: Math.round(performance.now()), kind: 'cut',
            iter: trainer.iter, ms: Math.round(cutMs) });
        }
        // the cut must ESCAPE, not just idle: the adapter only re-estimates
        // in the work branch, so a frozen batch over budget would cut
        // forever — the adapter never shrinks what never runs (same hang as
        // above, at any batch). Decay toward the floor, whose submit (see
        // _submitCutMs) guarantees progress, and say so while it bites.
        this._cutStreak = (this._cutStreak || 0) + 1;
        const cutFrom = Math.round(this._batch ?? batch);
        this._batch = Math.max(4, (this._batch ?? batch) * 0.7);
        if (this._cutStreak === 1 || this._cutStreak % 50 === 0) {
          this._log(`submit cut: holding ${Math.round(cutMs)}ms idle ` +
            `(batch ${cutFrom}→${Math.round(this._batch)}, streak ${this._cutStreak}) — GPU work over budget`);
        }
        this.view._tick(this._frameCount, this.training);
        await new Promise((r) => setTimeout(r, cutMs));
        if (this.training || (this.view.enabled && this.view._dirty)) this._scheduleFrame();
        return;
      }
      this._cutStreak = 0;
      const t0 = performance.now();
      this._govProbe(); // duty governor: one event-loop lag sample per frame
      // true throughput, stall time included: Safari's stale fences hide the
      // GPU's real pace from per-frame dt, and a slow GPU otherwise ends up
      // with many seconds of work queued (chunky metrics, laggy pause)
      if (this._thrT == null) {
        this._thrT = t0; this._thrIter = trainer.iter;
      } else if (t0 - this._thrT > 1000) {
        const ips = (trainer.iter - this._thrIter) / (t0 - this._thrT) * 1000;
        this._ips = this._ips == null ? ips : this._ips * 0.5 + ips * 0.5;
        this._thrT = t0; this._thrIter = trainer.iter;
      }
      // (2) the batch GPU timer brackets the steps; the adapter below paces
      // on measured GPU-busy ms, falling back to wall dt without timestamps
      let gpuMs = null, fenceMs = null;
      if (this._remote) {
        const s = await trainer.runSteps(batch);
        gpuMs = s.gpuMs ?? null;
        fenceMs = s.fenceMs ?? null;
      } else {
        if (typeof trainer.beginGpuBatch === 'function') trainer.beginGpuBatch();
        for (let k = 0; k < batch; k++) trainer.stepOnce();
        if (typeof trainer.endGpuBatch === 'function') gpuMs = await trainer.endGpuBatch();
      }
      // (3) the ceiling's own signal: reads before refine() below can move n
      this._govNoteGpu(gpuMs, batch, trainer.n);
      this._lastGpuMs = gpuMs; // (4) corrects the submit prediction next frame

      const tEnc = performance.now() - t0;

      // Score-driven refinement: ordinary filtered-score plateaus and the
      // rising raw variance of an under-capacity plateau are both eligible.
      // Evidence is reset at each growth, then a complete fresh window must
      // accumulate. There is no fixed cadence and no ever-growing delay.
      const sinceRefine = trainer.iter - (trainer.lastRefine || 0);
      const plateauWindow = this.opts.plateauWindow ?? 1200;
      const convergenceGap = this.opts.plateauMinGap ?? plateauWindow;
      const currentScore = this.zlemaHistory.length ? this.zlemaHistory.at(-1)[1] : null;
      const recovered = this._refineRecoveryTarget == null ||
        (Number.isFinite(currentScore) && currentScore >= this._refineRecoveryTarget - (this.opts.plateauRecoverySlack ?? 0.05));
      if (recovered && this._refineRecoveryTarget != null) this._refineRecoveryTarget = null;
      let plateau = false;
      if (this.opts.refinePlateau !== false && trainer.iter >= plateauWindow && recovered &&
          sinceRefine >= convergenceGap) {
        plateau = scorePlateaued(this.zlemaHistory, trainer.iter, {
          window: plateauWindow,
          slope: this.opts.plateauSlope ?? 0.1,
          noisySlope: this.opts.plateauNoisySlope ?? 0.35,
          rawHistory: this.lossHistory,
          noiseRatio: this.opts.plateauNoiseRatio ?? 1.35,
          noiseFloor: this.opts.plateauNoiseFloor ?? 0.08,
          after: trainer.lastRefine || 0,
        });
      }
      let refineResult = null;
      if (trainer.iter >= plateauWindow && plateau) {
        // Remember quality before mutation. The next refine cannot happen on
        // the low post-grow plateau; quality must first recover to this mark.
        this._refineRecoveryTarget = Number.isFinite(currentScore) ? currentScore : null;
        trainer.lastRefine = trainer.iter;
        // tandem prune: cull the lowest-mass fraction FIRST (until 85% of
        // the run), so this same refine regrows the freed capacity into
        // the highest-error donors — one prune/multiply cycle, not two
        // cadences. Survivors keep their accumulated error stats (the
        // compaction moves grad buffers with their rows), so the growth
        // donors below are scored on the same window.
        const sp = trainer.opts ? trainer.opts.softPrune : null;
        let tandemCulled = 0;
        if (sp && trainer.iter <= (sp.until ?? 0.85) * this._maxIters()) {
          const pruned = await trainer.softPrune();
          if (pruned.culled) {
            tandemCulled = pruned.culled;
            this._log(`tandem prune @${trainer.iter}: culled ${pruned.culled} -> ${pruned.n} splats` +
              (pruned.floorBound === 'ratchet' ? ' (growth ratchet held the floor)' : ''));
            this._em.emit('event', { kind: 'soft-prune', iter: trainer.iter, ...pruned });
          }
        } else if (!sp && trainer.iter <= 0.85 * this._maxIters()) {
          // soft-prune off: no ratio trims, but vote-confirmed dead splats
          // (zero rendered mass two windows running) still go — obstructed
          // rows are pure overhead, never signal
          const dead = await trainer.pruneDead();
          if (dead.culled) {
            this._log(`dead cull @${trainer.iter}: culled ${dead.culled} -> ${dead.n} splats`);
            this._em.emit('event', { kind: 'dead-cull', iter: trainer.iter, ...dead });
          }
        }
        const r0 = performance.now();
        // awaited: refine reads six buffers and writes them all back — steps
        // submitted meanwhile would be silently rewound by that write-back,
        // and the six snapshots would come from six different iterations
        const r = await trainer.refine();
        refineResult = r;
        if (this.perf) {
          this.perf.marks.push({ t: Math.round(r0), kind: 'refine', iter: trainer.iter,
            ms: Math.round(performance.now() - r0), moved: r.moved, grown: r.grown });
        }
        if (r.moved || r.grown) {
          this._log(`refine @${trainer.iter}: relocated ${r.moved}, grew +${r.grown} -> ${r.n} splats`
            + (r.dead != null ? ` (dead ${r.dead}, last round ${r.survived}/${r.lastReloc} survived)` : '')
            + (plateau ? ' [score plateaued]' : '')
            + (tandemCulled ? ` [tandem: culled ${tandemCulled} first]` : ''));
          this._em.emit('event', { kind: 'refine', iter: trainer.iter, ...r });
        }
      }

      // the ceiling's own prompt: a refine reporting growth-still-scheduled
      // against a full TRUE cap pauses and asks — keep training at the cap,
      // or prune to half and grow on. The UI answers via start() (growth
      // stays blocked by the limit) or pruneAndContinue() (which resets
      // _capAsked, so the next ceiling hit asks again).
      if (refineResult && refineResult.capped && !this._capAsked) {
        this._capAsked = true;
        this.training = false;
        this._log(`splat cap reached at ${trainer.n} — paused: keep training at the cap, or prune and continue`);
        this._em.emit('event', { kind: 'cap-hit', iter: trainer.iter, splats: trainer.n });
      }

      const now = performance.now();
      let tMet = 0;
      if (now - this._lastStats > (this._statsGap ?? 2000)) {
        const m0 = performance.now();
        await this._emitMetrics();
        tMet = performance.now() - m0;
        this._govUpdate(); // duty governor runs on the stats cadence
      }

      const v0 = performance.now();
      this.view._tick(this._frameCount, this.training);
      const tView = performance.now() - v0;

      // Deep pipelining: keep a RING of fences in flight, not one. Safari
      // resolves onSubmittedWorkDone hundreds of ms late even when the GPU is
      // idle; gating each batch on a single fence made that latency the loop
      // period, and the batch adapter — reading the latency as GPU time —
      // shrank the batch into its floor (an iPhone sat at ~15 it/s on a scene
      // it can train at hundreds). With 4 fences outstanding the late fences
      // overlap; the frame period becomes ~latency/4 and the same adapter now
      // GROWS the batch until real GPU work dominates. Prompt-fence devices
      // (desktop) behave as before.
      // remote training gates its own fence ring worker-side and the round
      // trip above already carries the backpressure — the local ring is local-only
      let tStall = 0;
      if (!this._remote) {
        this._fences.push(trainer.device.queue.onSubmittedWorkDone());
        // fence ring depth trades UI latency for fence-latency overlap
        // (4 x 0.4s of queued dispatches is the "whole system stalls" feel —
        // hosts sharing the GPU with a compositor can pass a lower fenceRing)
        if (this._fences.length > (this.opts.fenceRing ?? 4)) {
          const s0 = performance.now();
          await this._fences.shift();
          tStall = performance.now() - s0;
          fenceMs = tStall;
        }
      }

      // instant-fence check (timerless loops): three meaningless fences
      // engage drain pacing below; the batch cap above holds the ramp
      // until the check concludes either way
      this._noteFence(fenceMs, batch, gpuMs);
      if (this._lastGpuMs == null && (this._instantFences || this._readOverrun)) {
        // drain pacing: without timestamps the readback is the ONLY honest
        // completion signal (mapAsync waits for preceding work) — the wall
        // dt below then measures real GPU time and the adapter self-regulates
        // instead of queueing backlog behind instant fences
        await this._locked(() => trainer.readLoss());
      }

      // Duty honesty: the ring above keeps several batches in flight, so
      // the throttle sleep below would merely run the backlog while the
      // GPU never frees up (10% delivered ~100% heat). Drain the in-flight
      // queue first so the sleep is genuinely GPU-free — bounded by a
      // quarter of the coming idle, so a late fence (Safari resolves
      // hundreds of ms after the GPU is already idle) fails open to
      // pipelined behavior instead of stalling every frame. Skipped
      // entirely at full duty (duty 1 never sleeps: provable no-op).
      let dt = Math.max(5, performance.now() - t0);
      const duty = gpuDuty(this._effUsage());
      if (duty < 1 && this._fences.length) {
        const budget = Math.max(0, gpuIdleMs(dt, this._effUsage()) * 0.25);
        const w0 = performance.now();
        const drained = await Promise.race([
          this._fences[this._fences.length - 1].then(() => true, () => true),
          new Promise((r) => setTimeout(() => r(false), budget)),
        ]);
        dt = Math.max(5, dt + (performance.now() - w0));
        if (drained) this._fences.length = 0;
      }

      // adapt the batch to the measured cadence: honest GPU-busy ms when the
      // batch timer measured it (fence latency, throttle sleeps and encode
      // gaps excluded — thermal sag shows as rising ms and shrinks the batch
      // instead of queueing backlog), wall dt otherwise; damped so it settles
      const paceMs = Number.isFinite(gpuMs) && gpuMs > 0 ? gpuMs : dt;
      const ideal = this._batchIdeal(batch, paceMs, gpuMs);
      this._batch = Math.max(4, Math.min(64, (this._batch ?? 15) * 0.7 + ideal * 0.3));
      // throughput cap: ~0.4s of measured GPU work per batch. The dt adapter
      // reads 5ms frames when fences resolve stale and pushes the batch to
      // the clamp — fine on a desktop, ten queued seconds on a phone.
      if (this._ips != null) {
        // gpuChunkMs bounds how much GPU work one batch queues at full
        // duty; below, the bound scales with the slider (gpuBurstMs) so a
        // burst never outruns its idle window as backlog
        const chunk = gpuBurstMs(this.opts.gpuChunkMs ?? 400, this._effUsage()) / 1000;
        // _ips is wall-time (throttle idle included): divide the requested
        // duty back out against the duty-scaled bound, so batches stay
        // saturating at full duty and shrink with the slider below
        this._batch = Math.min(this._batch, Math.max(8, Math.round(this._ips * chunk / gpuDuty(this._effUsage()))));
      }
      if (this.perf) {
        this.perf.frames.push([Math.round(now), trainer.iter, batch, trainer.n,
          +tEnc.toFixed(1), +tView.toFixed(1), +tStall.toFixed(1), +tMet.toFixed(1), +dt.toFixed(1)]);
      }
      // GPU-usage throttle (the slider): duty-cycle idle AFTER the full
      // batch — the adapters above keep every submitted batch at saturating
      // size, and the pause between batches is what yields the GPU to the
      // system. Read live so the slider bites on the next batch; before the
      // reschedule check so a pause during the sleep still stops the loop.
      const idleMs = gpuIdleMs(dt, this._effUsage());
      if (idleMs > 0) await new Promise((r) => setTimeout(r, idleMs));
    } else {
      this.view._tick(this._frameCount, this.training);
      if (this._remote) await trainer.sync();
      else await trainer.device.queue.onSubmittedWorkDone();
      this._fences.length = 0;
    }
    if (this.training || (this.view.enabled && this.view._dirty)) this._scheduleFrame();
  }

  /** serialize every metric readback: they share the trainer's staging
   *  buffers, and two in flight = "outstanding map pending". On a lost
   *  device readbacks would hang forever — fail fast instead. */
  _locked(fn) {
    const call = () => {
      if (this._lost) throw new Error('GPU device lost');
      return fn();
    };
    const run = (this._metricsLock || Promise.resolve()).then(call, call);
    this._metricsLock = run.catch(() => {});
    return run;
  }

  _emitMetrics(final = false) {
    return this._locked(() => this._emitMetricsInner(final));
  }

  async _emitMetricsInner(final = false) {
    const trainer = this.trainer;
    const now = performance.now();
    const itersPerSec = (trainer.iter - this._itersAtStats) / Math.max(1, now - this._lastStats) * 1000;
    const m = { iter: trainer.iter, splats: trainer.n, itersPerSec: Math.round(itersPerSec),
      duty: +this._effUsage().toFixed(3), autoDuty: +this._govAuto.toFixed(3),
      gpuRatio: +this._govGpuRatio().toFixed(3) };
    this._lastIps = m.itersPerSec;
    const tRead = performance.now();
    // breathe first: the readback drains every queued batch, so a pending
    // frame present goes out now instead of queueing behind the stall
    await tick();
    const mse = await trainer.readLoss();
    if (mse != null && mse > 0) {
      m.psnrTrain = -10 * Math.log10(mse);
      this.lossHistory.push([trainer.iter, m.psnrTrain]);
      // the trigger's history shadows the graph's: past the cold-start
      // hold, every raw readout also feeds the shared filter (one update
      // per push — a causal pass, identical to the graph's fresh series)
      if (this.lossHistory.length > SKIP_FIRST) {
        this.zlemaHistory.push([trainer.iter, this._zlema.update(m.psnrTrain)]);
      }
    }
    // tile-sort overflow used to be silent (dropped tiles render as
    // background): the count is cumulative on the trainer in both the
    // local and worker paths, so any increase is news, once per increase
    const ov = trainer.entryOverflowTiles || 0;
    if (ov > (this._lastOverflow || 0)) {
      this._lastOverflow = ov;
      this._log(`WARNING: tile-sort overflow — ${ov} tiles dropped so far; the splat budget is too large for this adapter's entry binding`);
      this._em.emit('event', { kind: 'overflow', iter: trainer.iter, tiles: ov });
    }
    const holdEvery = this.opts.evalHoldEvery ?? 4000;
    // (7) hidden tabs skip the holdout eval (training continues); the stale
    // _lastHoldEval fires it immediately when the tab returns. Final always
    // evaluates — the done screen needs its number.
    // A deep queue drain already stalled this readout — stacking the
    // holdout's extra full-frame render + readbacks onto it turns one hitch
    // into two, so the holdout defers to the next readout instead. The
    // cadence is iter-based and _lastHoldEval only advances on an actual
    // eval, so the deferred eval fires there; meanwhile the stale-score
    // branch below keeps the last known value on screen.
    const drainDefer = !final && (performance.now() - tRead) > 1500;
    if (this.holdout >= 0 &&
        (final || (!isHidden() && !drainDefer && trainer.iter - this._lastHoldEval >= holdEvery))) {
      this._lastHoldEval = trainer.iter;
      m.psnrHold = await trainer.evalCamPsnr(this.holdout);
      this._lastHold = m.psnrHold;
    } else if (this._lastHold != null) {
      m.psnrHold = this._lastHold;
    }
    // The readbacks above drain every queued batch first — seconds on a slow
    // GPU. Pace the next readout from COMPLETION and by what this one cost,
    // or a fixed 2s cadence turns into back-to-back queue drains.
    const end = performance.now();
    this._lastStats = end;
    this._itersAtStats = trainer.iter;
    this._statsGap = Math.min(8000, Math.max(2000, (end - tRead) * 2.5));
    // a multi-second drain means the loop submitted faster than the GPU
    // drains — the backlog instant fences hide everywhere else. Without a
    // batch timer there is nothing else to pace off, so stay conservative
    // (small batches + drain pacing). Sticky: one overrun proves the
    // pacing signal dishonest for this run.
    this._lastReadMs = end - tRead;
    if (this._lastReadMs > 3000 && this._lastGpuMs == null && !this._readOverrun) {
      this._readOverrun = true;
      this._log(`metrics readback drained ${(this._lastReadMs / 1000).toFixed(1)}s of queued GPU work — ` +
        'holding small batches + drain pacing (no batch timer to pace off)');
    }
    this._em.emit('metrics', m);
    return m;
  }

  /** One-shot quality readout (used by tests and the done screen). */
  metrics({ refined = false } = {}) {
    return this._locked(async () => {
      const trainer = this.trainer;
      const out = { iter: trainer.iter, splats: trainer.n };
      if (this.holdout >= 0) {
        out.psnrHold = await trainer.evalCamPsnr(this.holdout);
        if (refined) out.psnrHoldRefined = await trainer.evalCamPsnrRefined(this.holdout);
      }
      return out;
    });
  }

  /** PSNR of one training camera against its own photograph (serialized with
   *  the other metric readbacks; safe to call while training). */
  evalFramePsnr(ci) {
    return this._locked(() => this.trainer.evalCamPsnr(ci));
  }

  /** PSNR and true Gaussian-window SSIM for one training frame. */
  evalFrameMetrics(ci) {
    return this._locked(() => this.trainer.evalCamMetrics(ci));
  }

  /** Paired train/holdout samples for the graph, serialized as one readback job. */
  evalGraphMetrics(trainCi, holdCi = -1) {
    return this._locked(async () => {
      const train = await this.trainer.evalCamMetrics(trainCi);
      const hold = holdCi >= 0 ? await this.trainer.evalCamMetrics(holdCi) : null;
      return { train, hold };
    });
  }

  /** Mean PSNR over the evalSplit test cameras — the number quality papers
   *  report. Resolves null when no eval split was requested. */
  evalTestPsnr() {
    if (!this.testCams || !this.testCams.length) return Promise.resolve(null);
    if (this._remote) return this._locked(() => this.trainer.evalTestPsnr(this.testCams));
    return this._locked(async () => {
      const frames = [];
      let sum = 0;
      for (const c of this.testCams) {
        const psnr = await this.trainer.evalCamPsnr(c);
        frames.push({ cam: c, imgIdx: this.trainer.camMeta[c].imgIdx, psnr });
        sum += psnr;
      }
      return { psnr: sum / frames.length, frames };
    });
  }

  /** Index of the camera the trainer most recently stepped on (UI pulse). */
  get activeCam() { return this.trainer ? this.trainer.lastCam : -1; }

  /** Standard 3DGS .ply with Mip opacity compensation baked (what external
   *  sorted viewers expect). */
  exportPlyBlob() {
    // shares the trainer's staging buffers with the metric readbacks
    return this._locked(async () => {
      const { data, n, sh, shK } = await this.trainer.readGaussians();
      return this._bakedPly(data, n, sh, shK);
    });
  }

  /** Simplified .ply: training-free NanoGS-method merge-down to `ratio` of
   *  the splats, then the same bake + dead-splat filter as a full export.
   *  The live model is untouched — the simplification runs on a readback. */
  exportSimplifiedPlyBlob(ratio, { onStatus = () => {}, onProgress = null } = {}) {
    return this._locked(async () => {
      const { data, n, sh, shK } = await this.trainer.readGaussians();
      const simp = await simplifyGaussians(
        { data, n, sh, shK, dc: this.trainer.dcMode },
        { ratio, onStatus, onProgress });
      return this._bakedPly(simp.data, simp.n, simp.sh, simp.shK);
    });
  }

  /** Mip-compensation bake + dead-splat filter + PLY pack, shared by the
   *  full and simplified export paths. */
  _bakedPly(data, n, sh, shK) {
    const meta = this.trainer.camMeta[0];
    // a view-only restore has no training cameras — its model came from an
    // export that already baked the compensation, so pass it through.
    // A trainer without Mip compensation (opts.mipComp false) rendered
    // exactly what external rasterizers render: nothing to bake.
    const baked = (meta && this.trainer.mipComp !== false)
      ? bakeOpacityCompensation(data, n, meta.f,
          Float32Array.from(this.trainer.camMeta.flatMap(camPosition)), this.trainer.dilate)
      : data;
    // dead splats — invisible at the smallest 8-bit alpha — are pure
    // file-size and render tax for every viewer. Long classic-era runs
    // accumulated 50%+ of them (opacity-reg ratchet, lab log 2026-09-01):
    // drop them at the door.
    const A_DEAD = Math.log(1 / 254);   // logit of alpha 1/255
    let live = 0;
    for (let i = 0; i < n; i++) if (baked[i * 16 + 13] > A_DEAD) live++;
    let oData = baked, oSh = sh, oN = n;
    if (live < n) {
      oData = new Float32Array(live * 16);
      const shPer = sh ? sh.length / n : 0;
      oSh = sh ? new Float32Array(live * shPer) : sh;
      let w = 0;
      for (let i = 0; i < n; i++) {
        if (baked[i * 16 + 13] <= A_DEAD) continue;
        oData.set(baked.subarray(i * 16, i * 16 + 16), w * 16);
        if (sh) oSh.set(sh.subarray(i * shPer, (i + 1) * shPer), w * shPer);
        w++;
      }
      oN = live;
    }
    return gaussiansToPly(oData, oN, oSh, shK, this.trainer.dcMode);
  }

  /** The trainer's RAW parameter state ({ data, n, sh, shK }) — the exact
   *  floats, no opacity baking or color activation: what seedFrom() takes
   *  back for a bit-identical resume. */
  exportRawState() {
    return this._locked(() => this.trainer.readGaussians());
  }

  dispose() {
    this.training = false;
    clearTimeout(this._stallT);
    if (this._sched && this._sched.tickWorker) this._sched.tickWorker.terminate();
    if (this.gpu && this.gpu.owned) this.gpu.dispose();
    this._terminateWorker();
    this.trainer = null;
  }
}

/** The session's render target: one canvas, one camera. */
class SessionView {
  constructor(session) {
    this.s = session;
    this.canvas = null;
    this.ctx = null;
    this.camera = null;   // { R, t, f, cx, cy, w, h } at canvas resolution
    this.enabled = true;  // app viewport is optional; the trainer's own loss renderer is not
    this._dirty = false;
    this._renderInFlight = null;   // latest-wins gate (see renderNow)
    this._offset = 0;     // training-target offset when looking through a frame
    this._lookIdx = null; // frame index when looking through a frame (else the free camera holds)
    this._offs = new Map();  // canvas element -> { off, key, gen, sent } (remote only)
    this._remoteInFlight = false; // latest-wins gate for remote sends (see renderNow)
    this._chain = null;      // serializes remote view commands (attach before render)
    this._viewFailed = false; // a view-error event is already out for this failure streak
    this._falseSince = null;  // first consecutive renderNow=false (no-present streak clock)
  }

  /** Remote view commands run strictly ordered — overlapping ticks must not
   *  reorder attach/setCamera/renderNow across the round trips. A rejected
   *  command surfaces once per streak as a view-error event (the app
   *  re-mounts a fresh canvas on it); renderNow resolving false (disabled,
   *  in-flight, no camera yet) is normal flow, not an error. */
  _send(cmd, params, transfer) {
    const run = () => this.s.trainer.viewCmd(cmd, params, transfer);
    const p = (this._chain || Promise.resolve()).then(run, run);
    this._chain = p.catch(() => {});
    p.then(
      () => { this._viewFailed = false; },
      (e) => {
        if (this._viewFailed) return;
        this._viewFailed = true;
        const msg = (e && e.message) || String(e);
        this.s._log(`view ${cmd} failed: ${msg}`);
        this.s._em.emit('event', { kind: 'view-error', cmd, message: msg });
      });
    return p;
  }

  attach(canvas) {
    if (!this.s._remote) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('webgpu');
      this.ctx.configure({
        device: this.s.trainer.device,
        format: this.s.trainer.canvasFormat,
        alphaMode: 'opaque',
      });
      this._dirty = true;
      if (!this.s.training && this.enabled) this.s._ensureScheduler(), this.s._scheduleFrame();
      return;
    }
    let rec = this._offs.get(canvas);
    if (!rec) {
      rec = { off: canvas.transferControlToOffscreen(), key: this._offs.size,
              gen: this.s._workerGen, sent: false };
      this._offs.set(canvas, rec);
    }
    if (rec.gen !== this.s._workerGen) {
      // the element's offscreen died with a previous worker (recover); a
      // transferred element can never move again — attach a fresh canvas
      throw new Error('view: this canvas was transferred to a previous worker — attach a fresh canvas');
    }
    this.canvas = canvas;
    const first = !rec.sent;
    rec.sent = true;
    this._dirty = true;
    const p = this._send('attach',
      { key: rec.key, canvas: first ? rec.off : undefined, w: canvas.width, h: canvas.height },
      first ? [rec.off] : undefined);
    // a failed attach must not poison the element: sent=true would make
    // every later attach omit the canvas and the worker answer 'unknown
    // canvas key' forever — a permanent black view from one transient
    // rejection (2026-11). The next attach retries with the canvas.
    p.catch(() => { if (this._offs.get(canvas) === rec) rec.sent = false; });
    if (!this.s.training && this.enabled) this.s._ensureScheduler(), this.s._scheduleFrame();
    return p;
  }

  /** Re-push the live view state (enabled + camera) behind the queued
   *  commands. After a re-attach the worker may have missed any of it —
   *  rejections don't retry — while the main side looks unchanged, so plain
   *  setters would early-return and the desync would stick. */
  resync() {
    if (!this.s._remote || !this.canvas) return Promise.resolve(false);
    this._viewFailed = false;   // the re-attach starts a fresh failure streak
    this._send('setEnabled', { enabled: this.enabled });
    if (this._lookIdx != null) this._send('lookThrough', { i: this._lookIdx });
    else if (this.camera) this._send('setCamera', { cam: this.camera });
    return Promise.resolve(true);
  }

  /** Enable/disable the app's standalone viewport without touching trainer
   *  steps, training renders, losses, or metric readbacks. */
  setEnabled(enabled) {
    enabled = !!enabled;
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    this._dirty = true; // repaint the latest model/camera when re-enabled
    if (enabled && !this.s.training) this.s._ensureScheduler(), this.s._scheduleFrame();
    if (this.s._remote) return this._send('setEnabled', { enabled });
  }

  /** Point the camera at training frame i (exact pose + intrinsics). Returns
   *  the camera meta so the UI can overlay the photograph. */
  lookThrough(i) {
    const meta = this.s.trainer.camMeta[i];
    if (!this.s._remote) {
      this.setCamera({ ...meta });
      this._lookIdx = i;   // setCamera clears it (it is the free-camera path)
      this._offset = meta.offset;
      return meta;
    }
    this._lookIdx = i;
    this.camera = { ...meta };
    this._offset = meta.offset;
    this._dirty = true;
    return this._send('lookThrough', { i }).then(() => meta);
  }

  /** Free camera: { R, t, f, cx, cy, w, h }. */
  setCamera(cam) {
    this.camera = cam;
    this._lookIdx = null;
    this._offset = 0;
    this._dirty = true;
    if (!this.s.training && this.s._sched) this.s._scheduleFrame();
    if (this.s._remote) return this._send('setCamera', { cam });
  }

  /** Render the current camera now — latest-wins: while a render is still
   *  in flight (continuous orbiting submits faster than the GPU drains),
   *  mark dirty and return instead of queueing stale frames behind it;
   *  the host's next frame (or the loop's tick) retries with the newest
   *  camera. `force` submits unconditionally for synchronous snapshots
   *  that read the canvas back immediately. */
  renderNow(force = false) {
    if (this.s._remote) {
      if (!this.enabled || !this.camera) return Promise.resolve(false);
      // latest-wins like the local branch: a render already draining its
      // round trip coalesces — mark dirty (the next tick retries with the
      // newest camera) instead of queueing stale renders behind the chain
      // (continuous orbiting used to backlog them all: orbit lag vs main,
      // 2026-11). A coalesced render is accepted work, not a refusal: it
      // resolves true without touching the no-present streak, which only
      // counts worker refusals.
      if (this._remoteInFlight && !force) { this._dirty = true; return Promise.resolve(true); }
      this._remoteInFlight = true;
      // a submitted render clears dirty (latest-wins: an in-flight one keeps it)
      return this._send('renderNow', { force }).then((ok) => {
        this._remoteInFlight = false;
        // accepted renders push back the auto-refresh like the local branch:
        // without this the tick re-fires every frame during training and the
        // worker never parks its present loop (same-phone lag vs main, 2026-11)
        if (ok) { this._dirty = false; this._falseSince = null; this._lastAuto = performance.now(); return ok; }
        // sustained refusal WITHOUT a rejection is silent black (worker
        // disabled/desynced, stuck in-flight, or never attached) — after a
        // grace window it surfaces as view-error like a rejection does,
        // with a probe diagnosis of WHICH link is broken (see below)
        const now = performance.now();
        if (this._falseSince == null) this._falseSince = now;
        const stallMs = this.s.opts.viewStallMs ?? 5000;
        if (!this._viewFailed && now - this._falseSince > stallMs) {
          this._viewFailed = true;
          const secs = ((now - this._falseSince) / 1000).toFixed(0);
          return this._diagnoseNoPresent().then((d) => {
            const msg = `worker presented no frames for ${secs}s ` +
              `(renders refused: disabled, busy, or no canvas) — probe: ${d.summary}`;
            this.s._log(`view renderNow stalled: ${msg}${d.worker ? ` [worker ${d.worker}]` : ''}`);
            this.s._em.emit('event',
              { kind: 'view-error', cmd: 'renderNow', message: msg, diagnosis: d.kind });
            return ok;
          });
        }
        return ok;
      }, (e) => { this._remoteInFlight = false; throw e; });
    }
    if (!this.enabled || !this.ctx || !this.camera) return;
    if (this._renderInFlight && !force) { this._dirty = true; return; }
    this._dirty = false;
    this.s.trainer.renderView(this.camera, this.ctx, 0, this._offset);
    const t = this.s.trainer.device.queue.onSubmittedWorkDone();
    this._renderInFlight = t;
    t.then(
      () => { if (this._renderInFlight === t) this._renderInFlight = null; },
      () => { if (this._renderInFlight === t) this._renderInFlight = null; });
    // any render satisfies the auto-refresh — a host that renders on its own
    // cadence keeps pushing this back and the tick below never double-fires
    this._lastAuto = performance.now();
  }

  /** Which link of the remote view path is broken: a canvas-free probe
   *  capture with content means the worker renders fine and only the
   *  present fails; a black capture means the raster/camera itself
   *  produces nothing. Plus the worker's own view state (desync shows
   *  as no canvas / no camera / disabled / stuck in flight). Best
   *  effort — a failed probe is itself a diagnosis. Remote only (the
   *  no-present streak that calls it never fires locally). */
  async _diagnoseNoPresent() {
    let worker = null;
    try {
      const i = await this._send('inspect', {});
      if (i) worker = `current=${i.current}/${i.canvases} canvases, enabled=${i.enabled}, ` +
        `cam=${i.camera ? `${i.camera.w}x${i.camera.h}` : 'none'}, ` +
        `offscreen=${i.canvasWH ? `${i.canvasWH.w}x${i.canvasWH.h}` : 'none'}, ` +
        `inFlight=${i.inFlight}`;
    } catch (e) { worker = `inspect failed: ${(e && e.message) || e}`; }
    let kind = 'unknown', summary = 'probe skipped (no camera)';
    if (this.camera) {
      try {
        const shot = await this.capture();
        const s = litSample(shot && shot.rgba);
        if (s.lit >= 3 && s.frac > 0.001) {
          kind = 'presentation';
          summary = `capture renders content (${(100 * s.frac).toFixed(1)}% lit) — the present fails, not the raster`;
        } else {
          kind = 'raster';
          summary = 'capture comes back black — the worker rasterizes nothing for this camera';
        }
      } catch (e) {
        kind = 'capture-failed';
        summary = `probe capture failed: ${(e && e.message) || e}`;
      }
    }
    return { kind, summary, worker };
  }

  /** Pixels for the current camera, no canvas involved (snapshots, error
   *  renders): { w, h, rgba }. The capture renders itself — no preceding
   *  renderNow needed — and works identically local or remote. */
  async capture() {
    if (!this.camera) throw new Error('view.capture: no camera');
    if (this.s._remote) return this.s.trainer.capture(this.camera);
    return this.s.trainer.renderCapture(this.camera);
  }

  _tick(frameCount, training) {
    if (!this.enabled || !this.camera) return;
    if (!this.s._remote && !this.ctx) return;
    if (isHidden()) return; // (7) hidden: skip renders, keep the cycles for training
    // remote sends are fire-and-forget here (the chain orders them); a dead
    // worker rejects and the loop must not die with an unhandled rejection
    const fire = () => {
      const r = this.renderNow();
      if (r && typeof r.catch === 'function') r.catch(() => {});
    };
    if (this._dirty) { fire(); return; }
    // during training, refresh the (unchanged-camera) view sparingly — every
    // render is a full raster pass stolen from the optimiser (~2/s at most,
    // fewer on slow devices: ~25 iterations between refreshes)
    const ips = this.s._itersAtStats && this.s._lastStats
      ? Math.max(1, this.s._lastIps || 100) : 100;
    const interval = Math.max(500, 25000 / ips);
    if (training && performance.now() - (this._lastAuto || 0) > interval) {
      fire();
    }
  }
}

/** @param {SessionOptions} [opts] */
export function createSession(opts = {}) { return new Session(opts); }
