// Splat.js — the app. One screen, four states:
//
//   ready → prep → train → done      (+ Details, on demand, once done)
//
// The interaction model is the v2 mockup's, verbatim; underneath it now sits
// the real library: prep beats are the solver's own progress events, the
// model on the stage is the WebGPU trainer's render, the curve can show
// measured PSNR or an SSIM estimate from that score, and Export writes a real .ply. The UI talks to ONE object — the
// splat.js Session — plus the trainer's rendered canvas.

import { createSession, parseCameras, gaussiansToPly, undistortFrames, solveTierOpts, litSample } from '../../src/index.js';
import { extractSharpFrames, isVideoFile } from '../../src/io/video.js';
import { recordCaptureVideo, cameraSupported } from './camera.js';
import { saveLastCapture, loadLastCapture } from './store.js';
import { initXRUI, reviewCapture, exportCapture, importNativeCapture } from './xr-ui.js';
import { datasetFromSelection, validateDataset, measuredSeeds, storedCapture, unpackZipFile } from './xr-dataset.js';
import { parseCycleCount, parseLODCount, maxLODLevels, buildLODPlan } from '../../src/training-settings.js';
import { zipStore } from './zip.js';
import { saveBlob } from './download.js';
import { buildSessionZip, fetchModel, gravityUpForKnownRun, continuedInputDim, continuedBuf, trimProgressToIter } from './session_io.js';
import { buildDisplayCloud } from './scene-cloud.js';
import { REPO, DISCORD, DATA, ownSet } from './data.js';

import { Viewport, camCentre, quatFromR, quatToR, qslerp, rollAngleToUp, snapRollAngle, rerollPose, exifIntrinsics } from './viewport.js';
import { Developer, fitRect } from './develop.js';
import { Chart, psnrToApproxSsimPercent } from './chart.js';
import { bmp, readyBmp, sizedUrl } from './img.js';
import * as Headless from './headless.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// Two front doors, one codebase: classic and wall-first layouts use local
// photo intake and device-saved work (splat-js2, or ?v2 anywhere).
// One UI, two shells: the browser/ folder boots this same app with a marker
// (its generated index.html sets __SPLAT_SHELL) or ?shell=browser — local
// only, no AR capture, no avatar route, no share backend. Everything else
// is the same code; the gates below are the whole difference.
const SHELL = new URLSearchParams(location.search).get('shell') || globalThis.__SPLAT_SHELL || 'app';
const WALL_FIRST = SHELL === 'browser' || /splat-js2/.test(location.pathname)
  || new URLSearchParams(location.search).has('v2');
// main's browser trainer always exposed its orbit preview. The shared app
// may default visuals off on phones, but /browser is a trainer/viewer and
// presents the evolving splat without another opt-in.
const defaultTrainingVisuals = () => SHELL === 'browser';

// short first run (phones especially) — the done screen offers "+10k cycles"
// which stretches the trainer's schedules and resumes. 20k, not 10k: with a
// 10k horizon the growth phase is squeezed against too few settle iterations
// and the model comes out over-grown for its polish time.
const INITIAL_ITERS = 20000;
// a person is one small object seen from every side: 30k was visibly sharper than
// 20k on the 1080p orbit (face splats 3.4k -> 4.7k), 40k added nothing with or
// without a larger cap (2026-09-14, docs/lab-log.md); the phone budget gets
// measured against this later, not the other way round
const AVATAR_ITERS = 30000;
// how long the cut-out strip stays up after the matte while the solve starts
// behind it (?cutouts=ms, 0 = straight on)
const CUTOUTS_MS = (() => { const q = new URLSearchParams(location.search).get('cutouts'); return q != null && +q >= 0 ? +q : 6000; })();
const AVATAR_MAX_FRAMES = 0;     // frame thinning is OFF (0): thinning 208 -> 100 broke Lisa's chain and 208 -> 200 broke her landmark stage (2026-09-15); ?avframes=N thins for experiments
const MORE_ITERS = 10000;

// ?perf runs a short instrumented benchmark (default 1000 iterations, or
// ?perf=2500) and offers the frame-loop timing log as a text file — for
// diagnosing devices at arm's length (phones, other people's machines)
const PERF_Q = new URLSearchParams(location.search).get('perf');
const PERF = { on: PERF_Q != null, iters: Math.max(200, parseInt(PERF_Q, 10) || 1000) };

// ?2x trains against a DOUBLE-resolution working buffer (the photos are only
// re-gridded, no new information): the loss then sees and suppresses the
// bright edge ringing that otherwise appears when the view renders above
// training resolution. Costs ~1.7x training time and a little native-res
// PSNR — an experiment flag, off by default.
const BUF2X = new URLSearchParams(location.search).has('2x');

// ?eval runs the standard benchmark protocol: every Nth photo (default 8,
// ?eval=4 etc.) is held out of training and scored together at the end — the
// novel-view PSNR that quality papers report. Normal visitors train on every
// photo; this flag exists for honest measurement.
const EVAL_Q = new URLSearchParams(location.search).get('eval');
const EVAL = { on: EVAL_Q != null, split: Math.max(2, parseInt(EVAL_Q, 10) || 8) };

// ?mcmc + ?iters=N: the experimental optimizer set. STICKY for the browser
// session — preset tiles are real links, so a plain query flag would be shed
// on the first click. ?mcmc=0 clears; the toast fires at boot and at train.
{
  const q = new URLSearchParams(location.search);
  if (q.get('mcmc') === '0') { sessionStorage.removeItem('splatjs_mcmc'); sessionStorage.removeItem('splatjs_iters'); }
  else if (q.has('mcmc')) sessionStorage.setItem('splatjs_mcmc', '1');
  if (q.get('iters')) sessionStorage.setItem('splatjs_iters', q.get('iters'));
}
const MCMC_ON = sessionStorage.getItem('splatjs_mcmc') === '1';
const ITERS_OVERRIDE = parseInt(sessionStorage.getItem('splatjs_iters'), 10) || 0;
// solver switches (2026-09-17, branch-vs-main A/B on Tom's orbit): ?sfmsubabove=N (focal search subsample
// above N images; default 48 again since 09-17d, the branch's 120 is the experiment), ?finaltrials=N (final passes from the top
// focal candidates, default 2), ?pairrelax=0 (no retry) / ?pairrelax=gap (the gap rule on) — the branch's three extra passes are opt-in
const SFM_BISECT = (() => { const q = new URLSearchParams(location.search); const o = {};
  if (+q.get('sfmsubabove') > 0) o.searchSubsetAbove = +q.get('sfmsubabove');
  if (+q.get('finaltrials') > 0) o.finalTrials = +q.get('finaltrials');   // final passes from the top focal candidates, keep the lowest BA rms (default 2 up to 120 images; 1 = single pass)
  if (+q.get('inittrialsupto') > 0) o.initTrialsUpTo = +q.get('inittrialsupto');   // init-pair trials on the final pass up to N images (default 60)
  if (q.get('pairdebug') === '1') o.pairDebug = true;          // per-neighbour-pair RANSAC probe (plain route; the avatar route reads it too)
  if (q.get('pairrelax') === '0') o.pairRelax = false;         // no retry at all
  if (q.get('pairrelax') === 'gap') o.pairRelaxGap = true;     // the gap rule on top of the < 70 % rule
  // ?pairabs=A,N: the absolute pair gates (any pair A inliers, neighbours N) in the FIRST pass — what the retry uses (100 / 15)
  if (q.get('pairabs')) { const [a, nb] = q.get('pairabs').split(',').map(Number); if (a > 0) o.pairMinInliers = a; if (nb > 0) o.pairMinInliersAdj = nb; }
  return o; })();
// A video is a chain: consecutive frames overlap by construction, and the solver's default gate
// (inlier RATIO >= 40 %, no absolute count since b806846) throws neighbour pairs with 215 of 600
// inliers away. On Tom's orbit that left 175 usable pairs and a solve that depended on which video
// DECODER produced the pixels (hardware vs software: 14 % apart, the user's tab a third answer,
// 34 % off live). With the retry's gates in the FIRST pass — any pair 100 inliers, neighbours 15 —
// 223 pairs, and both decoders land on live's solve within 0.2 % / 0.2° (2026-09-17). Photo sets
// keep the ratio gate: the same gates cost the Truck 3 dB in 09-08's measurement. ?pairabs=0 turns
// it off for an A/B; ?pairabs=A,N sets other values (SFM_BISECT wins over this).
const VIDEO_GATES = () => (S.preset && S.preset.video && new URLSearchParams(location.search).get('pairabs') !== '0'
  ? { pairMinInliers: 100, pairMinInliersAdj: 15 } : {});

// ?dilate=<px2> + ?aniso=<w>: the NEEDLE experimental set — lower the
// screen-space AA dilation floor and/or the isotropy spring so splats may
// train thin (Brush-like strokes). Sticky like ?mcmc; ?dilate=0 clears both.
// In-session view renders with the same kernels (faithful); exports/shares
// still bake and rasterize at the stock 0.3, so needles view fatter there.
{
  const q = new URLSearchParams(location.search);
  if (q.get('dilate') === '0') {
    for (const k of ['splatjs_dilate', 'splatjs_aniso', 'splatjs_minscale']) sessionStorage.removeItem(k);
  } else if (q.get('dilate')) sessionStorage.setItem('splatjs_dilate', q.get('dilate'));
  if (q.get('aniso') != null && q.get('aniso') !== '') sessionStorage.setItem('splatjs_aniso', q.get('aniso'));
  if (q.get('minscale')) sessionStorage.setItem('splatjs_minscale', q.get('minscale'));
}
const DILATE_OVR = parseFloat(sessionStorage.getItem('splatjs_dilate'));
const ANISO_OVR = sessionStorage.getItem('splatjs_aniso') == null ? NaN : parseFloat(sessionStorage.getItem('splatjs_aniso'));
const MINSCALE_OVR = parseFloat(sessionStorage.getItem('splatjs_minscale'));
const NEEDLE_ON = (Number.isFinite(DILATE_OVR) && DILATE_OVR > 0 && DILATE_OVR <= 1)
  || Number.isFinite(ANISO_OVR) || (Number.isFinite(MINSCALE_OVR) && MINSCALE_OVR > 0);

// ?placement=1: the PLACEMENT experimental set — the 2026-09-02 ladder combo
// (docs/plan-placement-2026-09-02.md): base A (mipComp off, anisoReg 0,
// minScale 1e-5, stock dilate 0.3) + refineV2 + growRate 0.1. Truck 30k
// +0.30, garden +0.03 over the MCMC set. Without the Mip compensation the
// trained model IS what PlayCanvas / pcview / SuperSplat rasterize, so the
// export carries no opacity bake (session.exportPlyBlob) and a continued
// share is not un-baked. Sticky like ?mcmc; ?placement=0 clears.
{
  const q = new URLSearchParams(location.search);
  if (q.get('placement') === '0') sessionStorage.removeItem('splatjs_placement');
  else if (q.get('placement')) sessionStorage.setItem('splatjs_placement', '1');
}
const PLACEMENT_ON = sessionStorage.getItem('splatjs_placement') === '1';

// training settings (start-card panel), persisted across visits.
// res 0 = auto, iters 0 = the 20k default, buf = working-buffer scale.
// Phones start from lighter defaults; anything saved wins.
// The avatar specifics, each a toggle (the user, 2026-09-17: "when I toggle all specifics off, it should
// solve like non-avatar"). All on = today's avatar run; all off = a scene run that happens to be followed
// by the avatar stages. Persisted with the settings (settings.av), rows in the panel's Avatar group.
const AV_DEFAULTS = {   // ALL OFF by default (the user, 2026-09-17: "pretty happy with all off, make this the default")
  crops: 'off',   // 'off' | 'room' (windows keep the frame's pose) | 'face' (head windows re-posed by face-PnP when the head moved)
  tier: false,    // the precise solve tier instead of the panel's Camera solve
  budget: false,  // 1.1 GB decode budget (frames decoded larger than the default 700 MB allows)
  seed: false,    // 100k seed instead of the point-count law
  opa: false,     // opacity pressure 0.004 stopping at half the run (scene: 0.01 throughout)
  needle: false,  // needle term 0.03
};   // no 'cap' toggle: an avatar never limits its splats — the ceiling is always the allocation maximum
const AV_KEYS = Object.keys(AV_DEFAULTS);
/** Coarse pointer + small screen: the phone class that sizes auto budgets,
 *  the seed target, and the memory-saver decode path. */
function isPhoneClass() {
  return matchMedia('(any-pointer: coarse)').matches &&
    Math.min(screen.width, screen.height) <= 820;
}
/** Where the GPU training core lives: the worker keeps encodes/submits off
 *  the UI thread — but its OffscreenCanvas view path never presented
 *  reliably on phone WebViews (black + laggy where main's main-thread
 *  canvas is smooth on the same phone, 2026-11), so phones train locally
 *  like main. ?trainworker=1 forces the worker back for A/B;
 *  ?trainworker=0 forces local anywhere. */
function trainWorkerOpt() {
  const q = new URLSearchParams(location.search).get('trainworker');
  if (q === '1' || q === 'true') return true;
  if (q === '0' || q === 'false') return false;
  return isPhoneClass() ? false : 'auto';
}
function deviceDefaults() {
  const phone = isPhoneClass();
  return phone
    // 10k cycles: a phone's first run should hit the magic moment in a few
    // minutes — the done screen offers more cycles for anyone who wants them
    // solve: the camera-solve tier (SOLVE_TIERS in the library) — phones stay
    // on the lean solve, desktops take the middle tier; the precise tier is
    // the High/Showcase macro's (a ten-minute solve behind a two-minute
    // draft training was the complaint, 2026-09-09)
    ? { v: 2, res: 480, feat: 0, buf: 1, sh: 0, iters: 10000, splats: 0, lod: false, mcmc: false, solve: 'quick', gpu: 100, av: { ...AV_DEFAULTS } }
    : { v: 2, res: 0, feat: 0, buf: 1, sh: 3, iters: 0, splats: 0, lod: false, mcmc: false, solve: 'standard', gpu: 100, av: { ...AV_DEFAULTS } };
}
function loadSettings() {
  const d = deviceDefaults();
  try {
    const saved = JSON.parse(localStorage.getItem('splatjs_settings') || 'null');
    // v gates out saves from older panel layouts (e.g. the phone-preset
    // button that wrote sh 0 onto desktops)
    const m = saved && saved.v === 2 ? { ...d, ...saved } : d;
    m.av = { ...AV_DEFAULTS, ...(saved && saved.av || {}) };
    if (m.av.crops === true) m.av.crops = 'face'; else if (m.av.crops === false) m.av.crops = 'off';   // saves from the boolean toggle
    // saved sh 2 predates the degree-3 default: those sessions were silently
    // training degree 3 (the old `!== 2` guard), so 3 preserves real behavior
    if (m.sh === 2) m.sh = 3;
    // gpu: a corrupt save must neither park the throttle nor slam it —
    // non-numbers fall back to default, anything else clamps to the slider
    if (!Number.isFinite(+m.gpu)) m.gpu = d.gpu;
    m.gpu = Math.min(100, Math.max(10, Math.round(m.gpu)));
    // phones: iters 0 (the old 20k default) was never an explicit choice —
    // migrate to the 10k draft default so first runs stay ~5 minutes
    if (d.iters === 10000 && m.iters === 0) m.iters = 10000;
    try { m.iters = parseCycleCount(m.iters || ''); } catch { m.iters = d.iters; }
    // The settings panel uses a fixed 10k–50k menu; migrate older custom values
    // to the nearest offered cycle budget while URL performance overrides remain independent.
    if (m.iters) m.iters = [10000, 20000, 30000, 40000, 50000].reduce((best, value) =>
      Math.abs(value - m.iters) < Math.abs(best - m.iters) ? value : best, 10000);
    try { m.lodLevels = m.lod ? parseLODCount(m.lodLevels || 0, m.splats) : 0; } catch { m.lodLevels = 0; }
    if (BUF2X) m.buf = 2;
    return m;
  } catch { return d; }
}
function saveSettings() {
  try { localStorage.setItem('splatjs_settings', JSON.stringify(S.settings)); } catch { /* private mode */ }
  // the quick quality dropdown next to Start mirrors every change
  const q = $('q-quick');
  if (q) q.value = qualityOf(S.settings);
}

/** Effective soft-prune trim: '' off, else a decimal fraction string. A
 *  Custom selection reads the adjacent percent input (clamped 1–50%). */
function readSoftprune() {
  const sel = $('set-softprune');
  if (!sel || sel.value !== 'custom') return sel ? sel.value : '';
  const pct = Math.min(50, Math.max(1, Math.round(parseFloat($('set-softprune-custom').value) || 0)));
  return String(pct / 100);
}
/** Restore the soft-prune row: fixed trims select directly; a Custom
 *  selection (or a stored non-preset fraction) shows the percent input. */
function showSoftprune(v) {
  const sel = $('set-softprune'), custom = $('set-softprune-custom');
  const fixed = ['', '0.01', '0.02', '0.05', '0.1'];
  if (fixed.includes(v) && sel.value !== 'custom') {
    sel.value = v || '';
    custom.hidden = true;
    return;
  }
  sel.value = 'custom';
  custom.hidden = false;
  const pct = Math.round(parseFloat(v) * 100);
  if (v !== 'custom' && Number.isFinite(pct)) custom.value = String(Math.min(50, Math.max(1, pct)));
}
/** Custom keep percentage from a picker input (clamped 5–95%, 50% when the
 *  input is empty or garbage). Call BEFORE removing the card. */
function readKeepCustom(id) {
  const pct = Math.round(parseFloat($(id).value) || 0);
  return (pct >= 5 ? Math.min(95, pct) : 50) / 100;
}
/** Custom PRUNE percentage from a picker input. Only whole values 1–99
 *  are accepted; invalid input returns null and must not execute a prune. */
function readPruneCustom(id) {
  const raw = Number($(id).value);
  if (!Number.isInteger(raw) || raw < 1 || raw > 99) return null;
  return 1 - raw / 100;
}

// quality macros: the one-knob row that drives the individual rows below it.
// Standard = this device's defaults; anything that matches no macro shows as
// Custom. Macros never touch the 2× working buffer (an experiment flag).
const QKEYS = ['res', 'buf', 'sh', 'iters', 'splats', 'solve'];
function qualityMacros() {
  const d = deviceDefaults();
  return {
    draft:    { res: 480,   buf: 1, sh: 0,    iters: 10000,  splats: 0, solve: 'quick' },
    standard: { res: d.res, buf: 1, sh: d.sh, iters: 0,      splats: 0, solve: d.solve },
    high:     { res: 1280,  buf: 1, sh: 3,    iters: 40000,  splats: 0, solve: 'standard' },
    showcase: { res: 1280,  buf: 1, sh: 3,    iters: 100000, splats: 0, solve: 'standard' },
  };
}
function qualityOf(st) {
  if (st.lod || st.lodLevels) return 'custom';
  for (const [k, m] of Object.entries(qualityMacros())) {
    if (QKEYS.every((f) => st[f] === m[f])) return k;
  }
  return 'custom';
}

const S = {
  state: 'ready',              // ready | prep | train | done
  preset: null,
  session: null,
  photos: [],                  // [{ url, name }] — the strip + overlays
  scene: null,                 // { cams, center, radius, xyz, rgb } for overlays
  sel: 0, atFrame: -1, compare: 'swipe',
  pending: null, picking: false,
  ownUrls: null,
  fade: 0, fadeTo: 0,
  loupe: { x: 0, y: 0, r: 104 }, swipe: .5, rect: null,
  iter: 0, splats: 0, psnrTrain: null, psnrHold: null, itersPerSec: 0,
  minutes: 0, trainT0: 0,
  prep: null,                  // latest solve stage event
  feats: new Map(),            // image -> { n, x, y } (real keypoints)
  lastPairEv: null,            // latest surviving pair with sample matches
  shownPair: null,             // the pair ON STAGE: { ev, a, b, since } — held, never black
  regCams: [],                 // cameras as they register (beat 3)
  solveStats: { pairsChecked: 0, pairsUsable: 0, solveSec: 0 },
  chartEvents: [],             // real refine/growth moments for the curve
  scoreScale: 'db',
  flash: null,
  detailTab: 'score',
  trainVisuals: false,         // the app's model/compare/camera overlays sleep during training by default
  _savedGpuFacts: null,        // GPU facts captured with a local run, not the current viewer device
  _savedDetailSettings: null,
  keys: new Set(),             // held WASD keys (camera-relative fly)
  maxIters: INITIAL_ITERS,     // grows when the user continues training
  settings: loadSettings(),    // training knobs from the start-card panel
  gen: 0,                      // run generation — stale async work checks it
};

let vp, dev, chart, dchart, dvp;
// The trainer renders here. The canvas LIVES IN THE DOM, composited under the
// overlay canvas — drawImage from a WebGPU canvas is not safe on iOS Safari
// (it can return either of the last two presented frames, which flickers).
let gpuCanvas = null;

function mountModelCanvas() {
  document.getElementById('cv-model-cap')?.remove();   // snapshot fallback goes with the canvas
  S._capView = null;
  document.getElementById('cv-model')?.remove();
  gpuCanvas = document.createElement('canvas');
  gpuCanvas.id = 'cv-model';
  $('stage').insertBefore(gpuCanvas, $('cv'));
  quietView(S.session.view.attach(gpuCanvas));
  gpuCanvas.style.display = S.state === 'train' && !S.trainVisuals ? 'none' : '';
  S._viewKey = '';
}

/** Remote view calls reject when the worker is troubled; the session's
 *  view-error event carries the surfaced handling (re-mount) — these call
 *  sites just must not turn them into unhandled rejections. */
function quietView(p) {
  if (p && typeof p.catch === 'function') p.catch(() => {});
}

/** A view-error means the worker missed some view state (or the canvas
 *  transfer never landed). A FRESH canvas element re-attaches cleanly (a
 *  transferred one can never move again) and resync re-pushes enabled +
 *  camera behind the attach. Budgeted: a worker that fails every attach
 *  would otherwise remount forever — past the budget training continues
 *  without visuals, loudly. */
function remountModelCanvas(why) {
  if (!S.session || !S.session.view || !S.session.view.resync) return;
  // a live snapshot view already covers this failure — leave it alone
  if (S._capView && S._capView.active && !S._capView.dead) return;
  S._viewRemounts = (S._viewRemounts || 0) + 1;
  if (S._viewRemounts > 3) {
    // the canvas path is fundamentally broken on this device — fall back
    // to worker renders blitted as pixels (no worker canvas needed)
    engageCaptureView(why);
    return;
  }
  try {
    mountModelCanvas();
    syncTrainingVisualsControl();   // re-assert enabled/display on the fresh canvas
    quietView(S.session.view.resync());
  } catch { /* the next view-error retries within budget */ }
}

/** Snapshot view: when the worker's canvas path is broken (OffscreenCanvas
 *  WebGPU contexts don't exist on every WebView), the worker can still
 *  rasterize — renderCapture needs no canvas. Blit its pixels to a 2D
 *  canvas in the stage at the refresh cadence. Sticky for the run. */
function engageCaptureView(why) {
  if (!S.session || !S.session.view || (S._capView && S._capView.active)) return;
  S._capView = { active: true, dead: false, busy: false, fails: 0, canvas: null };
  if (gpuCanvas) gpuCanvas.style.display = 'none';
  // stop the canvas RPCs (and their no-present streak): the snapshot path
  // doesn't use them — capture() reads the main-side camera directly
  try { S.session.view.setEnabled(false); } catch { /* remote sends never throw sync */ }
  let cap = document.getElementById('cv-model-cap');
  if (!cap) {
    cap = document.createElement('canvas');
    cap.id = 'cv-model-cap';
    $('stage').insertBefore(cap, $('cv'));
  }
  S._capView.canvas = cap;
  S._viewKey = '';
  flash(`Live view unavailable (${why || 'GPU view failed'}) — snapshot view instead.`, 7000);
}

/** Snapshot-view refresh: one worker renderCapture blitted to the 2D
 *  fallback canvas (see engageCaptureView). Same key/refresh cadence as
 *  the canvas path, lower pixel budget (every refresh is a full readback). */
function drawCaptureView(capView, vp, w, h, pose) {
  const now = performance.now();
  if (vp.dirty) S._camMovedAt = now;
  const training = S.state === 'train' && S.session.training;
  const sc = Math.min(1, Math.sqrt(Math.min(S.viewPixBudget || 2560 * 1440, 0.5e6) / (w * h)));
  const gw = Math.max(2, Math.round(w * sc)), gh = Math.max(2, Math.round(h * sc));
  const key = `${gw}x${gh}|${Math.round(pose.f)}|` +
    pose.R.map((v) => Math.round(v * 8192)).join(',') + '|' +
    pose.t.map((v) => Math.round(v * 8192)).join(',');
  const refreshMs = Math.max(800, 25000 / Math.max(1, S.itersPerSec || 100));
  if (capView.busy) return;
  if (key === S._viewKey && !S.walking && !(training && now - (S._lastViewAt || 0) > refreshMs)) return;
  S._viewKey = key;
  S._lastViewAt = now;
  const cap = capView.canvas;
  if (!cap) return;
  if (cap.width !== gw || cap.height !== gh) { cap.width = gw; cap.height = gh; }
  const cam = { R: pose.R, t: pose.t, f: pose.f * sc, cx: pose.cx * sc, cy: pose.cy * sc, w: gw, h: gh };
  quietView(S.session.view.setCamera(cam));   // syncs the main-side camera capture() reads
  capView.busy = true;
  S.session.view.capture().then((shot) => {
    capView.busy = false;
    if (S._capView !== capView || capView.canvas !== cap) return;   // superseded
    if (!shot || !shot.rgba || !shot.w || !shot.h) throw new Error('empty capture');
    capView.fails = 0;
    const data = shot.rgba instanceof Uint8ClampedArray ? shot.rgba : new Uint8ClampedArray(shot.rgba);
    cap.getContext('2d').putImageData(new ImageData(data, shot.w, shot.h), 0, 0);
  }).catch((e) => {
    capView.busy = false;
    if (S._capView !== capView) return;
    if (++capView.fails >= 3) {
      capView.dead = true;
      if (capView.canvas) capView.canvas.style.display = 'none';
      if (!S._viewFailedFlash) {
        S._viewFailedFlash = true;
        flash(`Visuals unavailable — training continues without them (${(e && e.message) || 'capture failed'}).`, 9000);
      }
    }
  });
}

/** Silent-black watchdog: renders can be ACCEPTED (no error, no streak)
 *  while the screen stays black — a worker canvas that never composites
 *  (seen on Android WebViews even with in-rAF submits). Once per run per
 *  canvas, shortly after the canvas path goes live, the raster is probed:
 *  content in the probe but black in the placeholder means the present is
 *  broken and the pixel-blit snapshot view takes over (its capture is
 *  proven working by the probe itself). A black probe twice running says
 *  the raster renders nothing for this camera instead — the snapshot
 *  couldn't help, so it surfaces as words. A lit placeholder disarms
 *  silently: healthy views are never touched. */
function armPresentWatchdog() {
  if (!S.session || !S.session._remote) return;   // local views present in-thread
  if (S.state !== 'train' && S.state !== 'done') return;
  if (S._capView && S._capView.active) return;
  if (!gpuCanvas || !S.session.view || !S.session.view.enabled) return;
  if (S._presentWd && S._presentWd.canvas === gpuCanvas) return;   // already owns this canvas
  clearTimeout(S._presentWd && S._presentWd.timer);
  const canvas = gpuCanvas, gen = S.gen;
  S._presentWd = { canvas, timer: setTimeout(() => presentWatchdogFire(canvas, gen, 0), 8000) };
}

async function presentWatchdogFire(canvas, gen, blackProbes) {
  if (S._presentWd) S._presentWd.timer = null;
  if (S.gen !== gen || !S.session) return;
  if (S.state !== 'train' && S.state !== 'done') return;
  if ((S._capView && S._capView.active) || gpuCanvas !== canvas) return;
  const view = S.session.view;
  // one rehearing for anything unconcluded (black raster, missing camera,
  // failed probe, unsized canvas): a spent arm RELEASES the canvas instead
  // of disarming silently, so a later transition re-arms the check — a
  // one-shot that judged nothing used to own the canvas forever (2026-11)
  const rehear = () => {
    if (blackProbes < 1) {
      S._presentWd = { canvas, timer: setTimeout(() => presentWatchdogFire(canvas, gen, 1), 15000) };
      return true;
    }
    S._presentWd = null;
    return false;
  };
  // the view went quiet before the check ran (visuals hidden): release —
  // the next Show visuals re-arms through its transition
  if (!view || !view.enabled || typeof view._diagnoseNoPresent !== 'function') { S._presentWd = null; return; }
  if (!canvas.width || !canvas.height) { rehear(); return; }   // never sized — nothing attached yet
  let d;
  try {
    d = await view._diagnoseNoPresent();
  } catch { rehear(); return; }   // the probe failed: rehear once, then the error paths own it
  if (S.gen !== gen || (S._capView && S._capView.active) || gpuCanvas !== canvas) return;
  if (!d || d.kind === 'unknown') { rehear(); return; }   // no camera to probe yet — rehear, then release
  if (d.kind !== 'presentation') {
    // black twice running (15s apart) before crying raster: early training
    // frames can be genuinely dark while the model finds itself
    if (rehear()) return;
    flash(`Model view is black — ${d.summary}.`, 9000);
    return;
  }
  // the raster renders — is the placeholder showing it? (drawImage of a
  // WebGPU canvas isn't promised everywhere: unverifiable keeps the canvas)
  let lit = false;
  try {
    const w = 64, h = Math.max(1, Math.round(64 * canvas.height / Math.max(1, canvas.width)));
    const scratch = document.createElement('canvas');
    scratch.width = w; scratch.height = h;
    const sctx = scratch.getContext('2d', { willReadFrequently: true });
    sctx.drawImage(canvas, 0, 0, w, h);
    const s = litSample(sctx.getImageData(0, 0, w, h).data);
    lit = s.lit >= 3 && s.frac > 0.001;
  } catch { return; }
  if (lit) return;   // healthy: the present works, disarm silently
  engageCaptureView('the live canvas stays black on this device');
}

function syncTrainingVisualsControl() {
  const training = S.state === 'train';
  const detailOpen = !!($('details') && !$('details').hidden);
  const visualsActive = !training || (S.trainVisuals && !detailOpen);
  const button = $('btn-training-visuals');
  if (button) {
    button.hidden = !training || detailOpen;
    button.textContent = S.trainVisuals ? 'Hide visuals' : 'Show visuals';
    button.setAttribute('aria-pressed', String(!!S.trainVisuals));
  }
  // snapshot view active: the dead canvas stays hidden and the worker view
  // stays disabled (its renders would only refuse) — pixels come from capture()
  const capActive = !!(S._capView && S._capView.active && !S._capView.dead);
  if (gpuCanvas) gpuCanvas.style.display = !capActive && visualsActive ? '' : 'none';
  if (S._capView && S._capView.canvas) S._capView.canvas.style.display = capActive && visualsActive ? '' : 'none';
  if (S.session && S.session.view && S.session.view.setEnabled) S.session.view.setEnabled(capActive ? false : visualsActive);
  const stage = $('stage'), dock = $('dock');
  const plot = dock?.querySelector('.chartwrap') || stage?.querySelector('.chartwrap-expanded');
  const expandPlot = training && !S.trainVisuals && !detailOpen;
  let plotMoved = false;
  if (expandPlot && plot) {
    if (plot.parentElement !== stage) {
      stage.insertBefore(plot, $('hud'));
      plotMoved = true;
    }
    plot.classList.add('chartwrap-expanded');
    dock.classList.add('chart-expanded');
  } else {
    if (plot) plot.classList.remove('chartwrap-expanded');
    if (plot && plot.parentElement === stage && dock) {
      dock.insertBefore(plot, dock.querySelector('.tscores'));
      plotMoved = true;
    }
    dock?.classList.remove('chart-expanded');
  }
  // The canvas CSS fills its new container. Redraw its backing pixels after
  // reparenting instead of letting the old dock-sized bitmap get stretched.
  if (plotMoved) requestAnimationFrame(() => chart?.resize());
  // every transition into the live canvas path (Show visuals, done screen,
  // fresh mount, remount) gets its silent-black check; the arm itself is
  // idempotent per canvas
  armPresentWatchdog();
}

function toggleTrainingVisuals() {
  if (S.state !== 'train') return;
  S.trainVisuals = !S.trainVisuals;
  if (S.trainVisuals) S._viewKey = ''; // render the current model at the current camera pose
  else if (vp && vp.ctx) vp.ctx.clearRect(0, 0, vp.w, vp.h);
  syncTrainingVisualsControl();
}

// the OAuth popup lands back on this page with ?code= — report and close
boot();

// ── boot ────────────────────────────────────────────────────────────────────
function boot() {
  vp = new Viewport($('cv'));
  vp.onLeave = leaveFrame;
  // Touch-hold pausing lives on the stage (wireStage), not here: the
  // viewport's drag hooks miss frame-locked compare touches, and pausing
  // belongs to every finger on the visualization, not just orbit drags.
  vp.onDragStart = () => { stopTour(); };
  dev = new Developer();

  $('btn-go').addEventListener('click', async () => {
    if (S.picking) { const p = S.pending || S.preset; closePicker(); await open(p, true); return; }
    if (S.pendingShare) {
      // a community set: its photographs were deliberately NOT loaded with
      // the detail card — resolve the recon now, on the actual start
      const it = S.pendingShare;
      const go = $('btn-go');
      go.disabled = true;
      flash('Fetching the photographs …', 30000);
      try {
        const rj = await fetchShareRecon(it);
        if (!rj.source || !rj.source.urls || !rj.source.urls.length || !rj.source.urls.every(Boolean)) {
          throw new Error('this creation has no public photographs to train from');
        }
        S.preset = { id: '__sample', name: it.title || 'Shared creation' };
        S.photos = rj.source.names.map((n, i) => ({ url: rj.source.urls[i], name: n }));
        S.sel = 0;
        S.pendingShare = null;
      } catch (e) {
        flash(`Could not start: ${e.message}`, 8000);
        go.disabled = false;
        return;
      }
      go.disabled = false;
    }
    await startPrep();
  });
  $('btn-new').addEventListener('click', async (e) => {
    e.stopPropagation();
    await goBack();
  });
  $('btn-training-visuals').addEventListener('click', toggleTrainingVisuals);
  $('card-x').addEventListener('click', closePicker);
  $('file-input').addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    e.target.value = '';   // so the same file can be picked again (a cancelled video review, a retry)
    useOwnPhotos(files);
  });
  if (cameraSupported()) {
    const rb = $('btn-record');
    rb.hidden = false;
    rb.addEventListener('click', async () => {
      try {
        const got = await recordCaptureVideo();
        if (!got) return;
        if (got.kind === 'video') useOwnVideo(got.file);
        else useOwnPhotos(got.files);   // stills: straight in, no extraction
      } catch (e) {
        console.error(e);
        flash(`Camera unavailable: ${e.message}`, 6000);
      }
    });
  }

  const card = $('start');
  initXRUI({
    flash,
    onCapture: (capture) => useOwnPhotos(capture.files, capture),
    // the native import runs before any review modal exists: Start refuses
    // while the counter stands (see startPrep)
    setBusy: (on) => { S._intakeN = Math.max(0, (S._intakeN || 0) + (on ? 1 : -1)); },
  });
  $('xr-review-set').addEventListener('click', async () => {
    const dataset = S.preset?.capture || { kind: 'photos', files: S.preset?.files || [], points: [] };
    try { const retained = await reviewCapture(dataset); if (retained) await useOwnPhotos(retained.files, retained); }
    catch (e) { flash(e.message, 7000); }
  });
  $('xr-export-set').addEventListener('click', async () => {
    try {
      const saved = await exportCapture(S.preset?.capture || { files: S.preset?.files || [], points: [] });
      flash(saved ? 'Capture export saved.' : 'Export cancelled.', 4000);
    } catch (e) { flash(`Capture export failed: ${e.message}`, 7000); }
  });
  ['dragenter', 'dragover'].forEach((t) => card.addEventListener(t, (e) => {
    e.preventDefault(); card.classList.add('drop');
  }));
  ['dragleave', 'dragend'].forEach((t) => card.addEventListener(t, () => card.classList.remove('drop')));
  card.addEventListener('drop', async (e) => {
    e.preventDefault(); card.classList.remove('drop');
    const files = e.dataTransfer.files;
    // a saved run comes back through the same door as photos
    if (files.length === 1 && /\.(sog|ply)$/i.test(files[0].name)) {
      restoreSession({ bytes: new Uint8Array(await files[0].arrayBuffer()) });
      return;
    }
    // zips ride the photo flow: useOwnPhotos sniffs session vs capture
    useOwnPhotos(files);
  });
  $('d-close').addEventListener('click', closeDetails);
  $('d-prev').addEventListener('click', () => detailFlip(-1));
  $('d-next').addEventListener('click', () => detailFlip(1));

  // the settings panel: values in, values out, persisted
  const st = S.settings;
  const showSettings = () => {
    // the Avatar group only when this run is going to be an avatar run — the box
    // on the start card (or a restored capture's manifest); a plain run never reads it
    document.querySelectorAll('#settings .av-only').forEach((el) => { el.hidden = !(S.avatarWanted || S.avatar); });
    // Training resolution: only the steps the loaded pictures can fill (a 1080p video has no
    // 1600 px row), and Full says what size that is. -1 = the pictures' own size (the user,
    // 2026-09-17). A saved step at or above the set's size shows as Full.
    {
      const native = S.setMaxDim || 0;
      for (const o of $('set-res').options) {
        const v = parseInt(o.value, 10) || 0;
        if (v > 0) { o.hidden = native > 0 && v >= native; o.textContent = `${v} px`; }
        else if (v === -1) o.textContent = native ? `Full (${native} px)` : 'Full size';
      }
      const want = st.res > 0 && native > 0 && st.res >= native ? -1 : st.res;
      $('set-res').value = want ? String(want) : '';
    }
    $('set-feat').value = st.feat ? String(st.feat) : '';
    $('set-solve').value = st.solve === 'quick' ? 'quick' : 'standard';   // 'precise' (an alias) shows as Standard
    $('set-buf').value = String(st.buf);
    $('set-buf-hint').hidden = st.buf !== 2;
    $('set-sh').value = String(st.sh);
    $('set-shup').value = st.shup ? '1' : '0';
    for (const k of AV_KEYS) { const el = $('set-av-' + k); if (!el) continue; el.value = k === 'crops' ? String(st.av && st.av.crops || 'face') : (st.av && st.av[k] === false ? '0' : '1'); }
    $('set-iters').value = String(st.iters || 20000);
    $('set-iters').setCustomValidity('');
    $('set-splats').value = st.splats ? String(st.splats) : '';
    $('set-q').value = qualityOf(st);
    $('q-quick').value = qualityOf(st);   // the dropdown next to Start
    // LOD levels only make sense from 1M splats up
    const lodOk = st.splats >= 1000000;
    $('set-lod').disabled = !lodOk;
    $('set-lod').value = lodOk && st.lod ? '1' : '';
    $('set-lod-count').disabled = !lodOk || !st.lod;
    $('set-lod-count').max = String(maxLODLevels(st.splats) || 4);
    $('set-lod-count').value = String(st.lodLevels || 0);
    $('set-lod-count').setCustomValidity('');
    $('set-lod-hint').textContent = lodOk
      ? `0 = Auto (${maxLODLevels(st.splats)} snapshots). Choose 1–${maxLODLevels(st.splats)} earlier saved snapshots; final full detail is always included.`
      : 'Enable LOD with a splat budget of at least 1,000,000. 0 = Auto.';
    $('set-mcmc').value = st.mcmc === '0' ? '0' : '';
    showSoftprune(st.softprune);
    $('set-gpu').value = String(st.gpu ?? 100);
    $('set-gpu-v').textContent = `${$('set-gpu').value}%`;
    if ($('t-gpu')) { $('t-gpu').value = String(st.gpu ?? 100); $('t-gpu-v').textContent = `${$('t-gpu').value}%`; }
  };
  showSettings();
  // the quality knob exists twice — full row in the gear panel, quick
  // dropdown beside Start — one handler, either element
  const pickQuality = (v) => {
    const m = qualityMacros()[v];
    if (!m) return;           // Custom is a display state, not a choice
    Object.assign(st, m);
    st.lod = false; st.lodLevels = 0;
    showSettings();
    saveSettings();
  };
  $('set-q').addEventListener('change', () => pickQuality($('set-q').value));
  $('q-quick').addEventListener('change', () => pickQuality($('q-quick').value));
  $('btn-settings').addEventListener('click', () => {
    const open = $('settings').hidden;
    if (open) showSettings();   // the rows depend on the loaded set (avatar or not, its picture size) — rebuilt on every open
    // the gear lives on whichever card is showing — start page or detail
    const card = $('detail').hidden ? $('start') : $('detail');
    if (open && matchMedia('(min-width: 641px)').matches) {
      // pin the card's top edge: the panel extends DOWNWARD only, and the
      // card scrolls if it outgrows the screen (full-screen phones skip this)
      const top = Math.max(10, card.getBoundingClientRect().top);
      card.style.top = `${top}px`;
      card.style.margin = '0 auto';
      card.style.bottom = 'auto';
      card.style.maxHeight = `calc(100% - ${top + 12}px)`;
    } else {
      card.style.top = ''; card.style.margin = '';
      card.style.bottom = ''; card.style.maxHeight = '';
    }
    $('settings').hidden = !open;
    $('btn-settings').setAttribute('aria-expanded', String(open));
    // the panel reflows the page under a stationary cursor — drop any stale
    // chart hover so the readout cannot re-fire detached on the next redraw
    chart?.clearHover();
    dchart?.clearHover();
  });
  const readSettings = () => {
    const cyclesInput = $('set-iters'), lodInput = $('set-lod-count');
    cyclesInput.setCustomValidity(''); lodInput.setCustomValidity('');
    let cycleCount, lodCount = 0;
    const splatBudget = parseInt($('set-splats').value, 10) || 0;
    const lodEnabled = !!$('set-lod').value && splatBudget >= 1000000;
    try { cycleCount = parseCycleCount(cyclesInput.value); }
    catch (e) { cyclesInput.setCustomValidity(e.message); cyclesInput.reportValidity(); flash(e.message, 7000); return false; }
    if (!cyclesInput.validity.valid) { cyclesInput.reportValidity(); return false; }
    if (lodEnabled) {
      try { lodCount = parseLODCount(lodInput.value, splatBudget); }
      catch (e) { lodInput.setCustomValidity(e.message); lodInput.reportValidity(); flash(e.message, 7000); return false; }
    }
    st.res = parseInt($('set-res').value, 10) || 0;
    st.feat = parseInt($('set-feat').value, 10) || 0;
    st.solve = $('set-solve').value || 'standard';
    st.buf = parseFloat($('set-buf').value) || 1;
    st.sh = parseInt($('set-sh').value, 10);
    st.shup = $('set-shup').value === '1';
    st.av = { ...AV_DEFAULTS }; for (const k of AV_KEYS) { const el = $('set-av-' + k); if (!el) continue; st.av[k] = k === 'crops' ? el.value : el.value !== '0'; }
    st.iters = cycleCount;
    st.splats = splatBudget;
    st.lod = lodEnabled;
    st.lodLevels = lodCount;
    st.mcmc = $('set-mcmc').value;  // '' Auto | '1' on | '0' off
    st.softprune = readSoftprune();  // '' off | decimal trim fraction (Custom reads 1–50%)
    st.gpu = Math.min(100, Math.max(10, parseInt($('set-gpu').value, 10) || 100));
    showSettings();
    saveSettings();
    return true;
  };
  S.commitSettings = readSettings;
  // every select in the panel persists on change — the avatar rows (and Horizontal SH) were
  // missing from this list, so their toggles changed the dropdown and nothing else (2026-09-17)
  for (const id of ['set-res', 'set-feat', 'set-solve', 'set-buf', 'set-sh', 'set-iters', 'set-splats', 'set-lod', 'set-lod-count', 'set-mcmc', 'set-softprune', 'set-softprune-custom', 'set-shup', 'set-gpu', ...AV_KEYS.map((k) => 'set-av-' + k)]) {
    const el = $(id); if (el) el.addEventListener('change', readSettings);
  }
  // the avatar route: the box on the start card. It is a property of THIS run,
  // not a persisted setting — the Avatar settings group follows it at once so
  // the knobs are there to adjust before Start training.
  $('set-avatar').addEventListener('change', (e) => {
    S.avatarWanted = !!e.target.checked;
    document.querySelectorAll('#settings .av-only').forEach((el) => { el.hidden = !(S.avatarWanted || S.avatar); });
  });
  // count slider: live label while dragging, the (cheaper) photo-list rebuild
  // on release; the value label is also the "use all" button
  const countLabel = () => {
    const p = S.preset;
    if (p) $('set-count-v').textContent = `${$('set-count').value} / ${p.maxCount || p.count}`;
  };
  $('set-count').addEventListener('input', countLabel);
  $('set-count').addEventListener('change', () => {
    const p = S.preset;
    if (!p || p.files) return;
    p.useCount = parseInt($('set-count').value, 10) || p.count;
    countLabel();
    applyCount(p);
  });
  $('set-count-v').addEventListener('click', () => {
    const p = S.preset;
    if (!p || p.files) return;
    $('set-count').value = p.maxCount || p.count;
    $('set-count').dispatchEvent(new Event('change'));
  });
  // gpu slider: live label while dragging, and a running training follows
  // at once — the frame loop reads opts.gpuUsage every batch, so no
  // restart; the change commit persists it like every other row
  $('set-gpu').addEventListener('input', () => {
    $('set-gpu-v').textContent = `${$('set-gpu').value}%`;
    if (S.session) S.session.opts.gpuUsage = parseInt($('set-gpu').value, 10) / 100;
    const tGpu = $('t-gpu');   // the training-dock slider mirrors live
    if (tGpu) { tGpu.value = $('set-gpu').value; $('t-gpu-v').textContent = `${tGpu.value}%`; }
  });

  addEventListener('resize', () => { vp.resize(); chart?.resize(); dchart?.resize(); dvp?.resize(); });
  $('cv').addEventListener('wheel', stopTour, { passive: true });
  // Safari's proprietary pinch channel — it ignores user-scalable=no, and the
  // page must never zoom itself (pinch will become a camera control)
  for (const t of ['gesturestart', 'gesturechange', 'gestureend']) {
    document.addEventListener(t, (e) => e.preventDefault());
  }
  const WASD = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight'];
  addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (!$('about').hidden) { if (e.key === 'Escape') closeAbout(); return; }
    if (!$('details').hidden) {
      if (e.key === 'Escape') closeDetails();
      if (S.detailTab === 'marks' && e.key === 'ArrowLeft') detailFlip(-1);
      if (S.detailTab === 'marks' && e.key === 'ArrowRight') detailFlip(1);
      return;
    }
    if (S.picking && e.key === 'Escape') { closePicker(); return; }
    if (e.key === ' ' && S.state === 'train') { e.preventDefault(); toggleTrain(); }
    if (e.key === ' ' && S.state === 'done') {
      // viewing a result: space plays/pauses the capture-path flight
      e.preventDefault();
      if (S.tour) stopTour();
      else { if (S.atFrame >= 0) leaveFrame(); startTour(true); }
    }
    if (e.key === 'ArrowRight') select(S.sel + 1);
    if (e.key === 'ArrowLeft') select(S.sel - 1);
    if (WASD.includes(e.code)) S.keys.add(e.code);
  });
  addEventListener('keyup', (e) => S.keys.delete(e.code));
  addEventListener('blur', () => S.keys.clear());
  wireStage();

  // pointerdown, not click: iOS never synthesises clicks for taps on
  // non-interactive elements (the canvas), so a click listener misses them
  addEventListener('pointerdown', (e) => {
    if (!e.target.closest('.exportwrap')) {
      document.querySelectorAll('.menu').forEach((m) => { m.hidden = true; });
    }
    if (S.picking && !e.target.closest('#start')) closePicker();
  }, true);

  $('gh').href = REPO;
  $('about-gh').href = REPO;
  // the feedback channel shows only when this build has one to point at
  if (DISCORD) { $('discord').href = DISCORD; $('discord').hidden = false; }
  // the brand: on the home tile view (nothing open) it tells the story —
  // the About sheet; from inside a scene it stays the way back home
  // the card's state lives in the address (?about=1): a refresh keeps it
  // open, a sent link opens it — with the GPU line — on the recipient's device
  const aboutUrl = (on) => {
    const u = new URL(location.href);
    if (on) u.searchParams.set('about', '1'); else u.searchParams.delete('about');
    return u;
  };
  const openAbout = (fromUrl = false) => {
    $('about').hidden = false;
    renderAboutGpu();
    // a pushed UI state: the phone's Back closes the sheet, not the app —
    // unless the address already said so at load (nothing to go back to)
    if (fromUrl || (history.state && history.state.sj)) history.replaceState({ sj: 'about' }, '', aboutUrl(true));
    else history.pushState({ sj: 'about' }, '', aboutUrl(true));
  };
  const closeAbout = () => {
    const fromUrl = S._aboutFromUrl;
    S._aboutFromUrl = false;
    if (history.state && history.state.sj === 'about' && !fromUrl) { history.back(); return; }
    $('about').hidden = true;
    history.replaceState(null, '', aboutUrl(false));
  };
  window.closeAbout = closeAbout;
  $('brand').addEventListener('click', async () => {
    const atHome = !$('start').hidden && $('detail').hidden;
    if (atHome) openAbout();
    else await goBack();   // off the home tile the brand IS Back (checkpoint included)
  });
  $('read-more').addEventListener('click', (e) => {
    e.preventDefault(); e.stopPropagation();
    openAbout();
  });
  $('about-x').addEventListener('click', closeAbout);
  $('about').addEventListener('click', (e) => {
    if (!e.target.closest('.about-card')) closeAbout();
  });

  checkGpu();
  if (new URLSearchParams(location.search).get('about')) { S._aboutFromUrl = true; openAbout(true); }
  // a refresh or tab-close mid-run would throw away the model (and, before
  // storage landed, the capture) — ask first. A Back-leave never asks: it
  // secured the run first, so there is nothing left to warn about.
  addEventListener('beforeunload', (e) => {
    if (S._leaving) return;   // Back already saved — wave it through
    if (S.state === 'train' || S.state === 'prep') { e.preventDefault(); e.returnValue = ''; }
  });
  window.__splat = S;          // console access
  window.__vp = () => vp;      // console access (camera state)
  // installable PWA: the worker is a no-op (no caching), the manifest does
  // the rest. Relative URL -> correct scope on every deploy base path.
  if (SHELL !== 'browser' && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
  showIntro();
  if (MCMC_ON) {
    flash(`MCMC experimental set armed${ITERS_OVERRIDE ? ` · ${ITERS_OVERRIDE.toLocaleString()} cycles` : ''} — sticky this session (?mcmc=0 clears)`, 6000);
  }
  const mp = new URLSearchParams(location.search);
  const viewing = mp.get('space') || mp.get('model');
  if (mp.get('details')) {
    // the sheet's state came with the address (a refresh, or a link sent to
    // someone to read their GPU): open it on that tab — on the wall only the
    // GPU tab has anything to say, a viewer opening below re-renders it
    S.detailTab = mp.get('details');
    S._detailsFromUrl = true;
    if (!viewing) openDetails();
  }
  if (WALL_FIRST) {
    // Nothing is preselected. The home screen offers local capture and saved work only.
    $('btn-go').disabled = true;
    $('start').hidden = false;
    $('start').querySelector('.origin').textContent =
      'Drop in 20–200 overlapping photos of one place, or capture them right ' +
      'here — the camera solve, the training and the export all run in this ' +
      'tab. Everything runs locally on this device; photos and trained models stay here.';
  } else {
    // Keep the classic setup card with saved captures and runs from this device.
    $('start').appendChild($('gallery'));
    $('detail-body').append($('set-desc'), $('av-choice'), $('xr-set-tools'), document.querySelector('.startrow'), $('settings'));
    $('detail-back').addEventListener('click', detailClose);
    $('detail-x').addEventListener('click', detailClose);
    // no implicit boot set: nothing loads until the visitor picks — the old
    // truck default sat invisibly behind the start card and leaked into
    // Back navigation ("why is the truck loaded?")
  }
  if (SHELL === 'browser') {
    // the browser profile boots this same UI local-only: no AR capture
    // actions, no avatar route, no backend — files in, training, model
    // out. Reviews of picked photo+pose sets stay (they are local).
    for (const id of ['btn-ar', 'btn-record', 'btn-xr-help', 'xr-open-tab', 'xr-ar-state', 'xr-device-guide', 'av-choice']) {
      const el = $(id);
      if (el) el.hidden = true;
    }
    document.querySelector('.xr-flow')?.remove();
  }
  syncTrainingVisualsControl();
  requestAnimationFrame(loop);

  // a shared result: load it straight into the done-state viewer
  if (mp.get('space') && SHELL !== 'browser') restoreShared(mp.get('space'));
  else if (mp.get('model')) restoreSession({ url: mp.get('model'), reconUrl: mp.get('recon') });
  else if (mp.get('back')) headlessClearMarker().finally(() => mountWall());
  else headlessBootCheck().then((took) => { if (!took) mountWall(); });
}

// WebGPU probe: navigator.gpu can EXIST while the adapter is unavailable
// (Safari before macOS/iOS 26 keeps it behind a feature flag, Linux builds,
// hardware acceleration switched off). Probe the real adapter and, when it
// fails, say exactly how to switch it on in THIS browser instead of a
// generic shrug. `?nogpu` forces the card for testing.
async function checkGpu() {
  let ok = false;
  try {
    // the SAME request the trainer makes (src/gpu/context.js): on a dual-GPU
    // laptop the default adapter is often the integrated one while training
    // lands on the discrete one — the report must name the training adapter
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    ok = !!adapter;
    let def = null;
    try { def = adapter && await navigator.gpu.requestAdapter(); } catch { /* optional */ }
    // kept for the Details sheet's GPU tab (and its report) — the only
    // source when no session exists yet
    S.gpuProbe = adapter
      ? { adapter, info: adapter.info || {}, defaultInfo: (def && def.info) || null }
      : { failed: true };
  } catch { S.gpuProbe = { failed: true }; }
  if (!$('details').hidden && S.detailTab === 'gpu') renderDetails();
  if (!$('about').hidden) renderAboutGpu();
  if (location.search.includes('nogpu')) ok = false;
  if (ok) return;
  S.noGpu = true;
  $('btn-go').disabled = true;
  const ua = navigator.userAgent;
  const iosLike = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const safari = /Safari\//.test(ua) && !/Chrome|Chromium|CriOS|Edg|Android|Firefox|FxiOS/.test(ua);
  const firefox = /Firefox|FxiOS/.test(ua);
  const linux = /Linux/.test(ua) && !/Android/.test(ua);
  let how;
  if (safari && iosLike) {
    how = 'iOS 26 has it on by default — updating is the easy fix. On earlier iOS: ' +
      '<b>Settings → Apps → Safari → Advanced → Feature Flags</b>, switch on <b>WebGPU</b>, reload this page.';
  } else if (safari) {
    how = 'Safari on macOS 26 has it on by default — updating is the easy fix. On earlier macOS: ' +
      '<b>Safari → Settings → Advanced</b>, tick “Show features for web developers”, then ' +
      '<b>Develop → Feature Flags</b>, switch on <b>WebGPU</b> and reload.';
  } else if (firefox) {
    how = 'Current Firefox ships it on Windows and macOS — updating usually fixes this.' +
      (linux ? ' On Linux, switch <b>dom.webgpu.enabled</b> on in <b>about:config</b> and restart.' : '');
  } else {
    how = 'Update the browser and make sure hardware acceleration is on ' +
      '(<b>Settings → System</b> in Chrome and Edge).' +
      (linux ? ' On Linux, also enable <b>chrome://flags/#enable-unsafe-webgpu</b> and restart.' : '');
  }
  const d = document.createElement('div');
  d.className = 'gpuwarn';
  d.innerHTML = `<b>The GPU is out of reach in this browser</b><span>` +
    `Camera solving and splat training need WebGPU, and this browser is not exposing it yet. ` +
    `Supported AR capture and capture export can still work; transfer the dataset to a GPU-capable device to train. ${how}</span>`;
  $('start').insertBefore(d, $('upload'));
}

/** the untouched start card: header static, no selection, no caption */
function showIntro() {
  S.preset = null;
  S.photos = [];
  $('strip').innerHTML = '';
  $('set-desc').hidden = true;
  $('btn-go').disabled = true;
  $('btn-settings').disabled = true;
  $('settings').hidden = true;
  $('btn-settings').setAttribute('aria-expanded', 'false');
  $('start').hidden = false;
}

/** Build an own-photos set from a stored capture record (fetches the record
 *  when none is passed). Records saved before capture-sorting existed replay
 *  in stored order — sorted on the way out, same rules as a fresh pick. */
async function openCaptureSet(rec = null) {
  rec = rec || await loadLastCapture();
  if (rec && rec.nativeCaptureId) {
    try {
      const capture = await importNativeCapture(globalThis.AndroidARBridge, rec.nativeCaptureId);
      rec = {
        ...rec, files: capture.files.map((f) => ({ name: f.name, blob: f, url: f.url })),
        ...storedCapture(capture), metadata: capture.metadata,
      };
    } catch (e) { throw new Error(`The native scan archive is unreadable: ${e.message}`); }
  }
  if (!rec || !rec.files || rec.files.length < 2) return null;
  const files = rec.files.map((e) => {
    const f = Object.assign(new File([e.blob], e.name, { type: e.blob.type || 'image/jpeg' }),
      { ...(e.url ? { url: e.url } : {}) });
    if (e.mask) f.mask = e.mask;   // avatar mode: the person's matte rides with the frame
    return f;
  });
  // an avatar capture restores as an avatar run (the manifest names the stages done so far)
  S.avatar = rec.avatar && rec.avatar.kind === 'avatar' ? rec.avatar : null;
  S.avatarOpts = S.avatar ? (await import('../avatar/index.js')).trainingOptions(S.avatar, { iters: S.settings.iters || AVATAR_ITERS, shHorizontal: !!S.settings.shup , av: S.settings.av }) : null;
  await sortByCapture(files);
  if (S.ownUrls) S.ownUrls.forEach(URL.revokeObjectURL);
  S.ownUrls = files.map((f) => URL.createObjectURL(f));
  const set = ownSet(files, S.ownUrls);
  set.id = '__last';
  set.kind = 'Saved on this device';
  set.video = rec.kind === 'video';
  set.avatar = S.avatar;
  set.source = rec.source || { picks: files.length };
  set.origin = `${files.length} frames from your last capture, restored from this browser's ` +
    'own storage. They never left this device.';
  if (rec.transforms) {
    set.capture = validateDataset({ kind: 'ar', files, transforms: rec.transforms, points: rec.points || [], pointsByFrame: rec.pointsByFrame, metadata: rec.metadata || {} });
    set.name = rec.metadata?.synthetic ? 'Synthetic Corner · known poses' : 'Your AR capture';
    set.kind = 'Known camera poses';
    set.origin = `${files.length} pose-tagged frames restored with their camera transforms; camera solve skipped.`;
  }
  return set;
}

/** The previous own capture, restored from this device's storage — a local-library tile. */
async function lastCaptureTile() {
  const rec = await loadLastCapture();
  // Native scans are listed from their immutable Android archives below; do
  // not duplicate their JPEGs (or their tile) in IndexedDB.
  if (!rec || rec.nativeCaptureId || !rec.files || rec.files.length < 2) return null;
  const b = document.createElement('div');
  b.className = 'galtile';
  b.style.cursor = 'pointer';
  b.title = `${rec.files.length} frames, saved on this device`;
  const capWhen = rec.created
    ? new Date(rec.created).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : null;
  b.innerHTML = `
    <button class="run-menu" title="Options">⋯</button>
    <span class="galname">${rec.files.length} photos</span>
    <span class="galmeta">${capWhen ? `${capWhen} · ` : ''}local</span>`;
  b.prepend(Object.assign(new Image(), { src: URL.createObjectURL(rec.files[0].blob), alt: '' }));
  const train = async () => {
    const set = await openCaptureSet(rec);
    if (!set) return;
    open(set);
    if (!WALL_FIRST) showDetail(set);
  };
  tileMenu(b.querySelector('.run-menu'), [
    { label: 'Train', act: train },
    { label: 'Delete', danger: true, act: async () => {
      const { deleteLastCapture } = await import('./store.js');
      await deleteLastCapture(); b.remove();
    } },
  ]);
  b.addEventListener('click', async () => {
    if (S.picking) { S.pending = await openCaptureSet(rec); paintCard(S.pending); return; }
    await train();
  });
  return b;
}

let legacyCaptureMigration = null;
let legacyCaptureRemounted = false;
/** Move an old IndexedDB-only AR capture into the canonical native archive.
 * This is migration, not duplication: the blob record is deleted only after
 * the native catalog verifies the complete archive. */
async function migrateLegacyNativeCapture() {
  const bridge = globalThis.AndroidARBridge;
  if (!bridge || typeof bridge.writeCaptureArchiveChunk !== 'function') return false;
  if (legacyCaptureMigration) return legacyCaptureMigration;
  legacyCaptureMigration = (async () => {
    const rec = await loadLastCapture();
    if (!rec || rec.nativeCaptureId || !rec.transforms || !Array.isArray(rec.files) || rec.files.length < 3) return false;
    const captureId = `legacy_${Math.max(1, Math.round(rec.created || Date.now()))}`;
    const fileMap = new Map();
    const files = rec.files.map((entry, i) => {
      const name = `frame_${String(i).padStart(4, '0')}.jpg`;
      fileMap.set(String(entry.name || '').split('/').pop(), name);
      return { ...entry, name };
    });
    const writeBytes = async (name, bytes) => {
      const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      for (let o = 0; o < data.length; o += 262144) {
        if (!bridge.writeCaptureArchiveChunk(captureId, name,
          Headless.encodeB64(data.subarray(o, o + 262144)), o + 262144 >= data.length)) {
          throw new Error(`native archive rejected ${name}`);
        }
      }
    };
    for (const file of files) {
      const blob = file.blob;
      if (!blob || typeof blob.arrayBuffer !== 'function') throw new Error(`legacy frame ${file.name} is missing`);
      await writeBytes(file.name, await blob.arrayBuffer());
    }
    const transforms = JSON.parse(JSON.stringify(rec.transforms));
    for (const [i, frame] of (transforms.frames || []).entries()) {
      const old = String(frame.file_path || '').split('/').pop();
      frame.file_path = `./${fileMap.get(old) || files[i]?.name || ''}`;
    }
    const points = (rec.points || []).map((point) => ({ ...point,
      ...(point.frameName ? { frameName: fileMap.get(String(point.frameName).split('/').pop()) || point.frameName } : {}) }));
    const enc = new TextEncoder();
    await writeBytes('transforms.json', enc.encode(JSON.stringify(transforms)));
    await writeBytes('seed_points.json', enc.encode(JSON.stringify({ points, metadata: rec.metadata || {} })));
    await writeBytes('capture.json', enc.encode(JSON.stringify({ ...(rec.metadata || {}), frames: files.length,
      pointCount: points.length, migratedFromIndexedDB: true })));
    const catalog = JSON.parse(bridge.listHomeLibrary());
    if (!(catalog.captures || []).some((row) => row.captureId === captureId)) {
      throw new Error('native archive verification failed');
    }
    const { deleteLastCapture, listRuns, patchRun } = await import('./store.js');
    // Repoint legacy model records before deleting their old image blobs.
    // Their model state stays their own; image/pose reads now resolve only
    // through the migrated canonical archive.
    for (const run of await listRuns()) {
      const source = run && run.recon && run.recon.source;
      if (!source || !Array.isArray(source.names) || !source.names.length) continue;
      const urls = source.names.map((name, i) => {
        const old = String(name || '').split('/').pop();
        const nativeName = fileMap.get(old) || files[i]?.name;
        return nativeName ? `https://appassets.androidplatform.net/captures/${captureId}/${nativeName}` : null;
      });
      if (!urls.every(Boolean)) continue;
      await patchRun(run.id, { srcKey: `native:${captureId}`,
        recon: { ...run.recon, source: { ...source, urls, captureId } } });
    }
    await deleteLastCapture();
    return true;
  })().catch((e) => { console.warn('[home] legacy AR migration failed', e); return false; });
  return legacyCaptureMigration;
}

/** Every native ARCore archive is a ground-truth scan tile. Only references
 * cross into JS; opening one fetches its files from the stable native folder. */
async function nativeCaptureTiles(rows = null) {
  const b = globalThis.AndroidARBridge;
  if (!b || typeof b.listCaptureArchives !== 'function') return [];
  if (rows == null) {
    try { rows = JSON.parse(b.listCaptureArchives()); } catch { return []; }
  }
  if (!Array.isArray(rows)) return [];
  return rows.map((row) => {
    const tile = document.createElement('div');
    tile.className = 'galtile'; tile.style.cursor = 'pointer';
    const when = row.created ? new Date(row.created).toLocaleString([],
      { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : null;
    tile.innerHTML = `<button class="run-menu" title="Options">⋯</button>
      <span class="galname">${fmt(row.frameCount || 0)} ARCore photos</span>
      <span class="galmeta">${when ? `${when} · ` : ''}native scan</span>`;
    if (row.previewUri) tile.prepend(Object.assign(new Image(), { src: row.previewUri, alt: '' }));
    tileMenu(tile.querySelector('.run-menu'), [
      { label: 'Delete', danger: true, act: async () => {
        let answer = { deleted: false, reason: 'unsupported' };
        try { answer = JSON.parse(b.deleteCaptureArchive(row.captureId)); } catch { /* report below */ }
        if (answer.deleted) tile.remove();
        else if (answer.reason === 'referenced') flash('Delete its trained models first; this scan is still their source.', 7000);
        else flash('This scan could not be deleted.', 6000);
      } },
    ]);
    tile.addEventListener('click', async () => {
      try {
        const capture = await importNativeCapture(b, row.captureId);
        const rec = { kind: 'photos', created: row.created || Date.now(),
          nativeCaptureId: row.captureId,
          files: capture.files.map((f) => ({ name: f.name, blob: f, url: f.url })),
          ...storedCapture(capture) };
        const set = await openCaptureSet(rec);
        if (!set) return;
        open(set); if (!WALL_FIRST) showDetail(set);
      } catch (e) { flash(`Could not open native scan: ${e.message}`, 8000); }
    });
    return tile;
  });
}


/** tiles for the local runs library — every training run this device has
 *  started. Finished runs reopen in the viewer straight from their stored
 *  result; interrupted ones stay listed (and deletable) as a record. */
/** The tile's "⋯" options menu. items: [{ label, act, danger }] — danger
 *  items arm into "Delete?" first; anything else fires and closes. */
function tileMenu(btn, items) {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const existing = btn.parentElement.querySelector('.tilemenu');
    document.querySelectorAll('.tilemenu').forEach((x) => x.remove());
    if (existing) return; // second tap on the same ⋯ just closes
    const m = document.createElement('div');
    m.className = 'tilemenu';
    for (const it of items) {
      if (!it) continue;
      const b = document.createElement('button');
      b.textContent = it.label;
      if (it.danger) b.className = 'danger';
      b.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        if (it.danger && b.dataset.armed !== '1') {
          b.dataset.armed = '1';
          b.textContent = 'Delete?';
          b.classList.add('armed');
          setTimeout(() => {
            b.dataset.armed = '';
            b.textContent = it.label;
            b.classList.remove('armed');
          }, 3000);
          return;
        }
        m.remove();
        await it.act();
      });
      m.appendChild(b);
    }
    btn.parentElement.appendChild(m);
    const close = (ev) => {
      if (m.contains(ev.target) || ev.target === btn) return;
      m.remove();
      removeEventListener('pointerdown', close, true);
    };
    addEventListener('pointerdown', close, true);
  });
}

async function localRunTiles(nativeRows = [], captureRows = []) {
  const { listRuns, deleteRun } = await import('./store.js');
  const runs = await listRuns();
  const nativeByRun = new Map((nativeRows || []).map((row) => [row.runId, row]));
  const captureById = new Map((captureRows || []).map((row) => [row.captureId, row]));
  const tiles = [];
  for (const r of runs) {
    const live = S.runId === r.id && S.state === 'train';
    const src = r.recon && r.recon.source;
    const urlSrc = !!(src && Array.isArray(src.urls) && src.urls.length && src.urls.every(Boolean));
    const retrainable = !!(r.ownSrc || urlSrc);
    const b = document.createElement('div');
    b.className = 'galtile';
    b.dataset.runId = r.id;
    b.style.cursor = (r.sog || (r.state && r.recon) || retrainable) ? 'pointer' : 'default';
    // finished scenes wear the same sub-header as the preset tiles:
    // splats · dB · MB — a scene is a scene, wherever it was trained
    const state = r.status === 'finished'
      ? `${fmt(r.splats || 0)} splats${r.psnr != null ? ` · ${(+r.psnr).toFixed(1)} dB` : ''}${r.sog ? ` · ${Math.max(1, Math.round(r.sog.size / 1e6))} MB` : ''}`
      : live
        ? `training now · ${fmt(r.iter || 0)} cycles`
        : (r.state && r.recon)
          ? `paused · ${fmt(r.iter || 0)} cycles — tap to continue`
          : `interrupted · ${fmt(r.iter || 0)} cycles — tap to train again`;
    // same anatomy as a preset tile: name, a description referring back to
    // the capture, then the stats line
    const desc = r.status === 'finished'
      ? `Trained on this device from ${r.frames ? `${fmt(r.frames)} photos` : 'your photos'}` +
        `${r.iter ? `, ${fmt(r.iter)} cycles` : ''}${r.minutes ? ` in ${r.minutes} min` : ''}. ${r.spaceId ? 'Shared — anyone with the link can view it.' : 'Never uploaded.'}`
      : '';
    if (r.spaceId) b.dataset.spaceId = r.spaceId;
    b.innerHTML = `
      <button class="run-menu" title="Options">⋯</button>
      <span class="galname">${esc(r.name || 'Training run')}</span>
      ${desc ? `<span class="galdesc">${esc(desc)}</span>` : ''}
      <span class="galmeta">${state}</span>`;
    if (r.thumb) b.prepend(Object.assign(new Image(), { src: URL.createObjectURL(r.thumb), alt: '' }));
    else {
      const nativeRow = nativeByRun.get(r.id);
      const captureId = String(r.srcKey || '').replace(/^native:/, '');
      const fallbackPreview = nativeRow?.previewUri || captureById.get(captureId)?.previewUri;
      if (fallbackPreview) b.prepend(Object.assign(new Image(), { src: fallbackPreview, alt: '' }));
    }
    const viewFromState = (rec) => {
      S._localRun = rec;
      continueLocalRun(true).catch((e) => {
        console.error(e);
        flash(`Could not open this run: ${e.message}`, 8000);
      });
    };
    const openRun = () => restoreSession({
      url: URL.createObjectURL(r.sog),
      reconUrl: URL.createObjectURL(new Blob([JSON.stringify(r.recon)], { type: 'application/json' })),
      localRun: r,
    });
    const resumeRun = () => {
      S._localRun = r;
      continueLocalRun().catch((e) => {
        console.error(e);
        flash(`Could not continue this run: ${e.message}`, 8000);
      });
    };
    const viewable = !!(r.recon && (r.sog || (r.status === 'finished' && r.state)));
    const renameLocalModel = async () => {
      const previousName = r.name || 'Training run';
      const name = await textCard('Rename this model', previousName);
      if (!name || name === previousName) return;
      try {
        const { patchRun } = await import('./store.js');
        if (!await patchRun(r.id, { name })) {
          flash('Could not save the new model name on this device.', 6000);
          return;
        }
        await renameAndroidRun(r.id, name);
        r.name = name;
        const title = b.querySelector('.galname');
        if (title) title.textContent = name;
        flash(`Renamed to ${name}.`, 4000);
      } catch (e) { flash(`Rename failed: ${e.message}`, 6000); }
    };
    tileMenu(b.querySelector('.run-menu'), [
      viewable && { label: 'View', act: r.sog ? openRun : () => viewFromState(r) },
      (r.sog || retrainable || (r.state && r.recon)) && { label: 'Train', act: () => { S._localRun = r; trainLocalChoice(); } },
      { label: 'Rename', act: renameLocalModel },
      r.spaceId && { label: 'Copy link', act: async () => { try { await navigator.clipboard.writeText(shareLinkOf(r.spaceId)); flash('Link copied.', 2500); } catch { flash(shareLinkOf(r.spaceId), 8000); } } },
      { label: 'Delete', danger: true, act: async () => {
        const bridge = headlessBridge();
        if (bridge && typeof bridge.deleteRunArchive === 'function' && bridge.deleteRunArchive(r.id) === false) {
          flash('A running model cannot be deleted.', 6000); return;
        }
        await deleteRun(r.id); b.remove();
      } },
    ]);
    b.addEventListener('click', () => {
      if (r.sog && r.recon) {
        restoreSession({
          url: URL.createObjectURL(r.sog),
          reconUrl: URL.createObjectURL(new Blob([JSON.stringify(r.recon)], { type: 'application/json' })),
          localRun: r, // the viewer's Train button reads this
        });
      } else if (r.status === 'finished' && r.state && r.recon) {
        // finished but never compressed: the viewer reopens from the raw
        // state (Compress there turns it into a sog tile)
        viewFromState(r);
      } else if (r.state && r.recon && !(S.runId === r.id && S.state === 'train')) {
        // a leave checkpoint: no sog to view yet, but the run continues
        // exactly where it stopped
        resumeRun();
      } else if (retrainable && !(S.runId === r.id && S.state === 'train')) {
        // interrupted before any result — but the photos are reachable, so
        // the tap leads straight back into training
        S._localRun = r;
        trainLocalChoice();
      }
    });
    tiles.push(b);
  }
  return tiles;
}


async function renameAndroidRun(runId, name) {
  const bridge = headlessBridge();
  if (!bridge || typeof bridge.writeRunJson !== 'function' || !Headless.validRunId(runId)) return false;
  try {
    const response = await fetch(Headless.fresh(Headless.runUrl(runId, 'manifest.json')));
    if (!response.ok) return false;
    const manifest = await response.json();
    manifest.name = name;
    const ok = bridge.writeRunJson(runId, 'manifest.json', JSON.stringify(manifest));
    if (ok && S._hlManifest?.runId === runId) S._hlManifest = manifest;
    return !!ok;
  } catch { return false; }
}

/** Lightweight Home tiles for checkpoints in Android app data. Unlike the
 * old rediscovery pass, this does not copy every state.bin into IndexedDB
 * before Home can paint. The full checkpoint is loaded only when tapped. */
async function androidRunTiles(ids = null) {
  const bridge = headlessBridge();
  if (!bridge || typeof bridge.listRecoverableRuns !== 'function') return [];
  if (ids == null) {
    try { ids = JSON.parse(bridge.listRecoverableRuns()); } catch { return []; }
  }
  if (!Array.isArray(ids) || !ids.length) return [];
  const { deleteRun } = await import('./store.js');
  const tiles = [];
  for (const entry of ids) {
    const runId = typeof entry === 'string' ? entry : entry && entry.runId;
    if (!Headless.validRunId(runId)) continue;
    try {
      let manifest = null;
      let status = { status: 'interrupted', owner: 'activity', iter: 0, at: 0 };
      let photoCount = 0;
      if (entry && typeof entry === 'object') {
        manifest = { name: entry.name || 'Training run' };
        photoCount = entry.photoCount | 0;
        status = { ...status, ...entry };
      } else {
        const manifestResponse = await fetch(Headless.fresh(Headless.runUrl(runId, 'manifest.json')));
        if (!manifestResponse.ok) continue;
        manifest = await manifestResponse.json();
        photoCount = Array.isArray(manifest.photos) ? manifest.photos.length : 0;
        try {
          const response = await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')));
          if (response.ok) status = { ...status, ...(await response.json()) };
        } catch { /* checkpoint remains usable without control metadata */ }
      }
      const tile = document.createElement('div');
      tile.className = 'galtile'; tile.style.cursor = 'pointer'; tile.dataset.runId = runId;
      const finished = status.status === 'finished';
      const state = finished
        ? `${fmt(status.splats || 0)} splats${status.psnr != null ? ` · ${(+status.psnr).toFixed(1)} dB` : ''}`
        : `saved checkpoint · ${fmt(status.iter || 0)} cycles`;
      tile.innerHTML = `<button class="run-menu" title="Options">⋯</button>
        <span class="galname">${esc(manifest.name || 'Training run')}</span>
        <span class="galdesc">Stored in Android app data${photoCount ? ` · ${fmt(photoCount)} photos` : ''}.</span>
        <span class="galmeta">${state}</span>`;
      if (entry && typeof entry === 'object' && entry.previewUri) {
        tile.prepend(Object.assign(new Image(), { src: entry.previewUri, alt: '' }));
      }
      const openFileRun = async () => {
        flash('Opening the saved model …', 60000);
        try {
          const rec = await mirrorHeadlessRun(runId, status);
          S._localRun = rec;
          await continueLocalRun(finished);
        } catch (e) {
          console.error(e);
          flash(`Could not open this saved model: ${e.message}`, 9000);
        }
      };
      tileMenu(tile.querySelector('.run-menu'), [
        { label: finished ? 'View' : 'Continue', act: openFileRun },
        { label: 'Rename', act: async () => {
          const previousName = manifest.name || 'Training run';
          const name = await textCard('Rename this model', previousName);
          if (!name || name === previousName) return;
          if (!(await renameAndroidRun(runId, name))) {
            flash('Could not save the new model name on this device.', 6000); return;
          }
          manifest.name = name;
          const title = tile.querySelector('.galname');
          if (title) title.textContent = name;
          flash(`Renamed to ${name}.`, 4000);
        } },
        { label: 'Delete', danger: true, act: async () => {
          if (typeof bridge.deleteRunArchive === 'function' && bridge.deleteRunArchive(runId) === false) {
            flash('A running model cannot be deleted.', 6000); return;
          }
          await deleteRun(runId); tile.remove();
        } },
      ]);
      tile.addEventListener('click', openFileRun);
      tiles.push(tile);
    } catch (e) { console.warn(`[home] unreadable Android run ${runId}`, e); }
  }
  return tiles;
}

/** the first `cnt` photos of a preset, honouring its skip list */
function presetPhotoList(preset, cnt) {
  // a set rebuilt from stored addresses (an interrupted run's record)
  // carries its photo list ready-made
  if (preset.urlList) return preset.urlList.slice(0, cnt || preset.urlList.length);
  const skip = new Set(preset.skip || []);
  const out = [];
  const start = preset.names ? 0 : preset.start;
  const limit = preset.names ? preset.names.length : Infinity;
  for (let k = 0, i = start; k < cnt && i < limit; i++) {
    if (skip.has(i)) continue;
    const url = presetUrl(preset, i);
    out.push({ url, name: url.split('/').pop() });
    k++;
  }
  return out;
}

function presetUrl(p, i) {
  if (p.names) return `${DATA}${p.dir}/${p.names[i]}`;
  return `${DATA}${p.dir}/` +
    p.pattern.replace(/\{i:(\d+)\}/, (_, w) => String(i).padStart(+w, '0'));
}

/** list-based presets fetch their file list once (BOM-tolerant) */
async function loadPresetList(p) {
  if (!p.list || p.names) return;
  const t = await (await fetch(`${DATA}${p.dir}/${p.list}`)).text();
  p.names = JSON.parse(t.replace(/^﻿/, ''));
  p.maxCount = Math.min(p.maxCount || p.names.length, p.names.length);
}

/** mouse drag-to-scroll for the horizontal rows (their scrollbars are
 *  hidden; touch pans natively via touch-action). A real drag captures the
 *  pointer and swallows the click that would otherwise hit a tile. */
function dragScroll(el) {
  let x0 = 0, s0 = 0, moved = 0, down = false;
  // images/tiles must not become native drag payloads — that ate the swipe
  el.addEventListener('dragstart', (e) => e.preventDefault());
  el.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'mouse' || e.button !== 0) return;
    down = true; moved = 0; x0 = e.clientX; s0 = el.scrollLeft;
  });
  el.addEventListener('pointermove', (e) => {
    if (!down) return;
    const dx = e.clientX - x0;
    if (Math.abs(dx) > 4 && moved <= 4) {
      try { el.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
      el.style.cursor = 'grabbing';
      document.body.style.cursor = 'grabbing';   // capture outlives the row
    }
    moved = Math.max(moved, Math.abs(dx));
    if (moved > 4) el.scrollLeft = s0 - dx;
  });
  const end = () => { down = false; el.style.cursor = ''; document.body.style.cursor = ''; };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el.addEventListener('click', (e) => {
    if (moved > 4) { e.stopPropagation(); e.preventDefault(); moved = 0; }
  }, true);
}

/** wall-time estimate: each set's measured time at its default count,
 *  scaled for other counts (training is ~fixed, pair matching is O(n²)) */
function approxFor(preset, n) {
  const base = parseInt((preset.approx || '').replace(/\D+/g, ''), 10);
  if (!base || !preset.count || n === preset.count) return preset.approx;
  const q = (n * n) / (preset.count * preset.count);
  return `~${Math.max(2, Math.round(base * (0.4 + 0.6 * q)))} min`;
}

function paintCard(preset) {
  // the header stays the product's; the selection describes itself in a
  // caption attached to the preset row
  const links = (preset.links || [])
    .map((l) => `<a href="${l.url}" target="_blank" rel="noopener">${l.label}</a>`).join(' · ');
  const cnt = preset.useCount || preset.count;
  $('set-desc').innerHTML = `<b>${preset.name}</b> — ${preset.origin}` +
    (links ? ` ${links}` : '') +
    `<span class="approx">${approxFor(preset, cnt)} on a fast GPU</span>`;
  $('set-desc').hidden = false;
  // the photo count lives in the settings panel — any fetched set can be
  // trimmed (or extended up to what exists on disk)
  const mx = preset.files ? 0 : (preset.maxCount || preset.count);
  $('row-count').hidden = mx < 3;
  if (mx >= 3) {
    $('set-count').max = mx;
    $('set-count').value = cnt;
    $('set-count-v').textContent = `${cnt} / ${mx}`;
  }
  $('btn-go').textContent = 'Start training';
}

/** re-cut the photo list to the chosen count (only while on the start card
 *  with this preset live — in the mid-run picker the choice applies when the
 *  switch commits through open()) */
function applyCount(preset) {
  const cnt = preset.useCount || preset.count;
  const ap = $('set-desc').querySelector('.approx');
  if (ap) ap.textContent = `${approxFor(preset, cnt)} on a fast GPU`;
  if (S.state === 'ready' && !S.picking && S.preset === preset && !preset.files) {
    S.photos = presetPhotoList(preset, cnt);
    buildStrip(true);   // still behind the card — no thumb fetches yet
  }
}

// (the old mid-run picker — the wall shown OVER a live run — is gone on
// purpose: leaving a run is a page navigation home, never an overlay)

function closePicker() {
  S.picking = false; S.pending = null;
  $('start').hidden = true;
  $('card-x').hidden = true;
  $('btn-go').textContent = 'Start training';
}

/** EXIF DateTimeOriginal (ms since epoch) from a JPEG's APP1 block — null
 *  when absent. lastModified is NOT a substitute on iOS: the picker often
 *  transcodes on selection and stamps THAT moment, shuffling the walk. */
async function exifCaptureTime(file) {
  try {
    const b = new Uint8Array(await file.slice(0, 262144).arrayBuffer());
    if (b[0] !== 0xff || b[1] !== 0xd8) return null;
    let p = 2;
    while (p + 4 < b.length) {
      if (b[p] !== 0xff) return null;
      const m = b[p + 1];
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { p += 2; continue; }
      if (m === 0xda || m === 0xd9) return null; // image data: no EXIF ahead
      const len = (b[p + 2] << 8) | b[p + 3];
      if (m === 0xe1 && len > 10 && b[p + 4] === 0x45 && b[p + 5] === 0x78 &&
          b[p + 6] === 0x69 && b[p + 7] === 0x66 && b[p + 8] === 0 && b[p + 9] === 0) {
        return tiffDate(b, p + 10, Math.min(len - 8, b.length - (p + 10)));
      }
      p += 2 + len;
    }
    return null;
  } catch { return null; }
}
function tiffDate(b, off, size) {
  if (size < 8) return null;
  const dv = new DataView(b.buffer, b.byteOffset + off, size);
  const le = dv.getUint16(0) === 0x4949;
  const u16 = (o) => dv.getUint16(o, le);
  const u32 = (o) => dv.getUint32(o, le);
  if (u16(2) !== 42) return null;
  const scan = (ifdOff, tags) => {
    const out = {};
    if (ifdOff + 2 > size) return out;
    const n = u16(ifdOff);
    for (let i = 0; i < n; i++) {
      const e = ifdOff + 2 + i * 12;
      if (e + 12 > size) break;
      const tag = u16(e);
      if (tags.includes(tag)) out[tag] = { type: u16(e + 2), count: u32(e + 4), value: u32(e + 8) };
    }
    return out;
  };
  const ifd0 = scan(u32(4), [0x8769, 0x0132]);
  let at = null;
  if (ifd0[0x8769]) {
    const exif = scan(ifd0[0x8769].value, [0x9003, 0x9004]); // DateTimeOriginal, Digitized
    const d = exif[0x9003] || exif[0x9004];
    if (d && d.type === 2 && d.count >= 19) at = d.value;
  }
  if (at == null && ifd0[0x0132] && ifd0[0x0132].type === 2 && ifd0[0x0132].count >= 19) at = ifd0[0x0132].value;
  if (at == null || at + 19 > size) return null;
  let s = '';
  for (let i = 0; i < 19; i++) s += String.fromCharCode(dv.getUint8(at + i));
  const mm = s.match(/^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  if (!mm) return null;
  const t = new Date(+mm[1], +mm[2] - 1, +mm[3], +mm[4], +mm[5], +mm[6]).getTime();
  return Number.isFinite(t) && t > 0 ? t : null;
}

/** Restore the capture sequence in place. EXIF time rules when present;
 *  when the WHOLE set has none (iOS transcodes on pick and strips EXIF,
 *  stamping mtime with the SELECTION moment — sorting by it preserves the
 *  shuffle), the numeric name order (IMG_0421…, wide_date_time_seq…) is the
 *  trustworthy key. Also feeds the landmarks beat's bottom-right overlay. */
async function sortByCapture(files) {
  const exifs = new Map(await Promise.all(files.map(async (f) => [f, await exifCaptureTime(f)])));
  const anyExif = [...exifs.values()].some((v) => v != null);
  const byName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
  if (anyExif) {
    const stamps = new Map(files.map((f) => [f, exifs.get(f) ?? f.lastModified ?? 0]));
    files.sort((a, b) => (stamps.get(a) - stamps.get(b)) || byName(a, b));
  } else {
    files.sort(byName);
  }
  S.capDates = new Map(files.map((f) => [f.name, exifs.get(f) ?? null]));
  return files;
}

/** Keep the capture on-device, and KNOW whether it worked. The old fire-and-
 *  forget swallowed failures (strict storage, quota, a write starved by the
 *  solve's CPU rush) — the visitor then found no trace of their photos. */
function persistCapture(rec) {
  S._capRec = rec;
  S._capSave = saveLastCapture(rec).then(() => 'ok', (e) => {
    console.warn('capture save failed', e);
    return 'failed';
  });
}

// Async intake funnels through here (file picker, drops, captures, zips):
// the landing set's open() bumps S.gen, which can kill startPrep in flight.
// Start now queues behind this counter instead of making the user tap twice.
async function useOwnPhotos(list, capture = null) {
  S._intakeN = (S._intakeN || 0) + 1;
  try { return await useOwnPhotosInner(list, capture); }
  finally { S._intakeN--; }
}
async function useOwnPhotosInner(list, capture = null) {
  // capture zips (exported photos + camera JSON) unpack to files and join
  // the flow below; a session zip restores instead of training
  const all = [];
  let fromZip = false;
  const zipSkipped = [];
  for (const f of list) {
    if (f && /\.zip$/i.test(f.name || '')) {
      try {
        const u = await unpackZipFile(f);
        if (u.kind === 'session') { restoreSession({ bytes: u.bytes }); return; }
        if (u.kind === 'capture') {
          fromZip = true;
          all.push(...u.files);
          for (const s of u.skipped || []) zipSkipped.push(s);
          continue;
        }
        flash(`"${f.name}" holds no photos or session — skipped.`, 5000);
      } catch (e) { flash(`Could not unpack "${f.name}": ${e.message}`, 8000); }
      continue;
    }
    all.push(f);
  }
  if (!capture && all.some((f) => /\.json$/i.test(f.name))) {
    try {
      capture = await reviewCapture(await datasetFromSelection(all, fromZip ? { fromZip: true } : {}));
      if (!capture) return;
      // folder zips carry strays (OS metadata, thumbnails, second copies):
      // name them so the review count never surprises
      const leftOut = [...zipSkipped, ...(capture.dropped || [])];
      if (leftOut.length) {
        flash(`Left out of the zip import: ${leftOut.slice(0, 3).join('; ')}` +
          `${leftOut.length > 3 ? `; +${leftOut.length - 3} more` : ''}.`, 9000);
      }
    } catch (e) { flash(`Could not import capture: ${e.message}`, 9000); return; }
  }
  // video intake is back on (2026-09-08): the v2 extractor decodes every frame
  // through WebCodecs, scores it with the sharp-frames metric and picks by
  // camera motion — a video without photos goes through useOwnVideo
  const video = all.find(isVideoFile);
  const files = capture ? [...capture.files] : all.filter((f) => f.type.startsWith('image/'));
  if (video && files.length < 2) { useOwnVideo(video); return; }
  if (files.length < 2) {
    flash('Pick at least a couple of overlapping photos of the same place.', 4500);
    return;
  }
  // the phone picker hands files over in SELECTION order — restore the
  // capture sequence (shared with the saved-capture restore path)
  await sortByCapture(files);
  if (S.ownUrls) S.ownUrls.forEach(URL.revokeObjectURL);
  S.ownUrls = files.map((f) => URL.createObjectURL(f));
  // survive a refresh: the capture is kept on-device and offered back
  const nativeCaptureId = capture?.metadata?.nativeCaptureId || null;
  persistCapture({
    kind: 'photos', created: Date.now(),
    ...(capture ? storedCapture({ ...capture, files: nativeCaptureId ? [] : files }) : {}),
    ...(nativeCaptureId ? { nativeCaptureId } : {}),
    files: files.map((f) => nativeCaptureId
      ? { name: f.name, url: f.url }
      : { name: f.name, blob: f }),
  });
  const set = ownSet(files, S.ownUrls);
  if (capture?.transforms) {
    validateDataset({ ...capture, files });
    set.capture = { ...capture, files };
    set.name = capture.metadata?.synthetic ? 'Synthetic Corner · known poses' : 'Your AR capture';
    set.kind = 'Known camera poses';
    const seeds = measuredSeeds(capture).length;
    set.origin = `${files.length} pose-tagged views · camera solve skipped · ${seeds ? `${fmt(seeds)} measured seed points` : 'random-volume seed (orbit required)'}.`;
  }
  await open(set);
  showDetail(set);   // Start training lives on the detail card
}

/** A video: every frame is scored (sharpness, motion, cuts), the picks are
 *  shown on a timeline the user can trim, then the winners are captured and
 *  continue exactly like a photo set. */
// video intake lands through open() too — same guard as photo sets above
async function useOwnVideo(file) {
  S._intakeN = (S._intakeN || 0) + 1;
  try { return await useOwnVideoInner(file); }
  finally { S._intakeN--; }
}
async function useOwnVideoInner(file) {
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'vidcard';
  const meter = (title, sub) => {
    card.innerHTML = `
      <div class="vid-head"><b>${title}</b><span class="prep-sub" id="vid-sub">${sub}</span></div>
      <div class="prep-meter"><i id="vid-bar" style="width:0%"></i></div>`;
  };
  meter('Reading your video', 'decoding …');
  $('stage').appendChild(card);
  const LABEL = { scan: 'scoring every frame', capture: 'saving the winners' };
  let videoMeta = {};
  try {
    const { frames, duration } = await extractSharpFrames(file, {
      thumbs: { width: window.innerWidth <= 560 ? 112 : 224 },   // scan-time thumbnails for the review card's filmstrip and hover preview
      log: (m) => console.log('[video]', m),
      onProgress: (e) => {
        const bar = $('vid-bar');
        if (!bar) return;
        const half = e.stage === 'scan' ? 0 : 50;
        bar.style.width = `${half + (e.done / e.total) * 50}%`;
        $('vid-sub').textContent = `${LABEL[e.stage]} · ${e.done} / ${e.total}`;
      },
      review: async (ctx) => {
        const r = await videoReview(card, ctx);
        // every pick is kept here — an avatar run thins them at Start training
        // (avatarFrames), because that is where the choice is made now
        if (r) { videoMeta = { videoW: ctx.videoW, videoH: ctx.videoH }; meter('Saving the frames', `0 / ${r.picks.length}`); }
        return r;
      },
    });
    if (frames.length < 12) {
      flash('That video is too short — a slow 20+ second pass works best.', 6000);
      return;
    }
    // the container's lens data (iPhone MOV: lens model, 35 mm-equivalent focal): every
    // frame carries it as its EXIF, so the solver starts from a focal prior instead of
    // its search — an orbit around a person has weak focal observability (Tom's clip
    // solved to 0.44x, 0.55x and 0.78x maxDim at three feature resolutions, 2026-09-15)
    try {
      const { readVideoMeta } = await import('../../src/io/qtmeta.js');
      const vm = await readVideoMeta(file);
      if (vm.focal35) {
        for (const f of frames) f.exif = { f35: vm.focal35, lens: vm.lensModel, video: true };
        videoMeta.focal35 = vm.focal35; videoMeta.lens = vm.lensModel;
        console.log(`[video] lens: ${vm.lensModel || '?'} · ${vm.focal35} mm (35 mm-equivalent) · f/${vm.fNumber || '?'} — focal prior for the solve`);
      }
    } catch (e) { console.log('[video] lens metadata: ' + (e.message || e)); }
    // the avatar decision (and the matting minutes it costs) waits for the
    // start-training card: app.js avatarPrep(), on Start training
    S.avatar = null; S.avatarOpts = null;
    S.avatarSource = { video: file.name, picks: frames.length, duration, ...videoMeta };
    if (S.ownUrls) S.ownUrls.forEach(URL.revokeObjectURL);
    S.ownUrls = frames.map((f) => URL.createObjectURL(f.source));
    // persist the EXTRACTED frames (small JPEGs), not the raw video
    persistCapture({
      kind: 'video', created: Date.now(),
      files: frames.map((f) => ({ name: f.name, blob: f.source, ...(f.mask ? { mask: f.mask } : {}) })),
      source: S.avatarSource,   // the clip's name and size, for an avatar manifest later
      ...(S.avatar ? { avatar: S.avatar } : {}),
    });
    const set = ownSet(frames, S.ownUrls);
    set.kind = 'Your video';
    set.video = true;   // sequential frames: the solver's neighbour gate applies (VIDEO_GATES)
    set.source = S.avatarSource;   // the avatar manifest is built from this, if one is asked for
    set.origin = `${frames.length} sharp frames picked from your ${Math.round(duration)}s video, ` +
      'right here in this tab. Blurred moments lost to their sharper neighbours.';
    open(set);
    showDetail(set);   // Start training lives on the detail card (the photo path does the same)
  } catch (e) {
    if (e && e.message === 'cancelled') return;
    console.error(e);
    flash(`Could not read that video: ${e.message}`, 8000);
  } finally {
    card.remove();
  }
}

/** The review card: the scan's scores as a timeline (focus curve, blur dips,
 *  cuts, picks), a draggable time range, and a filmstrip of the picks. Resolves
 *  { picks } on "Use these frames", null on cancel. */
function videoReview(card, ctx) {
  return new Promise((resolve) => {
    const { frames, duration, thumbs, cap } = ctx;
    const full = { start: frames[0].t, end: frames[frames.length - 1].t };
    let range = ctx.suggested ? { ...ctx.suggested } : { ...full };
    let gapSec = null;   // preferred average spacing (null = natural density)
    const gap = () => (gapSec == null ? {} : { targetGapSec: gapSec });
    let plan = ctx.plan(ctx.suggested ? range : null, gap());
    S.videoReview = ctx;   // console / e2e access
    const fmt = (t) => `${t.toFixed(1)} s`;
    card.innerHTML = `
      <button class="card-x" id="vid-x" aria-label="Cancel">&times;</button>
      <div class="vid-head"><b>Your video</b><span class="prep-sub">${frames.length} frames scored · ${fmt(duration)} · ${ctx.videoW}×${ctx.videoH}</span></div>
      <div class="vid-legend"><span><i class="focus"></i>sharpness</span><span><i class="pick"></i>picked frame</span><span><i class="blur"></i>blur dip</span><span><i class="cut"></i>cut</span><span><i class="range"></i>range used</span></div>
      <canvas class="vid-tl" id="vid-tl"></canvas>
      <canvas class="vid-strip" id="vid-strip"></canvas>
      <div class="vid-gap"><label for="vid-gap-sel">Average spacing</label><select id="vid-gap-sel"><option value="">Auto</option><option value="0.5">0.5 s</option><option value="1">1 s</option><option value="2">2 s</option><option value="5">5 s</option><option value="10">10 s</option></select></div>
      <div class="vid-read"><b id="vid-n"></b><span id="vid-span"></span><button class="linkish" id="vid-reset" hidden>whole video</button></div>
      <div class="upcard-row"><p class="fine" id="vid-hint"></p><span style="display:flex;gap:8px"><button class="btn btn-outline" id="vid-cancel">Cancel</button><button class="btn btn-accent" id="vid-use">Use these frames</button></span></div>`;
    const tl = $('vid-tl'), strip = $('vid-strip');
    if (ctx.videoH > ctx.videoW) strip.style.height = (window.innerWidth <= 560 ? 150 : 110) + 'px';   // portrait clips: taller filmstrip
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const PAD = 10, LANE = 16;   // px: side padding, picks lane at the bottom
    let W = 0, H = 0;
    let hover = null;   // 'start' | 'end' | 'move' | null — what the pointer is over
    let previewT = null;   // time under the pointer (or under the dragged handle): its frame is shown
    const thumbAt = (t) => { let th = thumbs[0]; for (const c of thumbs) if (Math.abs(c.t - t) < Math.abs(th.t - t)) th = c; return th; };
    const frameAt = (t) => { let lo = 0, hi = frames.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (frames[m].t < t) lo = m + 1; else hi = m; } return frames[lo]; };
    const size = (cv) => {
      const r = cv.getBoundingClientRect();
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
      const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      return [g, r.width, r.height];
    };
    const xOf = (t) => PAD + (t / duration) * (W - 2 * PAD);
    const tOf = (x) => Math.max(0, Math.min(duration, ((x - PAD) / (W - 2 * PAD)) * duration));
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    const C = { accent: css('--accent'), accent14: css('--accent-14'), accent30: css('--accent-30'), paper: css('--paper'), muted: css('--muted'), dim: css('--dim'), red: css('--red') };
    // focus normalised by its 95th percentile so one bright spike does not flatten the curve
    const sorted = frames.map((f) => f.focus).sort((a, b) => a - b);
    const fMax = Math.max(1, sorted[Math.floor(sorted.length * 0.95)]);
    const whole = () => Math.abs(range.start - full.start) < 1e-6 && Math.abs(range.end - full.end) < 1e-6;

    function drawTimeline() {
      const [g, w, h] = size(tl); W = w; H = h;
      g.clearRect(0, 0, w, h);
      const top = 22, bottom = h - LANE - 6, ph = bottom - top;
      // range fill
      g.fillStyle = C.accent14;
      g.fillRect(xOf(range.start), top - 4, xOf(range.end) - xOf(range.start), bottom - top + LANE + 8);
      // focus curve: one point per frame when the frames are sparser than the
      // pixels, else the max per pixel column (a long video on a wide screen)
      const cols = Math.max(1, Math.floor(w - 2 * PAD));
      const pts = [];
      if (frames.length <= 2 * cols) {
        for (const f of frames) pts.push([xOf(f.t), bottom - Math.min(1, f.focus / fMax) * ph]);
      } else {
        const colMax = new Float32Array(cols);
        for (const f of frames) { const c = Math.min(cols - 1, Math.floor(((f.t / duration) * cols))); if (f.focus > colMax[c]) colMax[c] = f.focus; }
        for (let c = 0; c < cols; c++) pts.push([PAD + c, bottom - Math.min(1, colMax[c] / fMax) * ph]);
      }
      g.beginPath(); g.moveTo(pts[0][0], bottom);
      for (const [x, y] of pts) g.lineTo(x, y);
      g.lineTo(pts[pts.length - 1][0], bottom); g.closePath();
      g.fillStyle = C.accent30; g.fill();
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.strokeStyle = C.accent; g.lineWidth = 1.25; g.stroke();
      // blur dips
      g.fillStyle = C.red;
      for (const f of frames) if (f.blur) g.fillRect(xOf(f.t) - 0.5, bottom - 6, 1, 6);
      // cuts
      g.setLineDash([3, 3]); g.strokeStyle = C.muted;
      for (const sh of ctx.shots.slice(1)) { const x = xOf(frames[sh.start].t); g.beginPath(); g.moveTo(x, top - 4); g.lineTo(x, bottom + LANE); g.stroke(); }
      g.setLineDash([]);
      // picks lane
      g.fillStyle = C.paper;
      for (const i of plan.picks) g.fillRect(xOf(frames[i].t) - 0.5, bottom + 4, 1, LANE - 6);
      // dim outside the range
      g.fillStyle = 'rgba(7, 9, 9, .55)';
      g.fillRect(0, 0, xOf(range.start), h); g.fillRect(xOf(range.end), 0, w - xOf(range.end), h);
      // handles: the hovered / dragged one grows and glows
      const active = drag ? drag.kind : hover;
      if (active === 'move') { g.strokeStyle = C.accent; g.lineWidth = 1.5; g.strokeRect(xOf(range.start), top - 4, xOf(range.end) - xOf(range.start), bottom - top + LANE + 8); }
      for (const [t, kind] of [[range.start, 'start'], [range.end, 'end']]) {
        const x = xOf(t), on = active === kind, hw = on ? 8 : 5, hh = on ? 15 : 9, cy = (top + bottom) / 2;
        if (on) { g.shadowColor = C.accent; g.shadowBlur = 14; }
        g.fillStyle = C.accent; g.fillRect(x - (on ? 2 : 1.5), top - 6, on ? 4 : 3, bottom - top + LANE + 12);
        g.fillStyle = C.paper; g.beginPath(); g.roundRect(x - hw, cy - hh, 2 * hw, 2 * hh, 3); g.fill();
        g.shadowBlur = 0;
        g.fillStyle = C.accent;
        for (const dx of on ? [-3, 0, 3] : [0]) g.fillRect(x + dx - 0.5, cy - (on ? 7 : 5), 1, on ? 14 : 10);
      }
      // frame under the pointer: hairline + thumbnail + its scores
      if (previewT != null && thumbs && thumbs.length) {
        const th = thumbAt(previewT), fr = frameAt(previewT);
        const x = xOf(previewT);
        g.fillStyle = C.paper; g.fillRect(x - 0.5, top - 6, 1, bottom - top + LANE + 12);
        const pw = Math.min(260, Math.max(120, Math.round(w * 0.22))), phh = Math.round(pw * th.canvas.height / th.canvas.width);
        const px = Math.max(PAD, Math.min(w - PAD - pw, x - pw / 2)), py = top + 2;
        g.fillStyle = 'rgba(7, 9, 9, .92)'; g.beginPath(); g.roundRect(px - 3, py - 3, pw + 6, phh + 24, 6); g.fill();
        g.drawImage(th.canvas, px, py, pw, phh);
        g.strokeStyle = fr.blur ? C.red : fr.picked ? C.accent : C.muted; g.lineWidth = 1.5; g.strokeRect(px + 0.5, py + 0.5, pw - 1, phh - 1);
        g.font = `11px ${css('--mono')}`; g.textBaseline = 'middle'; g.textAlign = 'left'; g.fillStyle = C.paper;
        g.fillText(`${fmt(fr.t)}  sharpness ${Math.round(fr.focus)}`, px + 2, py + phh + 11);
        g.textAlign = 'right'; g.fillStyle = fr.blur ? C.red : fr.picked ? C.accent : C.dim;
        g.fillText(fr.blur ? 'blur dip' : fr.picked ? 'picked' : 'not picked', px + pw - 2, py + phh + 11);
      }
      // time labels
      g.fillStyle = C.dim; g.font = `10px ${css('--mono')}`; g.textBaseline = 'top';
      g.textAlign = 'left'; g.fillText('0 s', PAD, 4);
      g.textAlign = 'right'; g.fillText(fmt(duration), w - PAD, 4);
      g.textAlign = 'center'; g.fillStyle = C.paper;
      g.fillText(`${fmt(range.start)} – ${fmt(range.end)}`, (xOf(range.start) + xOf(range.end)) / 2, 4);
    }

    function drawStrip() {
      const [g, w, h] = size(strip);
      g.clearRect(0, 0, w, h);
      if (!thumbs || !thumbs.length || !plan.picks.length) return;
      const slot = Math.max(80, Math.round((h - 8) * 1.5)), n = Math.max(1, Math.min(plan.picks.length, Math.floor(w / slot)));
      const sw = w / n;
      for (let k = 0; k < n; k++) {
        const pi = plan.picks[Math.round((k / Math.max(1, n - 1)) * (plan.picks.length - 1))];
        const t = frames[pi].t;
        let th = thumbs[0];   // nearest thumbnail in time
        for (const c of thumbs) if (Math.abs(c.t - t) < Math.abs(th.t - t)) th = c;
        const cv = th.canvas, ar = cv.width / cv.height;
        const dh = h - 8, dw = dh * ar;   // fit height, centre-crop the width into the slot
        const x = k * sw;
        g.save(); g.beginPath(); g.rect(x + 1, 4, sw - 2, dh); g.clip();
        g.drawImage(cv, x + (sw - dw) / 2, 4, dw, dh);
        g.restore();
      }
      const label = n < plan.picks.length ? `${n} of ${plan.picks.length} picks, spread over the range` : `all ${n} picks`;
      g.font = `10px ${css('--mono')}`; g.textBaseline = 'middle'; g.textAlign = 'left';
      g.fillStyle = 'rgba(7, 9, 9, .78)'; g.fillRect(0, h - 15, g.measureText(label).width + 10, 15);
      g.fillStyle = C.muted; g.fillText(label, 5, h - 7.5);
    }

    function readout() {
      const n = plan.picks.length, span = range.end - range.start;
      $('vid-n').textContent = `${n} frame${n === 1 ? '' : 's'}`;
      $('vid-span').textContent = `${fmt(range.start)} – ${fmt(range.end)} · ${(n / Math.max(0.1, span)).toFixed(1)} per second`;
      $('vid-reset').hidden = whole();
      $('vid-hint').textContent = ctx.suggested && whole()
        ? `Long video: the whole take spreads the ${cap} frames this device can hold too thin. Slide the range to the part that matters.`
        : ctx.suggested
          ? `Long video: this is the sharpest span the ${cap} frames this device can hold cover at full density. Slide it, or drag its ends.`
          : n >= cap
            ? `The device holds ${cap} frames, so this range is sampled thinner than the footage allows. A shorter range packs them tighter.`
            : 'Drag the ends of the range, or slide it, to reconstruct only part of the video.';
      $('vid-use').textContent = `Use these ${n} frames`;
    }
    function redraw() { drawTimeline(); drawStrip(); readout(); }
    function setRange(r) {
      const minLen = Math.min(1.0, duration);
      let a = Math.max(full.start, Math.min(r.start, r.end - minLen));
      let b = Math.min(full.end, Math.max(r.end, a + minLen));
      if (b - a < minLen) { a = Math.max(full.start, b - minLen); }
      range = { start: a, end: b };
      plan = ctx.plan(whole() ? null : range, gap());
      S.videoPlan = plan;
      redraw();
    }

    // dragging: a handle (near it), or the whole range (inside it)
    let drag = null;
    const hit = (x) => {
      const xs = xOf(range.start), xe = xOf(range.end);
      if (Math.abs(x - xs) <= 14 && Math.abs(x - xs) <= Math.abs(x - xe)) return 'start';
      if (Math.abs(x - xe) <= 14) return 'end';
      if (x > xs && x < xe) return 'move';
      return null;
    };
    const cursorFor = (kind, down) => kind === 'move' ? (down ? 'grabbing' : 'grab') : kind ? 'ew-resize' : 'crosshair';
    tl.addEventListener('pointerdown', (e) => {
      const x = e.offsetX;
      const h = hit(x);
      if (h === 'move') drag = { kind: 'move', t0: tOf(x), s0: range.start, e0: range.end };
      else drag = { kind: h || (x < xOf(range.start) ? 'start' : 'end') };
      tl.style.cursor = cursorFor(drag.kind, true);
      try { tl.setPointerCapture(e.pointerId); } catch {}
      onMove(e);
    });
    tl.addEventListener('pointerleave', () => { if (!drag && (hover || previewT != null)) { hover = null; previewT = null; tl.style.cursor = 'crosshair'; drawTimeline(); } });
    let raf = 0;
    const onMove = (e) => {
      if (!drag) return;
      const t = tOf(e.offsetX);
      let r;
      if (drag.kind === 'start') r = { start: t, end: range.end };
      else if (drag.kind === 'end') r = { start: range.start, end: t };
      else {
        const len = drag.e0 - drag.s0;
        let s = drag.s0 + (t - drag.t0);
        s = Math.max(full.start, Math.min(full.end - len, s));
        r = { start: s, end: s + len };
      }
      previewT = drag.kind === 'start' ? Math.max(full.start, Math.min(r.start, r.end)) : drag.kind === 'end' ? Math.min(full.end, r.end) : t;
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; setRange(r); });
    };
    tl.addEventListener('pointermove', (e) => {
      if (drag) { onMove(e); return; }
      const h = hit(e.offsetX);
      hover = h; tl.style.cursor = cursorFor(h, false);
      previewT = tOf(e.offsetX);
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; drawTimeline(); });
    });
    const onUp = (e) => { drag = null; hover = hit(e.offsetX); tl.style.cursor = cursorFor(hover, false); if (e.pointerType === 'touch') { hover = null; previewT = null; } drawTimeline(); };
    tl.addEventListener('pointerup', onUp);
    tl.addEventListener('pointercancel', onUp);
    $('vid-reset').onclick = () => setRange({ ...full });
    $('vid-gap-sel').onchange = () => {
      const v = $('vid-gap-sel').value;
      gapSec = v === '' ? null : +v;
      plan = ctx.plan(whole() ? null : range, gap());
      S.videoPlan = plan;
      redraw();
    };
    $('vid-cancel').onclick = $('vid-x').onclick = () => { cleanup(); resolve(null); };
    const onKey = (e) => { if (e.key === 'Escape') { cleanup(); resolve(null); } };
    window.addEventListener('keydown', onKey);
    $('vid-use').onclick = () => { cleanup(); resolve({ picks: plan.picks }); };
    const onResize = () => redraw();
    window.addEventListener('resize', onResize);
    function cleanup() { window.removeEventListener('resize', onResize); window.removeEventListener('keydown', onKey); S.videoReview = null; }
    setRange(range);
  });
}

/** reset everything and show a set's start card (autostart commits a switch) */
async function open(preset, autostart = false) {
  document.getElementById('failcard')?.remove();
  if (preset.list && !preset.names) {
    S._intakeN = (S._intakeN || 0) + 1;
    try { await loadPresetList(preset); }
    catch { flash('Could not load that set\'s file list.', 5000); return; }
    finally { S._intakeN--; }
  }
  S.gen++;
  if (S.session) { S.session.pause(); S.session.dispose(); }
  S.session = null;
  S.preset = preset;
  S.state = 'ready';
  S.trainVisuals = false;
  S._carryTrainElapsed = false;
  S._savedGpuFacts = null; S._savedDetailSettings = null;
  S._localRun = null; S._securing = null;
  S.runId = null; S._resumeRunId = null;
  syncTrainingVisualsControl();
  S.picking = false; S.pending = null;
  S.scene = null;
  S.sel = 0; S.atFrame = -1; S.fade = 0; S.fadeTo = 0;
  S.iter = 0; S.splats = 0; S.psnrTrain = null; S.psnrHold = null;
  S.prep = null; S.feats = new Map(); S.lastPairEv = null; S.shownPair = null; S.regCams = [];
  S.regPts = null; S.regPtsCount = 0;
  // the avatar manifest comes from the SET being opened: a wall preset or a photo drop after an
  // avatar run used to inherit S.avatar and run the avatar pipeline (and show its settings)
  // (no await here: open() used to be synchronous to this point, and yielding let the start card
  // win over the detail card — the options were computed by whichever path built the set)
  S.walker = null; S.walking = false;
  S.avatar = preset.avatar || null;
  if (!S.avatar) S.avatarOpts = null;
  // the box on the start card: ticked for a capture that already went the
  // avatar route (its matte is stored with the frames), clear otherwise
  S.avatarWanted = !!S.avatar;
  S.growNote = null;
  S.tour = null;
  S.solveStats = { pairsChecked: 0, pairsUsable: 0, solveSec: 0 };
  S.chartEvents = [];
  S.maxIters = PERF.on ? PERF.iters : INITIAL_ITERS;
  S.perfMetrics = [];
  S.holdHist = [];

  S._lastReady = null;
  S.growthStopped = false;
  S.plyBlob = null; S.sogBlob = null;
  S._fromSpace = null;
  S._thumbsUrl = null; S._thumbPack = null;
  S._recovering = false;
  S._dragPaused = false;
  S._errRender = null;
  S._errRenderBusy = false;
  S._viewKey = '';
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  clearTimeout(S._presentWd && S._presentWd.timer);
  S._presentWd = null;
  setStartStyle(true);
  $('btn-go').disabled = !!S.noGpu;
  $('btn-settings').disabled = false;
  $('card-x').hidden = true;
  closeOverlays();
  $('start').hidden = true;
  $('detail').hidden = true;
  $('controls').hidden = true;
  $('btn-new').hidden = false;   // the setup card is not home
  $('strip').innerHTML = '';
  dock('');
  vp.resize();
  vp.lock = null; vp.pose = null; vp.enabled = true; vp.scene = null;
  vp.fpv = false; S.fpvSet = false;

  // the photographs: URLs only — decoding happens when the run starts
  if (preset.files) {
    S.photos = preset.files.map((f, i) => ({ url: preset.urls[i], name: f.name }));
  } else {
    S.photos = presetPhotoList(preset, preset.useCount || preset.count);
  }
  // deferred: the strip sits behind the fullscreen card — its thumbnails
  // load once training starts (or a tile is touched), not now
  buildStrip(true);
  paintCard(preset);
  if (autostart) await startPrep();
  else $('start').hidden = false;
}

// ── prep: the solve, live ───────────────────────────────────────────────────
const BEATS_BASE = [
  { id: 'decode',   label: 'Reading photographs' },
  { id: 'features', label: 'Finding landmarks' },
  { id: 'matching', label: 'Matching photos' },
  { id: 'cameras',  label: 'Solving positions' },
  { id: 'seed',     label: 'Seeding splats' },
];
// an avatar run cuts the person out of every frame BEFORE any of this: the
// matte is the run's first beat, not something that happened minutes earlier
// when the video was read (the decision moved to the start card, 2026-09-18)
const MATTE_BEAT = { id: 'matte', label: 'Cutting the person out' };
const BEATS_KNOWN = [
  { id: 'decode', label: 'Reading photographs' },
  { id: 'known', label: 'Using tracked cameras' },
  { id: 'seed', label: 'Seeding splats' },
];
let BEATS = BEATS_BASE;
const beatsFor = (avatar) => S.preset?.capture?.transforms ? BEATS_KNOWN : (avatar ? [MATTE_BEAT, ...BEATS_BASE] : BEATS_BASE);
const beatIndex = (stage) => {
  if (BEATS === BEATS_KNOWN) return stage === 'seed' ? 2 : (stage === 'known' || stage === 'solved' ? 1 : 0);
  const off = BEATS[0].id === 'matte' ? 1 : 0;
  return Math.max(0, off + ({ matte: -1, decode: 0, features: 1, matching: 2, focal: 3, register: 3, ba: 3, solved: 3, seed: 4 }[stage] ?? 0));
};

/** Own sets: how many of the picks an avatar run trains on. An avatar orbit
 *  needs the person AT RESOLUTION, not every frame — the decode budget (700 MB
 *  of targets) shrank 208 frames of 4K to 706 px, a third of a 1080p clip's
 *  per-frame detail (Lisa, 2026-09-15). Thinning is OFF by default
 *  (AVATAR_MAX_FRAMES 0: it broke Lisa's chain); ?avframes=N thins evenly for
 *  experiments. The full pick list is stashed, so unticking the box and
 *  training the scene is never short of frames. */
function avatarFrames(want) {
  const p = S.preset;
  if (!p || !Array.isArray(p.files)) return;
  if (!p._allFiles) { p._allFiles = p.files; p._allUrls = p.urls; }
  const all = p._allFiles;
  const maxFrames = +(new URLSearchParams(location.search).get('avframes') || 0) || AVATAR_MAX_FRAMES;
  let idx = all.map((_, i) => i);
  if (want && maxFrames > 1 && all.length > maxFrames) {
    idx = [...new Set(Array.from({ length: maxFrames }, (_, i) => Math.round((i * (all.length - 1)) / (maxFrames - 1))))];
    console.log(`[avatar] ${all.length} picked frames thinned to ${idx.length} for the avatar run`);
  }
  if (idx.length === p.files.length) return;
  p.files = idx.map((i) => all[i]);
  if (p._allUrls) p.urls = idx.map((i) => p._allUrls[i]);
  S.photos = p.files.map((f, i) => ({ url: p.urls ? p.urls[i] : null, name: f.name }));
  buildStrip(true);
}

/** The avatar route, on Start training: cut the person out of every frame and
 *  ask whether that is the subject. This runs BEFORE the session exists, so
 *  the mattes ride on the frame entries into it — and nothing expensive has
 *  happened yet when the answer is no (the run simply continues as a scene).
 *  Until 2026-09-18 it ran when the video was read, minutes before anyone had
 *  said they wanted to train it. Returns false when there is no avatar. */
async function avatarPrep(gen) {
  const av = await import('../avatar/index.js');
  const files = S.preset.files || [];
  // matte.js reads f.source; a restored capture's entries ARE the blobs
  const entries = files.map((f) => (f.source ? f : { name: f.name, source: f }));
  const card = document.createElement('div');
  card.className = 'upcard av-sheet';
  card.id = 'avprep';
  $('stage').appendChild(card);
  let manifest = null;
  try {
    manifest = await av.prepareCapture(entries, S.preset.source || { picks: files.length }, {
      card, flash,
      log: (m) => console.log('[avatar]', m),
      progress: (d, t) => { S.prep = { stage: 'matte', done: d, total: t }; },
    });
  } catch (e) { console.log('[avatar] ' + (e.message || e)); }
  // the cut-outs stay up for a few seconds while the solve gets going behind
  // them — a picture to look at, not a question to answer (2026-09-18)
  if (manifest) setTimeout(() => card.remove(), CUTOUTS_MS); else card.remove();
  if (S.gen !== gen) { card.remove(); return false; }
  // the mattes were written on the entries — hand them to the real frame list
  for (let i = 0; i < files.length; i++) {
    if (entries[i] === files[i]) continue;
    if (entries[i].mask) files[i].mask = entries[i].mask; else delete files[i].mask;
  }
  if (!manifest) { S.avatar = null; S.avatarWanted = false; for (const f of files) delete f.mask; return false; }
  S.avatar = manifest;
  // the mattes go into the stored capture as well: reopening it skips this stage
  if (S._capRec && Array.isArray(S._capRec.files)) {
    const masks = new Map(files.map((f) => [f.name, f.mask]));
    let n = 0;
    for (const e of S._capRec.files) { const m = masks.get(e.name); if (m) { e.mask = m; n++; } }
    S._capRec.avatar = manifest;
    if (n) persistCapture(S._capRec);
  }
  return true;
}

/** One Start request owns the button until it enters prep or explains why it
 *  cannot. Await every preflight (including AR intake completion and Android
 *  run ownership) under visible button state and one top-level error boundary:
 *  no guard or rejected promise may look like an inert button. */
async function startPrep() {
  if (S._startBusy) return false;
  S._startBusy = true;
  const go = $('btn-go');
  const oldLabel = go.textContent;
  go.disabled = true;
  go.textContent = (S._intakeN || 0) > 0 ? 'Preparing photos…' : 'Starting…';
  try {
    return await startPrepInner();
  } catch (e) {
    console.error('[start]', e);
    flash(`Could not start training: ${e && e.message ? e.message : e}`, 9000);
    return false;
  } finally {
    S._startBusy = false;
    if (S.state === 'ready' || !$('detail').hidden || !$('start').hidden) {
      go.textContent = oldLabel;
      go.disabled = !!S.noGpu;
    }
  }
}

async function startPrepInner() {
  if ((S._intakeN || 0) > 0) {
    const until = performance.now() + 15000;
    while ((S._intakeN || 0) > 0 && performance.now() < until) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    if ((S._intakeN || 0) > 0) {
      flash('The photos are still being prepared. Please try again.', 7000);
      return false;
    }
  }
  document.getElementById('failcard')?.remove();
  if (S.commitSettings && !S.commitSettings()) return;
  // migrate-on-swipe: one GPU — a live headless run owns it until adopted,
  // recovered or discarded (the monitor is the way back to it)
  if (await headlessMarkerLive()) {
    // Do not leave the user on a Start button which can never start. Put the
    // already-running job in front of them; if its files race this read, the
    // explanation remains as a fallback.
    if (!(await headlessBootCheck()))
      flash('A training run is still going in the background — reopen it from the monitor instead of starting over it.', 8000);
    return false;
  }
  if (!S.preset || S.photos.length < 3) { flash('Use at least 3 overlapping views; 20–200 are recommended.', 7000); return; }
  if (S.preset.capture?.transforms) {
    try { validateDataset(S.preset.capture); }
    catch (e) { flash(e.message, 8000); return; }
    S.avatarWanted = false;
  }
  // consume the detail-card history entry — Back during a run must not
  // resurrect a card that no longer applies
  if (history.state && history.state.sj) history.replaceState(null, '');
  // and the address bar stops naming the scene we came FROM: a run of your
  // own is not that space, and a refresh here used to reopen it. The tuning
  // flags (?iters, ?ipf, ?api …) describe this run and stay.
  {
    const u = new URL(location.href);
    const scene = ['space', 'model', 'recon', 'frame', 'cmp'];
    if (scene.some((k) => u.searchParams.has(k))) {
      for (const k of scene) u.searchParams.delete(k);
      history.replaceState(history.state, '', u);
    }
  }
  const gen = S.gen;
  $('start').hidden = true;
  $('detail').hidden = true;
  $('btn-new').hidden = false;
  // settle the pick-time capture save: retry once now that the picker rush
  // is over, and SAY SO if the device will not keep the photos
  (async () => {
    if (!S._capSave || (await S._capSave) !== 'failed' || !S._capRec) return;
    persistCapture(S._capRec);
    if ((await S._capSave) === 'failed') {
      flash('The photos could not be kept on this device — export the result before closing this tab.', 9000);
    }
  })().catch(() => {});
  // a fresh solve ALWAYS starts from zero. Paths that arrive here without
  // open() — Train on a shared scene, Start after a failed solve — would
  // otherwise leak the viewed model's numbers into the dock's first paint
  // (seen live: the % born at 50 or 100, jumping down to 1 on the first
  // metrics tick; stale dB and splat counts flash the same way).
  S.iter = 0; S.splats = 0; S.psnrTrain = null; S.psnrHold = null;
  S.itersPerSec = 0; S.minutes = 0; S.etaAt = null;
  S.holdHist = []; S.chartEvents = [];  S.growthStopped = false;
  S.feats = new Map(); S.lastPairEv = null; S.shownPair = null; S.regCams = [];
  S.regPts = null; S.regPtsCount = 0; S.growNote = null;
  S.solveStats = { pairsChecked: 0, pairsUsable: 0, solveSec: 0 };
  S.maxIters = ITERS_OVERRIDE >= 1000 ? ITERS_OVERRIDE : (PERF.on ? PERF.iters : (S.settings.iters || INITIAL_ITERS));
  S.state = 'prep';
  syncTrainingVisualsControl();
  // ── the avatar route, as decided on the start card ───────────────────────
  // Unticked, this is a plain scene run whatever the set once was: a restored
  // capture's stored mattes leave the frames the session sees.
  S.avatarOpts = null;
  if (SHELL === 'browser') S.avatarWanted = false;   // the browser profile trains scenes only — no avatar models ship there
  if (!S.avatarWanted) { S.avatar = null; for (const f of S.preset.files || []) delete f.mask; }
  avatarFrames(S.avatarWanted);
  const wantMatte = S.avatarWanted && !S.avatar && !!(S.preset.files && S.preset.files.length);
  if (S.avatarWanted && !S.avatar && !wantMatte) { console.log('[avatar] this set has no frames in hand — continuing as a scene'); S.avatarWanted = false; }
  BEATS = beatsFor(S.avatarWanted);
  S.prep = { stage: wantMatte ? 'matte' : 'decode', done: 0, total: S.photos.length };
  dock('prep');
  buildStrip();   // the card is gone — the strip is visible now, load live
  if (wantMatte) {
    const isAvatar = await avatarPrep(gen);
    if (S.gen !== gen) return;
    if (!isAvatar) BEATS = beatsFor(false);
    S.prep = { stage: 'decode', done: 0, total: S.photos.length };
  }
  if (S.avatar) {
    const av = await import('../avatar/index.js');
    S.avatarOpts = av.trainingOptions(S.avatar, { iters: S.settings.iters || AVATAR_ITERS, shHorizontal: !!S.settings.shup, av: S.settings.av });
    if (S.gen !== gen) return;
  }

  try {
    // view buffers sized for the screen at 1x CSS pixels (the stage renders
    // at 1x — splats don't reward supersampling; clamps are pixel-count based)
    const mvW = Math.ceil(screen.width || 1280);
    const mvH = Math.ceil(screen.height || 800);
    S.viewPixBudget = Math.min(
      mvW * mvH,
      16000 * 256, // per-raster tile-grid cap (16k tiles of 16x16)
    );
    // settings -> session options: res caps the input scale, the working
    // buffer scales the supervision grid on top of whatever that yields
    const st = S.settings;
    const frames = (st.res || st.feat || st.buf !== 1 || EVAL.on) ? {
      // benchmark mode pins NATIVE resolution: the adaptive memory budget
      // otherwise downscales big sets silently (truck-251 lands at 645px)
      // and PSNR at reduced resolution reads ~1 dB better than the papers'
      trainMaxDim: st.res === -1 ? (S.setMaxDim || undefined) : (st.res || (EVAL.on ? 1600 : undefined)),   // -1: the pictures' own size (as far as the memory budget allows)
      trainScale: st.buf !== 1 ? st.buf : undefined,
      ...(st.feat ? { featMaxDim: st.feat } : {}),
    } : undefined;
    // every photo trains by default — held-out scoring is the ?eval
    // benchmark protocol (every Nth photo scored, never learned from)
    const trainerOpts = {};
    // always pass the selected degree — the old `!== 2` guard dated from when
    // the trainer default WAS 2; after the default flipped to 3 it silently
    // trained degree 3 whenever the UI said 2
    if (st.sh != null) trainerOpts.shDeg = st.sh;
    if (st.splats) trainerOpts.maxSplats = st.splats;
    // big budgets: growth must be able to reach the cap even when the sparse
    // cloud can't seed budget/4 (measured: bar panos init ~500k, cap 4M)
    if (st.splats >= 1000000) trainerOpts.capMult = 16;
    // LOD training (opt-in, >= 1M budgets): the model pauses at halving
    // detail levels (250k, 500k, ...) for a polish-and-snapshot before
    // growing on — so it must START below the lowest level
    const lodOn = !!st.lod && st.splats >= 1000000;
    if (lodOn) {
      S.lodPlan = buildLODPlan(st.splats, st.lodLevels || 0, S.maxIters);
      trainerOpts.capMult = Math.max(16, Math.ceil(st.splats / 250000) + 1);
    } else {
      S.lodPlan = null;
    }
    // the ?iters override must land BEFORE anything reads the budget
    if (ITERS_OVERRIDE >= 1000) S.maxIters = ITERS_OVERRIDE;
    // MCMC set: ON by default — measured garden 20k/40k +1.12/+0.61 dB and
    // truck 30k +0.3; the one loss (truck 20k, -0.36) is scene-specific
    // (base growth happened to land near-optimal there). Off in the gear
    // restores the classic schedule.
    const mcmcActive = MCMC_ON || st.mcmc !== '0';
    if (mcmcActive) {
      Object.assign(trainerOpts, {
        growRate: 0.05, mcmcNoise: true, scaleReg: 0.01, moveCap: 0.25,
        ...((st.sh ?? 3) > 0 ? { shLr: 3e-4 } : {}),
      });
      flash('MCMC experimental set active', 5000);
    }
    if (NEEDLE_ON) {
      if (Number.isFinite(DILATE_OVR) && DILATE_OVR > 0 && DILATE_OVR <= 1) trainerOpts.dilate = DILATE_OVR;
      if (Number.isFinite(ANISO_OVR) && ANISO_OVR >= 0) trainerOpts.anisoReg = ANISO_OVR;
      if (Number.isFinite(MINSCALE_OVR) && MINSCALE_OVR > 0 && MINSCALE_OVR < 1e-3) trainerOpts.minScale = MINSCALE_OVR;
      flash(`Needle set active — dilate ${trainerOpts.dilate ?? 0.3}, anisoReg ${trainerOpts.anisoReg ?? 'default'}, minScale ${trainerOpts.minScale ?? '1e-4'} (?dilate=0 clears)`, 6000);
    }
    if (PLACEMENT_ON) {
      // applied after the MCMC set so growRate 0.1 wins over its 0.05
      Object.assign(trainerOpts, { mipComp: false, anisoReg: 0, minScale: 1e-5, refineV2: true, growRate: 0.1 });
      flash('Placement set active — mipComp off, anisoReg 0, minScale 1e-5, refineV2, growRate 0.1 (?placement=0 clears)', 6000);
    }
    // Soft prune (opt-in): periodic importance culls during growth — the
    // model holds more detail under the same splat budget
    if (st.softprune) trainerOpts.softPrune = { ratio: parseFloat(st.softprune), every: 3000, until: 0.85 };
    // Auto splat budget: sized from the CYCLE budget, not just the solve —
    // the measured capacity law (~15 splats/cycle classic, ~35 under MCMC),
    // clamped to the device class. Explicit user choices always win.
    const phoneClass = isPhoneClass();
    // 2× working buffer on a phone: the crash the user would otherwise meet
    // mid-run is an OOM device loss — say it upfront, while Start is still
    // the decision point (desktops with headroom are spared the nag)
    if (st.buf === 2 && phoneClass) {
      flash('Working buffer 2× roughly quadruples GPU working memory — on this phone the run may abort for lack of memory. 1× trains the same photos safely.', 8000);
    }
    if (!st.splats) {
      // mobile ceiling is 1M: past it the SH binding (deg 3 clamps at 745k
      // in the engine) and the tile-sort entries (~16.7/splat at 1M) bind
      // instead — overflow past that point warns, it never fails silent
      trainerOpts.maxSplats = Math.max(150000, Math.min(phoneClass ? 1000000 : 2000000,
        Math.round(S.maxIters * (mcmcActive ? 35 : 15))));
      trainerOpts.capMult = 8; // growth must reach the computed cap from a lean seed
    }
    // engine auto-select (measured 2026-08-30): v2 wins short/medium
    // budgets (+0.3..0.6 up to ~75k cycles), v1 wins marathons and stays
    // the phone engine (v2's SSIM passes cost ~1.8x time + GPU memory).
    // ?engine=v1|v2 overrides for A/B.
    const engineQ = new URLSearchParams(location.search).get('engine');
    // AUTO-SELECT DORMANT (2026-08-31): v2 wins live training but the SOG
    // storage tax (-0.95 vs v1's -0.36) eats the gain on stored/shared
    // scenes. Re-enable when the encoder handles unbounded DC.
    const engine = engineQ === 'v2' ? 'v2' : undefined;
    if (engine) { trainerOpts.engine = engine; }
    // avatar mode: one subject, not a room — its own sizing (app/avatar)
    if (S.avatarOpts) Object.assign(trainerOpts, S.avatarOpts.trainer);
    const knownCapture = S.preset.capture?.transforms ? S.preset.capture : null;
    if (knownCapture) Object.assign(trainerOpts, { camOpt: false, aspectOpt: false });
    const session = createSession({
      maxIters: S.maxIters, evalHoldEvery: 2500,
      holdout: -1,
      // the GPU-usage slider as a duty cycle: full batches with idle
      // between them (read live, so dragging mid-run needs no restart)
      gpuUsage: (st.gpu ?? 100) / 100,
      ...(engine === 'v2' ? { refineEvery: 200 } : (mcmcActive ? { refineEvery: 500 } : {})),
      evalSplit: EVAL.on ? EVAL.split : 0,
      initTarget: lodOn ? S.lodPlan.levels[0] : (st.splats ? Math.round(st.splats / 4) : (phoneClass ? 60000 : undefined)),
      maxViewW: mvW, maxViewH: mvH,
      ...(S.avatarOpts ? S.avatarOpts.session : {}),
      // phones: iOS jetsams the tab long before the GPU is the limit —
      // decode-to-target (in the library), 720px features, a smaller SIFT
      // worker pool, and dropping gray/rgb once each stage has consumed them
      ...(phoneClass ? {
        lowMem: true,
        // uiYield breaks BA's multi-second synchronous bursts during the
        // solve (10s+ frozen UI on phones). Training is untouched: measured
        // fine on-device, and fenceRing/gpuChunkMs (library opts) would tax
        // throughput for nothing.
        sfm: { ...solveTierOpts(st.solve || 'quick'), workers: 3, uiYield: true, ...VIDEO_GATES(), ...(S.avatarOpts && S.avatarOpts.session && S.avatarOpts.session.sfm || {}), ...SFM_BISECT },
      } : {
        // desktop solver default since 2026-09-04: 8000 SIFT features from the
        // upsampled first octave (COLMAP's default). Feature localisation is
        // the pose-precision ceiling: truck +0.38 dB, garden +0.26, camping ±0
        // at 30k; the hour-long truck 26.40 → 26.55. The solve takes ~3x
        // longer (truck 4 → 12 min). Phones keep the lean settings: the
        // upsampled octave is 4x the SIFT work on a thermally limited chip.
        // + bundle-adjustment pixel-aspect (fy/fx): the two go TOGETHER — with
        // coarse features the aspect BA drifted (camping −0.6), with these it
        // is the best solve on all three sets (camping 25.5 → 27.2). Truck's
        // release images are 0.6 % non-square, camping's video frames 2 %.
        // Since 2026-09-09 that is the 'precise' solve tier (High / Showcase);
        // Standard takes the 8000 budget at the base octave, Draft the lean
        // 3900 / octave-0 solve — see SOLVE_TIERS in the library.
        // the avatar's solve options win over the settings tier (they were overridden here until 2026-09-16: the precise tier and ?sfmall never reached the solver)
        sfm: { ...solveTierOpts(st.solve || 'standard'), ...VIDEO_GATES(), ...(S.avatarOpts && S.avatarOpts.session && S.avatarOpts.session.sfm || {}), ...SFM_BISECT },
      }),
      // phones solve at the desktop feature resolution again: 720 was part
      // of the OOM firefight, but the real culprit was the UI bitmap cache —
      // and feature res is the measured pose-precision ceiling
      // phone default 960 — an explicit Solve-resolution choice still wins
      // avatar runs on a desktop take a larger target budget (700 MB -> 1.1 GB, with the
      // crop windows capped at 240 the trainer stays under its 2 GB binding): a 4K
      // orbit's 208 frames decoded at 706 px under the default budget (2026-09-15)
      frames: knownCapture ? { ...(frames || {}), disablePano: true } : (phoneClass ? { featMaxDim: 960, ...(frames || {}) } : (S.avatarOpts && S.avatarOpts.budget ? { targetBudgetBytes: 1.1e9, ...(frames || {}) } : frames)),
      trainer: Object.keys(trainerOpts).length ? trainerOpts : undefined,
      trainWorker: trainWorkerOpt(),
    });
    S.session = session;
    session.on('stage', (e) => { if (S.gen === gen) onStage(e); });
    session.on('metrics', (e) => { if (S.gen === gen) onMetrics(e); });
    session.on('event', (e) => { if (S.gen === gen) onTrainEvent(e); });
    if (S.avatar || new URLSearchParams(location.search).has('sessionlog')) session.on('log', (m) => console.log('[session]', m));

    // 1) decode
    let files;
    if (S.preset.files) {
      files = S.preset.files;
    } else {
      files = [];
      for (let i = 0; i < S.photos.length; i++) {
        // uploads come through the edge transform at 2048 px: training caps the
        // long side below that anyway, and an 8064 px original is 10 MB a photo
        const r = await fetch(sizedUrl(S.photos[i].url, 2048));
        if (!r.ok) throw new Error(`could not fetch ${S.photos[i].name}`);
        // url rides along: the session export stores where each image lives
        files.push({ source: await r.blob(), name: S.photos[i].name, url: S.photos[i].url });
        S.prep = { stage: 'decode', done: i + 1, total: S.photos.length };
        if (S.gen !== gen) return;
      }
    }
    S.loadedFiles = files;   // originals, for "Download photos" in export
    await session.load(files);
    if (S.gen !== gen) return;

    // 2) solve — the beats are its real events
    const t0 = performance.now();
    if (knownCapture) {
      onStage({ stage: 'known', done: 0, total: 1 });
      const points = measuredSeeds(knownCapture);
      session.useKnownCameras(parseCameras(knownCapture.transforms), { ...(points.length ? { points } : {}), seedCount: phoneClass ? 8000 : 20000 });
      if (session.recon.cams.length !== session.frames.length) throw new Error('Some images lost their known cameras during decoding. No estimated poses will be substituted.');
      S.solveStats.solveSec = 0;
      onStage({ stage: 'solved', done: 1, total: 1, detail: { cams: session.recon.cams.length, points: session.recon.points.length, known: true } });
      flash(`${session.recon.cams.length} known camera poses loaded — camera solve skipped. ${points.length ? 'Measured seeds.' : 'Random-volume initialization.'}`, 7000);
    } else {
      await session.solve({ debug: (d) => { S.internals = d; } });
    }
    if (S.gen !== gen) return;
    if (!knownCapture) S.solveStats.solveSec = (performance.now() - t0) / 1000;

    // No threshold guessing: below 3 placed cameras there is no multi-view
    // problem left to solve — stop with advice. Anything above that trains,
    // and the truth is shown instead: unplaced images carry a red tag in the
    // strip, and a notice says how many made it.
    const placed = session.recon.cams.length;
    const isOwn = S.preset.id && S.preset.id.startsWith('__');
    if (placed < 3) {
      solveFailed(isOwn
        ? `Only ${placed} of ${S.photos.length} photos could be placed — ` +
          'the set doesn\'t connect well enough to reconstruct.'
        : `Only ${placed} of ${S.photos.length} images could be placed — ` +
          'that is unusual for this test set. Reloading the page and retrying usually clears it.');
      trainDeadEnd();
      return;
    }
    if (placed < S.photos.length) {
      flash(`${placed} of ${S.photos.length} images placed — the ones tagged in the strip never connected.`, 9000);
    }

    // avatar mode: native-resolution windows of the person join the cameras
    // before the seed (app/avatar/crops.js) — the person trains at full pixel
    // density inside the room's run
    if (S.avatar) {
      const av = await import('../avatar/index.js');
      const alog = (m) => console.log('[avatar]', m);
      // the joints and the face need only the solve: run them now, so the head
      // windows get the head-stabilised poses (the head moves against the room —
      // Filip's orbit ghosted by ~2 cm); the joints card is reviewed after training
      let faceCams = null;
      // the pre-training landmarks and face triangulation exist for the crop windows (head-stabilised
      // poses); with Person crops off they are skipped — the joints stage still runs after training
      const cropsMode = new URLSearchParams(location.search).get('crops') === '0' ? 'off' : ((S.settings.av && S.settings.av.crops) || 'face');
      const cropsOn = cropsMode !== 'off';
      if (!cropsOn) alog('person crops: off — landmarks and face wait until after training');
      try { if (!cropsOn) throw Object.assign(new Error('skipped'), { skipped: true });
        const lm = await import('../avatar/stages/landmarks.js');
        const res = await lm.run({ session, frames: files }, S.avatar, { log: alog, progress: (d, t) => { S.prep = { stage: 'landmarks', done: d, total: t }; } });
        av.setStage(S.avatar, 'landmarks', { status: 'done', ...res, reviewPending: true });
        S._lmRes = res; faceCams = res.cropCams || null; S._facePoints = res.face && res.face.points || null; // the nose tip shifts under 1 cm over the orbit on BOTH clips, so head motion is not
        // what separates them; what does is how well the room's poses agree on the person:
        // the pose landmarks' multi-view residual (nose) is 1.8 px on Tom, 3.2 px on Filip
        // (feature scale). Above 2.5 px the head windows take the face-PnP pose.
        const noseErr = res.report && res.report.nose && res.report.nose.err;
        // 'room': the head windows keep the frame's pose whatever the face says (the A/B against face-PnP re-posing, 2026-09-17)
        S._headMoved = cropsMode === 'room' ? false : (noseErr != null ? +noseErr > 2.5 : null);
        alog(`head windows: ${S._headMoved ? 'face-PnP poses' : 'room poses'}${cropsMode === 'room' ? ' (Person crops: room pose)' : ''} (nose residual ${noseErr} px, head shift ${res.headShiftCm} cm)`);
      } catch (e) { if (!e.skipped) alog(`landmarks before training failed: ${e.message || e} — after training instead`); }
      if (S.gen !== gen) return;
      // ?crops=0 (experiment, 2026-09-15): no person windows — the user's 09-14 model without them held up against today's
      if (!cropsOn) alog('person crops: off (settings / ?crops=0)');
      else await av.addPersonCrops(session, files, { log: alog, faceCams, facePoints: S._facePoints || null, headMoved: S._headMoved, progress: (d, t) => { S.prep = { stage: 'crops', done: d, total: t }; } });
      if (S.gen !== gen) return;
      // ?cropsonly=1 (experiment, 2026-09-15): registration in full, the loss on the person's
      // native windows only — every iteration at native resolution on the person, the room
      // frames keep their poses for the hull and the cut (with ?cropmask=0 the windows keep
      // the room around the person: rectangular, no mask)
      if (new URLSearchParams(location.search).get('cropsonly') === '1') { session.opts.lossCams = (c) => !!c.crop; alog('crops only: the room frames leave the training loss'); }
      // ?skin=wPos,wNorm[,thickMm] (experiment, 2026-09-15, the user's idea): the face-skin
      // field — splats near the face are pulled onto the triangulated face mesh and flattened
      // along its normal (a skin, not a volume); hair and everything past the face oval untouched
      const skinQ = new URLSearchParams(location.search).get('skin');
      if (skinQ && S._lmRes) {
        const [wPos, wNorm, thickMm] = skinQ.split(',').map(Number);
        try {
          const { buildFaceSkinField } = await import('../avatar/faceskin.js');
          const field = await buildFaceSkinField(S._lmRes, session.recon.cams, { log: alog });
          if (field) session.opts.trainer = { ...(session.opts.trainer || {}), skinField: field, skin: { wPos: wPos || 0, wNorm: wNorm || 0, thickMm: thickMm > 0 ? thickMm : 2 } };
          alog(`skin term: position ${wPos || 0}, normal extent ${wNorm || 0}, thickness ${thickMm > 0 ? thickMm : 2} mm`);
        } catch (e) { alog(`skin field failed: ${e.message || e}`); }
      }
      // ?blob=R&blobface=1 (experiment, 2026-09-15): the blob clamp on the FACE only — blobs on
      // the skin (no edge-on lines in the close-up), discs everywhere else (clean oblique views)
      if (new URLSearchParams(location.search).get('blobface') === '1' && S._facePoints && S._facePoints.length > 50) {
        const P = S._facePoints; const c = [0, 0, 0]; for (const p of P) { c[0] += p[0] / P.length; c[1] += p[1] / P.length; c[2] += p[2] / P.length; }
        let rad = 0; for (const p of P) rad = Math.max(rad, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
        session.opts.trainer = { ...(session.opts.trainer || {}), blobRegion: { centre: c, radius: rad * 1.15 } };
        alog(`blob clamp on the face only: sphere radius ${(rad * 1.15 / (S._lmRes && S._lmRes.fit ? S._lmRes.fit.scale : 1) * 100).toFixed(0)} cm around the face points`);
      }
      // a dense seed on the face mesh, as ready-made flat Gaussians (?faceseed=1 or =N; off
      // by default: as points it came out worse — see app/avatar/faceseed.js)
      const fsq = new URLSearchParams(location.search).get('faceseed');
      S._faceSeed = null;
      if (fsq && fsq !== '0' && S._lmRes) {
        try { S._faceSeed = await av.faceSeedGaussians(session, files, S._lmRes, { count: +fsq > 1 ? +fsq : 20000, log: alog }); } catch (e) { alog(`face seed failed: ${e.message || e}`); }
      }
      // the whole head from the body model, protected from relocation (?headseed=N[,protectIters])
      const hsq = new URLSearchParams(location.search).get('headseed');
      if (hsq && hsq !== '0' && S._lmRes) {
        const [hn, hp, hf] = hsq.split(','); const hnN = +hn, hpN = +hp;
        try {
          S._faceSeed = await av.headSeedGaussians(session, files, S._lmRes, { count: hnN > 1 ? hnN : 50000, protectIters: hpN || 8000, log: alog, progress: (d, t) => { S.prep = { stage: 'headseed', done: d, total: t }; } });
          if (S._faceSeed && hf === 'f') S._faceSeed.freezePos = true;   // ?headseed=N,P,f — the discs keep their positions
          if (S._faceSeed && S._faceSeed.head) {   // the cloud seed leaves the head to the discs
            const { centre: hc, radius: hr } = S._faceSeed.head; const before = session.recon.points.length;
            session.recon.points = session.recon.points.filter((p) => Math.hypot(p.X[0] - hc[0], p.X[1] - hc[1], p.X[2] - hc[2]) > 1.3 * hr);
            alog(`head seed: ${before - session.recon.points.length} sparse points inside the head sphere removed from the cloud seed`);
          }
        } catch (e) { alog(`head seed failed: ${e.message || e}`); }
      }
      if (S.gen !== gen) return;
    }

    // 3) seed + trainer
    S.prep = { stage: 'seed', done: 0, total: 1 };
    await session.seed(S._faceSeed ? { appendGaussians: S._faceSeed } : {});
    if (S.gen !== gen) return;
    if (S.lodPlan) session.trainer.growLimit = S.lodPlan.levels[0];

    buildSceneFromSession();
    mountModelCanvas();
    startTraining();
  } catch (e) {
    if (S.gen !== gen) return;
    console.error(e);
    solveFailed(/parallax|overlap|initialization|matches|register/i.test(e.message)
      ? 'The photos don\'t overlap enough to connect into one scene.'
      : (e.message || 'An unexpected error occurred during reconstruction.'));
    trainDeadEnd();
  }
}

/** Unrecoverable training failure: the train screen is dead (faulted
 *  session, stopped loop) — leaving it up wedges play/chart/hub until the
 *  user navigates back by hand (seen: full-resolution device loss with a
 *  failed recovery, and continuations that die rebuilding at higher
 *  resolution). Tear down to the setup card exactly like the pipeline's own
 *  catch: dispose, reset, clear the dock, back to setup. The caller flashes
 *  or cards the WHY on top (the card is body-level and survives this). */
function trainDeadEnd() {
  try { S.session && S.session.dispose(); } catch { /* already down */ }
  S.session = null;
  clearTimeout(S._presentWd && S._presentWd.timer);
  S._presentWd = null;
  document.getElementById('cv-model')?.remove();
  document.getElementById('cv-model-cap')?.remove();
  gpuCanvas = null;
  S._capView = null;
  S.gen++;
  S.state = 'ready';
  S.trainVisuals = false;
  syncTrainingVisualsControl();
  dock('');
  backToSetup();
}

/** Back — one routine for every door that means it (the header Back, the
 *  brand title off the home tile, the phone's own Back through popstate's
 *  sibling path): a live run is secured in IndexedDB and Android app data,
 *  then home. */
async function goBack() {
  // The orbit viewer appears immediately after Stop & keep, while its state
  // readback is asynchronous. Back must be a save barrier, not a teardown race.
  if (S.state === 'done' && S._securing) {
    flash('Saving the finished model …', 60000);
    const saved = await S._securing.catch(() => false);
    if (!saved) return;
  }
  const runId = S.runId || (S._localRun && S._localRun.id);
  if (runId) await persistRunDetails(runId, { full: true }).catch(() => {});
  // a shared creation holds nothing precious — Back is simply the way
  // home. When the visitor came from within the app, real history.back()
  // is better: the bfcache restores the feed with its scroll intact.
  if (S._localRun && S._viewerOpen) { closeViewerToHome(); return; }
  if (S.restored) {
    let fromApp = false;
    try {
      const r = document.referrer && new URL(document.referrer);
      fromApp = !!r && r.origin === location.origin;
    } catch (err) { /* opaque referrer -> treat as external */ }
    if (fromApp && history.length > 1) history.back();
    else location.href = 'index.html';
    return;
  }
  // Back out of a live run secures it first (the checkpoint lives here
  // now — pausing is free), then waves the navigation through with no
  // confirm: there is nothing left to warn about. ?back=1 tells the fresh
  // boot this was a deliberate leave: the handoff marker stands down
  // instead of hijacking the boot into recovery, and the Yours tile is
  // the way back.
  // The checkpoint is mirrored to Android app data before navigation, so
  // Home can rediscover it even if IndexedDB is unavailable or evicted.
  if (S.state === 'train' && S.runId) {
    const saved = await checkpointRun('back').catch(() => null);
    if (!saved) {
      flash('Could not save this model yet. Training is still open; free storage and try Back again.', 9000);
      return;
    }
  }
  // Back from a run leaves the same way the logo does: a page navigation
  // home. The wall is never rendered OVER a live scene. The flag waves the
  // navigation past beforeunload — the run above is already secured.
  S._leaving = true;
  location.href = (S.state === 'train' || S.state === 'prep') ? 'index.html?back=1' : 'index.html';
}
/** After a failed solve: back to the setup card the run started from — with
 *  its settings and Start button, so a different resolution is one tap away.
 *  (In classic mode Start lives ON the detail card; the bare start card was
 *  a dead end: upload buttons, no presets, no way to retry.) */
function backToSetup() {
  if (!WALL_FIRST && S.photos.length) showDetail(S.preset || undefined);
  else showHome();
}

/** Dismiss every floating layer: modal cards (.upcard), capture dialogs
 *  (.xr-dialog — closed, not just removed, so their promises resolve and
 *  their object URLs are revoked), the Details sheet with its score graph,
 *  and the About sheet. Every landing on the home wall or back on a
 *  trained model funnels through here, so nothing opened on the previous
 *  screen survives the trip. */
function closeOverlays() {
  document.querySelectorAll('.upcard').forEach((c) => c.remove());
  document.querySelectorAll('.xr-dialog').forEach((d) => { if (d.open) d.close(); else d.remove(); });
  if (!$('details').hidden) {
    $('details').hidden = true;
    syncDetailsUrl();
    syncTrainingVisualsControl();
  }
  if (!$('about').hidden) $('about').hidden = true;
  chart?.clearHover();
  dchart?.clearHover();
}

/** The home wall, properly: the start card is NEVER revealed without its
 *  scenes mounted — a bare upload card with no presets is a bug, not a
 *  state. Every path that lands on home goes through here. */
function showHome() {
  closeOverlays();
  $('detail').hidden = true;
  $('start').hidden = false;
  $('btn-new').hidden = true;   // home is where Back leads — it needs none
  dock('');
  mountWall();
}

/** A restore link failed (file gone, share revoked, a CDN edge caching an
 *  old 403): say it in plain words on a card OVER the mounted home — never
 *  a silent toast over a half-built page. */
function loadFailed(what, why) {
  S._viewerOpen = false;
  document.getElementById('share-hero')?.remove();
  dock('');
  showHome();
  document.getElementById('failcard')?.remove();
  const c = document.createElement('div');
  c.className = 'upcard failcard';
  c.id = 'failcard';
  c.innerHTML = `
    <b>${esc(what)}</b>
    <p class="fail-why">${esc(why)}</p>
    <ul class="fail-tips">
      <li><b>The link may be stale.</b> The file behind it moved, expired or is not public.</li>
      <li><b>Reload to retry.</b> A fresh load also clears a stale error cached at the CDN edge.</li>
    </ul>
    <div class="upcard-row"><button class="btn btn-accent" id="fail-ok">Got it</button></div>`;
  document.body.appendChild(c);
  c.querySelector('#fail-ok').addEventListener('click', () => c.remove());
}

/** the solve failed — say so in plain words and teach the capture that works.
 *  GPU/driver failures get their OWN card: telling someone whose graphics
 *  driver stalled to "move sideways" blames their photography for our
 *  hardware trouble (seen in the wild: DXGI_ERROR_DEVICE_REMOVED at
 *  requestDevice, surfaced under capture tips). */
function solveFailed(why) {
  androidNotifStop(false);
  document.getElementById('failcard')?.remove();
  const c = document.createElement('div');
  c.className = 'upcard failcard';
  c.id = 'failcard';
  const gpuTrouble = /DXGI|DEVICE_REMOVED|D3D12|requestDevice|GPUAdapter|WebGPU|GPUQueue|GPUBuffer|GPUDevice|writeBuffer|graphics device|device lost|Vulkan|out.?of.?memory|training worker failed|detached/i.test(why);
  c.innerHTML = gpuTrouble ? `
    <b>Your graphics device stalled</b>
    <p class="fail-why">${esc(why)}</p>
    <ul class="fail-tips">
      <li><b>Your photos are fine.</b> The graphics driver refused to start — this is the computer, not the capture.</li>
      <li><b>Restart the browser completely</b> (quit, not just the tab) — that clears it in most cases.</li>
      <li><b>Still stuck?</b> Close other GPU-heavy apps and tabs, or restart the machine; on laptops, plug in power.</li>
    </ul>
    <div class="upcard-row"><button class="btn btn-accent" id="fail-ok">Got it</button></div>` : `
    <b>That capture didn't solve</b>
    <p class="fail-why">${esc(why)}</p>
    <ul class="fail-tips">
      <li><b>Move sideways.</b> Depth comes from a change of viewpoint — turning on the spot gives the solver nothing.</li>
      <li><b>Overlap generously.</b> Each photo should share most of its view with the one before.</li>
      <li><b>Pause, then shoot.</b> Motion blur, mirrors and glass are the usual killers.</li>
    </ul>
    <div class="upcard-row"><button class="btn btn-accent" id="fail-ok">Got it</button></div>`;
  document.body.appendChild(c);
  c.querySelector('#fail-ok').addEventListener('click', () => c.remove());
}

function onStage(e) {
  if (S.state !== 'prep') return;
  S.prep = e;
  if (e.stage === 'features' && e.detail) {
    S.feats.set(e.detail.image, e.detail);
    S.sel = e.detail.image;
    // stay ahead of the decoder so the beat shows photos, not black
    for (let k = 1; k <= 3; k++) {
      const nx = S.photos[Math.min(e.detail.image + k, S.photos.length - 1)];
      if (nx && nx.url) bmp(nx.url);
    }
    paintStrip();
  }
  if (e.stage === 'matching' && e.detail) {
    S.solveStats.pairsChecked = e.done;
    S.solveStats.pairsUsable = e.detail.usable;
    if (e.detail.pair) {
      S.lastPairEv = e.detail.pair; S.sel = e.detail.pair.i;
      // start both decodes now so the pair is ready by the time the stage
      // may switch to it (the LRU rarely holds a pair's photos already)
      const p1 = S.photos[e.detail.pair.i], p2 = S.photos[e.detail.pair.j];
      if (p1 && p1.url) bmp(p1.url);
      if (p2 && p2.url) bmp(p2.url);
    }
  }
  // every solver PASS — each focal candidate, the final run, the bracket re-check, the gap
  // retry — registers the images again in its own world frame. The overlay used to keep
  // every registration ever emitted, so the sparse-cloud stage showed several solves'
  // frustums superimposed: one ring plus a stray arc, "wrong poses" that the finished
  // reconstruction never had (the user, 2026-09-17, judging the solve by that view).
  // A pass event clears the ring; a re-registration of an image replaces its frustum.
  // Only a FINAL pass (the one whose poses become the solve) reaches the overlay: the focal-search
  // candidates and the init trials register a subsample in throwaway frames — clearing on those
  // made the ring hop and the cloud vanish six times before the real solve (the user, 09-17)
  if (e.stage === 'pass' && e.detail && e.detail.final) { S.regCams = []; S.regPts = null; S.regRgb = null; S.regPtsCount = 0; }
  if (e.stage === 'register' && e.detail && e.detail.R && e.detail.final) {
    const fr = S.session && S.session.frames && S.session.frames[e.detail.image];
    if (fr) {
      const cam = {
        i: e.detail.image, R: e.detail.R, t: e.detail.t, f: e.detail.f,
        w: fr.fw, h: fr.fh, cx: fr.fw / 2, cy: fr.fh / 2, state: 'placed',
      };
      const k = S.regCams.findIndex((c) => c.i === e.detail.image);
      if (k >= 0) S.regCams[k] = cam; else S.regCams.push(cam);
    }
    if (e.detail.cloud && e.detail.cloud.length) {
      S.regPts = e.detail.cloud;
      S.regRgb = e.detail.cloudRgb || null;
      S.regPtsCount = e.detail.points || 0;
    }
    // an overview that keeps FOLLOWING the growing reconstruction — framing
    // once at 3 cameras left everything after out of shot
    const cs = S.regCams.map(camCentre);
    if (cs.length) {
      const c = cs.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0])
        .map((v) => v / cs.length);
      let tgt = c;
      if (S.regPts && S.regPts.length) {
        // midpoint of camera ring and cloud puts both in frame
        let px = 0, py = 0, pz = 0;
        const m = S.regPts.length / 3;
        for (let i = 0; i < S.regPts.length; i += 3) {
          px += S.regPts[i]; py += S.regPts[i + 1]; pz += S.regPts[i + 2];
        }
        tgt = [(c[0] + px / m) / 2, (c[1] + py / m) / 2, (c[2] + pz / m) / 2];
      }
      // radius from the CAMERAS only — the cloud has outliers, the ring doesn't
      const r = Math.max(1e-3, ...cs.map((p) => Math.hypot(p[0] - tgt[0], p[1] - tgt[1], p[2] - tgt[2])));
      vp.scene = { center: tgt, radius: r * 1.1, xyz: null, rgb: null };
      if (S.regCams.length === 3) {
        vp.detectUp(S.regCams);
        vp.frameScene();
      } else if (S.regCams.length > 3) {
        vp.detectUp(S.regCams);
        const k = 0.3;   // damped follow: no jumps, just a slow zoom-out
        for (let i = 0; i < 3; i++) vp.target[i] += (tgt[i] - vp.target[i]) * k;
        vp.dist += (r * 2.4 - vp.dist) * k;
        vp.dirty = true;
      }
    }
  }
}

/** the display-side scene: every photograph, placed or not, plus the cloud */
function buildSceneFromSession() {
  const ses = S.session;
  const recon = ses.recon;
  const panoish = !!ses.rigInfo || (ses.frames && ses.frames.length > S.photos.length);
  let cams;
  if (panoish) {
    const best = new Array(S.photos.length).fill(null);
    if (recon && recon.cams) {
      recon.cams.forEach((rc) => {
        const rig = ses.rigInfo ? ses.rigInfo[rc.imgIdx] : null;
        const panoIdx = rig ? rig.id : Math.floor(rc.imgIdx / 6);
        if (panoIdx >= 0 && panoIdx < S.photos.length) {
          const fr = (ses.frames && ses.frames[rc.imgIdx]) || {};
          const m = fr.name && String(fr.name).match(/_f(\d+)$/);
          const face = m ? +m[1] : (rc.imgIdx % 6);
          const ci = ses.trainer && ses.trainer.camMeta
            ? ses.trainer.camMeta.findIndex((meta) => meta.imgIdx === rc.imgIdx) : -1;
          if (!best[panoIdx] || face < best[panoIdx].face) {
            best[panoIdx] = { rc, ci, face };
          }
        }
      });
    }
    cams = S.photos.map((p, i) => {
      const b = best[i];
      if (!b) return { i, ci: -1, R: null, t: null, url: p.url, name: p.name,
                       w: 4, h: 3, cx: 2, cy: 1.5, state: 'unplaced', feats: 0, psnr: null, pano: true };
      const fr = (ses.frames && ses.frames[b.rc.imgIdx]) || {};
      const w = fr.fw || 1000, h = fr.fh || 1000;
      return {
        i, ci: b.ci, R: b.rc.R, t: b.rc.t, f: b.rc.f,
        w, h, cx: w / 2, cy: h / 2,
        url: p.url, name: p.name,
        state: b.ci === ses.holdout || (ses.testCams && ses.testCams.includes(b.ci)) ? 'holdout' : 'placed',
        feats: (S.feats.get(b.rc.imgIdx) || {}).n || 0,
        psnr: null, pano: true,
      };
    });
  } else {
    cams = S.photos.map((p, i) => ({
      i, url: p.url, name: p.name, R: null, t: null, state: 'unplaced', ci: -1, psnr: null,
    }));
    if (ses.trainer && ses.trainer.camMeta) {
      ses.trainer.camMeta.forEach((m, ci) => {
        const c = cams[m.imgIdx];
        if (!c) return;
        const rc = recon && recon.cams && recon.cams.find((r) => r.imgIdx === m.imgIdx);
        const fr = (ses.frames && ses.frames[m.imgIdx]) || {};
        const w = fr.fw || 1000, h = fr.fh || 1000;
        Object.assign(c, {
          R: m.R, t: m.t, f: rc ? rc.f : (m.f || 1000),
          w, h, cx: w / 2, cy: h / 2,
          state: ci === ses.holdout || (ses.testCams && ses.testCams.includes(ci)) ? 'holdout' : 'placed',
          ci,
          feats: (S.feats.get(m.imgIdx) || {}).n || 0,
        });
      });
    }
  }
  const capturedPoints = S.preset?.capture?.points || [];
  const savedCloud = S._localRun?.recon?.cloud || S.restored?.reconJson?.cloud || null;
  const { xyz, rgb } = buildDisplayCloud(capturedPoints, savedCloud, (recon && recon.points) || []);
  S.scene = {
    cams, xyz, rgb,
    center: (ses.model && ses.model.center) || [0, 0, 0],
    radius: (ses.model && ses.model.radius) || 1,
  };
  vp.setScene(S.scene);
  if (recon && recon.known) {
    // known-camera runs keep the capture world's y-up frame — level to its
    // gravity, not to the cameras' sensor-up (portrait-held phones point it
    // sideways, which used to roll the view 90°)
    vp.setUp(gravityUpForKnownRun({
      persisted: S._localRun && S._localRun.recon && S._localRun.recon.levelingUp,
      leveling: S.preset && S.preset.capture && S.preset.capture.metadata && S.preset.capture.metadata.leveling,
    }));
  } else {
    vp.detectUp(cams);
  }
  paintStrip();
}

// ── training ────────────────────────────────────────────────────────────────
/** The library tile's name for a run of this set (shared by the save at
 *  run start and the in-memory adoption at finish — the two must agree). */
function localRunName(preset) {
  const pid = String((preset && preset.id) || '');
  let name = (preset && preset.name) || 'Your photos';
  if ((pid === '__own' || pid === '__last') && (name === 'Your photos' || pid === '__last')) {
    name = 'Local Scene';
  }
  return name;
}

function startTraining() {
  S.state = 'train';
  // a history entry of our own: the phone's Back (and the browser's) pops
  // here, into the popstate checkpoint-leave — never straight out of the app
  if (!(history.state && history.state.sj === 'run')) history.pushState({ sj: 'run' }, '');
  S.trainVisuals = defaultTrainingVisuals();
  syncTrainingVisualsControl();
  const carryMinutes = S._carryTrainElapsed ? (S.minutes || 0) : 0;
  S._carryTrainElapsed = false;
  S.trainT0 = performance.now() - carryMinutes * 60000;
  S.etaAt = null;   // the countdown re-anchors on this run's own pace
  S._viewRemounts = 0; S._viewFailedFlash = false;   // fresh view-error budget
  // dual-GPU desktops: WebGPU can land on the integrated GPU even when a
  // dedicated card sits idle (seen in the wild: 3 h for a run a 3070 does in
  // minutes — the NVIDIA control panel does NOT govern this). Say so once.
  {
    const gi = (S.session && S.session.gpu && S.session.gpu.info) || {};
    if (/intel/i.test(gi.vendor || '') && !matchMedia('(any-pointer: coarse)').matches &&
        !localStorage.getItem('splatjs_gpu_note')) {
      localStorage.setItem('splatjs_gpu_note', '1');   // once — Intel-only laptops are fine
      flash('Training runs on Intel graphics. If this PC has a dedicated GPU, tell Windows to ' +
        'use it for your browser: Settings → System → Display → Graphics — then reload.', 15000);
    }
  }
  // local library: every run is listed from the moment it starts — a small
  // record now, the finished result later (see finish())
  // run identity: a continuation keeps its record; a same-session re-entry
  // (device-lost recovery, a double-pressed Start) keeps the live one —
  // minting new here duplicated the tile on every fault recovery. Fresh
  // runs resolve below: an own capture adopts its stale same-source
  // record instead of duplicating it.
  const resumeId = S._resumeRunId;
  S._securing = null;   // the previous run's finish work, if any, is history now
  S._resumeRunId = null;
  const keepId = resumeId || S.runId || null;
  S.runId = keepId || `run_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  androidNotifStart();   // background-safe from the first cycle
  stageHeadlessRun();   // migrate-on-swipe: the files-dir handoff stages from the first cycle
  S._ckptIter = 0;
  (async () => {
    const runGen = S.gen;
    const pid = String((S.preset && S.preset.id) || '');
    // own captures carry a stable source key — the adoption lookup below
    const capKind = S._capRec && S._capRec.kind;
    const capFiles = (S.preset && S.preset.files) || [];
    const { saveRun, patchRun, listRuns, captureSrcKey } = await import('./store.js');
    const nativeCaptureId = S.preset?.capture?.metadata?.nativeCaptureId || null;
    const srcKey = nativeCaptureId
      ? `native:${nativeCaptureId}`
      : (pid === '__own' || pid === '__last') && capKind && capFiles.length
        ? captureSrcKey(capKind + (S.preset.capture?.transforms ? '+poses' : ''), capFiles) : null;
    // marking a record live moves only the training head (status, horizon,
    // diagnostics) — name, thumb, poses and source photos stay untouched
    const livePatch = () => ({
      status: 'training', maxIters: S.maxIters, ...(srcKey ? { srcKey } : {}),
      detailSolve: captureRunSolveDetails(),
      detailProgress: captureRunProgressDetails(),
      detailGpu: gpuFacts(false),
    });
    if (keepId) {
      // the record already exists (thumb, checkpoint, name, and for a
      // finished run its last result) — just mark it live; the run's own
      // checkpoints and its final result overwrite the trained state in
      // place while the record's original camera poses stay untouched
      if (await patchRun(keepId, livePatch())) return;
      // storage lost the record mid-run — recreate it under the same id
    } else if (srcKey) {
      // a fresh run of an already-run capture replaces the stale record
      // instead of duplicating its tile — but only a stale INTERRUPTED
      // one: finished results are never adopted (a re-run that fails must
      // not eat a good result), and neither is a record written in the
      // last minute (a live run in another tab)
      const stale = (await listRuns().catch(() => []))
        .find((r) => r.srcKey === srcKey && r.status !== 'finished' &&
          Date.now() - (r.updatedAt || 0) > 60000);
      if (stale && S.gen === runGen && await patchRun(stale.id, livePatch())) {
        S.runId = stale.id;
        return;
      }
    }
    const runId = S.runId;
    // own scenes are SCENES, not photos — a continued run keeps the name it
    // already carries; the tile's description holds the capture details
    let runName = localRunName(S.preset);
    await saveRun({
      id: runId, name: runName, ...(srcKey ? { srcKey } : {}),
      status: 'training', createdAt: Date.now(), updatedAt: Date.now(),
      iter: S.iter || 0, maxIters: S.maxIters, splats: S.splats || 0, psnr: shownScore('train') ?? null,
      frames: (S.photos || []).length,
      // own-photo runs can train again from the device's capture store
      ownSrc: pid === '__own' || pid === '__last',
      detailSolve: captureRunSolveDetails(),
      detailProgress: captureRunProgressDetails(),
      detailGpu: gpuFacts(false),
    });
    // Save the model record before the optional thumbnail fetch; Back/close
    // should not race image decoding and lose the run's first snapshot.
    if (S.gen !== runGen) return;
    try {
      const p = S.photos && S.photos[0];
      if (p && p.url) {
        const bmp = await createImageBitmap(await (await fetch(p.url)).blob(), { resizeWidth: 320 });
        const cv = document.createElement('canvas');
        cv.width = bmp.width; cv.height = bmp.height;
        cv.getContext('2d').drawImage(bmp, 0, 0);
        bmp.close();
        const thumb = await new Promise((res) => cv.toBlob(res, 'image/webp', 0.8));
        if (thumb && S.gen === runGen) await patchRun(runId, { thumb });
      }
    } catch { /* the tile just goes textless */ }
  })().catch(() => {});
  const first = S.scene.cams.find((c) => c.R && c.state !== 'holdout') || S.scene.cams[0];
  if (first && first.R) {
    S.sel = first.i;
    vp.freeF = null;
    vp.syncTo(first);   // exactly the first photographer's viewpoint
    paintStrip();
  }
  renderControls();
  dock('train');
  // The first visual-sync ran before the training dock existed. Re-run it now
  // so the initial hidden-visuals state expands the graph into the stage.
  syncTrainingVisualsControl();
  if (chart) {
    chart.train = (S.session.lossHistory || []).map(([i, v]) => [i, v]);
    chart.hold = (S.holdHist || []).slice();
    chart.maxIter = S.maxIters;
    chart.events = (S.chartEvents || []).map((e) => ({ ...e, at: e.iter / S.maxIters }));
    chart.draw();
  }
  S.session.start();
}

function syncTrainButtons() {
  const ses = S.session;
  const on = !!(ses && ses.training);
  const b = $('t-play');
  if (b) {
    b.dataset.state = on ? 'pause' : 'play';
    b.innerHTML = on
      ? '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z"/></svg>'
      : '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M8 4.8v14.4l12.4-7.2z"/></svg>';
  }
  const f = $('t-finish');
  if (f) f.hidden = on;
  const p = $('t-prune');
  if (p) p.disabled = !ses || !ses.trainer || !!S._pruning || !!S._growing;
}

/** The ceiling's question: the session paused itself at a full cap — prune
 *  to half and grow on, or keep training at this size. */
function showCapCard(e) {
  syncTrainButtons();
  document.getElementById('capcard')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'capcard';
  card.innerHTML = `
    <b>Splat cap reached (${fmt(e.splats)})</b>
    <span class="sog-status">Growth is full. Prune and grow on, or keep training at this size.</span>
    <div class="row" id="cap-pick">
      <button class="btn" data-ratio="0.75">Prune to ¾</button>
      <button class="btn btn-accent" data-ratio="0.5">Prune to ½</button>
      <button class="btn" data-ratio="0.25">Prune to ¼</button>
      <input id="cap-custom" type="number" min="5" max="95" step="1" value="50" style="width:4.5em" title="Custom keep percentage (5–95)">
      <button class="btn" data-custom="1">Prune to %</button>
      <button class="btn linkish" id="cap-keep">Keep training at cap</button>
    </div>`;
  $('stage').appendChild(card);
  card.querySelector('#cap-pick').addEventListener('click', (ev) => {
    const d = ev.target && ev.target.dataset ? ev.target.dataset : null;
    if (!d) return;
    if (d.custom) {
      const keep = readKeepCustom('cap-custom');
      card.remove();
      runPruneAndContinue(keep);
      return;
    }
    if (d.ratio == null || d.ratio === '') return;   // the keep button answers for itself
    card.remove();
    runPruneAndContinue(parseFloat(d.ratio));
  });
  card.querySelector('#cap-keep').addEventListener('click', () => {
    card.remove();
    if (S.session) S.session.start();
    syncTrainButtons();
    flash('Training on at the cap — growth stays off.', 4000);
  });
}

/** Manual prune/grow, any time: pick how much to prune (the flow reseeds
 *  the trainer and grows on), or trigger one growth pass right now. */
function showPrunePicker() {
  const ses = S.session;
  if (!ses || !ses.trainer || S._pruning || S._growing) return;
  document.getElementById('prunepick')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'prunepick';
  card.innerHTML = `
    <b>Prune / grow</b>
    <span class="sog-status">Fewer splats with room to regrow — or grow into the error right now.</span>
    <div class="row" id="prunepick-pick">
      <!-- labels prune; data-ratio stays the keep fraction the flow wants -->
      <button class="btn" data-ratio="0.75">Prune ¼</button>
      <button class="btn" data-ratio="0.5">Prune ½</button>
      <button class="btn" data-grow="1">Grow now</button>
      <input id="prunepick-custom" type="number" min="1" max="99" step="1" value="50" style="width:4.5em" title="Custom prune percentage (1–99)">
      <button class="btn" data-custom="1">Prune %</button>
      <button class="btn linkish" data-ratio="">Cancel</button>
    </div>`;
  $('stage').appendChild(card);
  card.querySelector('#prunepick-pick').addEventListener('click', (e) => {
    // the custom input lives in this row: only BUTTON clicks answer —
    // tapping the input to type must not dismiss the card
    if (!e.target || e.target.tagName !== 'BUTTON') return;
    const d = e.target.dataset;
    const customKeep = d.custom ? readPruneCustom('prunepick-custom') : 0;
    if (d.custom && customKeep == null) {
      const input = $('prunepick-custom');
      input.focus(); input.select();
      flash('Enter a whole prune percentage from 1 to 99.', 6000);
      return;
    }
    card.remove();
    if (d.grow) { runGrowNow(); return; }
    if (d.custom) { runPruneAndContinue(customKeep); return; }
    if (d.ratio != null && d.ratio !== '') runPruneAndContinue(parseFloat(d.ratio));
  });
}

/** Manual growth pass, any time: one immediate refine outside the schedule
 *  (the Prune/Grow button's other half). Quick — no progress card. */
async function runGrowNow() {
  const ses = S.session;
  if (!ses || !ses.trainer || S._pruning || S._growing) return;
  S._growing = true;
  syncTrainButtons();
  try {
    const r = await ses.growNow();
    flash(`Grew +${(r.grown || 0).toLocaleString()} splats → ${(r.n || 0).toLocaleString()} total`, 4000);
  } catch (e) {
    console.error(e);
    flash(`Could not grow: ${e.message}`, 8000);
  } finally {
    S._growing = false;
    syncTrainButtons();
  }
}

async function runPruneAndContinue(ratio = 0.5) {
  const ses = S.session;
  if (!ses || !ses.trainer || S._pruning || S._growing) return;
  S._pruning = true;
  syncTrainButtons();
  document.getElementById('prunecard')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'prunecard';
  card.innerHTML = `
    <b>Pruning the model</b>
    <span class="sog-status" id="prune-status">Reading the model …</span>
    <div class="prep-meter"><i id="prune-bar" style="width:0%"></i></div>`;
  $('stage').appendChild(card);
  try {
    const st = card.querySelector('#prune-status');
    const bar = card.querySelector('#prune-bar');
    let from = 0;
    const out = await ses.pruneAndContinue(ratio, {
      onStatus: (m) => { if (st) st.textContent = String(m); },
      onProgress: (p) => {
        if (!from) from = p.currentCount;
        const span = from - p.targetCount;
        if (bar && span > 0) bar.style.width = `${(100 * (from - p.currentCount) / span).toFixed(1)}%`;
      },
    });
    card.remove();
    const d = out.delta == null ? 'no holdout to score against'
      : `${out.delta >= 0 ? '+' : ''}${out.delta.toFixed(2)} dB holdout`;
    flash(`Pruned ${fmt(out.from)} → ${fmt(out.to)} splats (${d}) — ${out.resumed ? 'training on with room to grow.' : 'press play to grow on.'}`, 7000);
  } catch (err) {
    card.remove();
    console.error(err);
    // a failed prune resumes a training run onto the unpruned model (the
    // session owns the loop now); an already-paused run stays paused
    flash(`Prune failed: ${err.message} — ${ses.training ? 'training on with the unpruned model.' : 'press play to resume.'}`, 8000);
  } finally {
    S._pruning = false;
    syncTrainButtons();
  }
}

function toggleTrain() {
  if (!S.session) return;
  if (S.session.training) S.session.pause();
  else S.session.start();
  const b = $('t-play');
  const on = S.session.training;
  if (b) {
    b.dataset.state = on ? 'pause' : 'play';
    // SVG icons, not font glyphs: the ▶ character carries right-side bearing
    // that drifts it off optical centre — a drawn triangle cannot
    b.innerHTML = on
      ? '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z"/></svg>'
      : '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M8 4.8v14.4l12.4-7.2z"/></svg>';
  }
  const f = $('t-finish');
  if (f) f.hidden = on;   // paused = the moment "stop here" makes sense
  // a user-pressed pause takes ownership from any touch-hold: the release
  // must not resume over it
  if (!on) {
    S._dragPaused = false;
    // pausing is free now — the checkpoint moved to leaving (Back). Say so
    // once: the old pause flash promised "safe to close", which pause no
    // longer means.
    try {
      if (!localStorage.getItem('splatjs_pause_hint')) {
        localStorage.setItem('splatjs_pause_hint', '1');
        flash('Paused. Your progress saves when you leave with Back.', 7000);
      }
    } catch { /* private mode */ }
  }
  headlessStatusWrite(on ? 'active' : 'paused');
  // Shade control: where the bridge can flip the notification's button the
  // service stays up across the pause (Resume in the shade); old bridges
  // shed it and restart it instead.
  if (on) { if (!notifSetPaused(false)) androidNotifStart(); }
  else if (!notifSetPaused(true)) androidNotifStop(false);
}

// Local-run diagnostics are a compact companion to the saved model. The
// reconstruction already carries cameras and a coarse score curve; these
// snapshots retain the solver's landmarks/match sample plus the detailed
// score and timing history that the Details sheet needs after a reload.
function compactRows(rows, max = 600) {
  if (!Array.isArray(rows) || !rows.length) return [];
  const stride = Math.max(1, Math.ceil(rows.length / max));
  const out = [];
  for (let i = 0; i < rows.length; i += stride) out.push(Array.from(rows[i] || []));
  const last = rows[rows.length - 1];
  if (out.length && rows.length > 1 && rows[rows.length - 1] !== rows[Math.min(rows.length - 1, (out.length - 1) * stride)]) {
    out.push(Array.from(last || []));
  }
  return out;
}

function captureRunSolveDetails() {
  const pair = S.lastPairEv;
  const samples = pair ? (pair.sample || []).slice(0, 50).map((p) => Array.from(p || [])) : [];
  const referenced = new Map();
  const refer = (image, index) => {
    if (!Number.isInteger(index) || index < 0) return;
    if (!referenced.has(image)) referenced.set(image, new Set());
    referenced.get(image).add(index);
  };
  for (const [a, b] of samples) { refer(pair.i, a); refer(pair.j, b); }
  const remaps = new Map();
  const features = [...(S.feats || new Map())].map(([image, f]) => {
    const n = Math.min(f.x ? f.x.length : 0, f.y ? f.y.length : 0);
    const indices = [];
    for (let i = 0; i < Math.min(1200, n); i++) indices.push(i);
    for (const i of referenced.get(image) || []) if (i < n && i >= 1200) indices.push(i);
    const remap = new Map(indices.map((old, next) => [old, next]));
    remaps.set(image, remap);
    return {
      image, n: f.n || 0,
      x: indices.map((i) => f.x[i]), y: indices.map((i) => f.y[i]),
    };
  });
  const savedPair = pair ? {
    i: pair.i, j: pair.j,
    sample: samples.map(([a, b]) => [remaps.get(pair.i)?.get(a), remaps.get(pair.j)?.get(b)])
      .filter(([a, b]) => a != null && b != null),
  } : null;
  return {
    solveStats: { ...(S.solveStats || {}) },
    features,
    lastPair: savedPair,
    selected: S.sel | 0,
  };
}

function captureRunProgressDetails() {
  const ses = S.session;
  const perf = (ses && ses.perf) || {};
  const history = ses && Array.isArray(ses.lossHistory) && ses.lossHistory.length
    ? ses.lossHistory : ((S.restored && S.restored.stats && S.restored.stats.chart) || []);
  const holds = S.holdHist && S.holdHist.length
    ? S.holdHist : ((S.restored && S.restored.stats && S.restored.stats.holds) || []);
  return {
    chart: compactRows(history, 800),
    holds: compactRows(holds, 800),
    events: (S.chartEvents || []).slice(-300).map((e) => ({ ...e })),
    perf: {
      frames: (perf.frames || []).slice(-6000).map((r) => Array.from(r || [])),
      marks: (perf.marks || []).slice(-1200).map((m) => ({ ...m })),
    },
    metrics: (S.perfMetrics || []).slice(-2000).map((m) => Array.from(m || [])),
    settings: { ...(S._savedDetailSettings || S.settings || {}) },
    tileGrad: (ses && ses.trainer && ses.trainer.tileGrad) ?? null,
    maxIters: S.maxIters || 0,
    iter: S.iter || 0,
    minutes: S.state === 'train'
      ? Math.max(S.minutes || 0, Math.round((performance.now() - (S.trainT0 || performance.now())) / 60000))
      : (S.minutes || 0),
    psnrTrain: shownScore('train') ?? null,
    psnrHold: shownScore('hold') ?? null,
  };
}

async function persistRunDetails(id = S.runId, { full = false } = {}) {
  if (!id) return false;
  const patch = {
    detailProgress: captureRunProgressDetails(),
    detailGpu: S.state === 'train' || !S._savedGpuFacts ? gpuFacts(false) : S._savedGpuFacts,
  };
  if (full) patch.detailSolve = captureRunSolveDetails();
  const { patchRun } = await import('./store.js');
  const updated = await patchRun(id, patch);
  if (updated && S._localRun && S._localRun.id === id) Object.assign(S._localRun, patch);
  return updated;
}

function restoreRunDetails(run, session = S.session, opts = {}) {
  if (!run) return;
  const solve = run.detailSolve || {};
  S.solveStats = solve.solveStats
    ? { pairsChecked: 0, pairsUsable: 0, solveSec: 0, ...solve.solveStats }
    : { pairsChecked: 0, pairsUsable: 0, solveSec: 0 };
  S.feats = new Map(Array.isArray(solve.features) ? solve.features.map((f) => [f.image, {
    n: f.n || 0, x: Float32Array.from(f.x || []), y: Float32Array.from(f.y || []),
  }]) : []);
  S.lastPairEv = solve.lastPair ? {
    i: solve.lastPair.i, j: solve.lastPair.j,
    sample: (solve.lastPair.sample || []).map((p) => Array.from(p || [])),
  } : null;
  if (Number.isFinite(solve.selected)) S.sel = solve.selected;
  const frameCount = session && session.frames ? session.frames.length : 0;
  if (frameCount) S.sel = Math.max(0, Math.min(frameCount - 1, S.sel | 0));
  S._savedGpuFacts = run.detailGpu || null;
  // a continued run restarts at its state checkpoint, but the record's
  // histories can run ahead of it (a Details persist lands checkpoint-free)
  // — trim them to the resume point so the curve continues instead of
  // plotting over itself. Viewing keeps the full record (no trim).
  const progress = opts.trimToIter != null
    ? trimProgressToIter(run.detailProgress || {}, opts.trimToIter)
    : (run.detailProgress || {});
  S._savedDetailSettings = progress.settings || null;
  S.chartEvents = Array.isArray(progress.events) ? progress.events.map((e) => ({ ...e })) : [];
  S.holdHist = Array.isArray(progress.holds) ? progress.holds.map((r) => Array.from(r || [])) : [];
  S.perfMetrics = Array.isArray(progress.metrics) ? progress.metrics.map((m) => Array.from(m || [])) : [];
  const lastMetric = S.perfMetrics[S.perfMetrics.length - 1];
  if (lastMetric && Number.isFinite(+lastMetric[2])) S.itersPerSec = +lastMetric[2];
  const fallbackChart = (S.restored && S.restored.stats && S.restored.stats.chart) || [];
  const history = Array.isArray(progress.chart) && progress.chart.length ? progress.chart : fallbackChart;
  if (session) {
    session.setLossHistory(history);
    session.perf = session.perf || { frames: [], marks: [] };
    const frameRows = ((progress.perf && progress.perf.frames) || []).map((r) => Array.from(r || []));
    const markRows = ((progress.perf && progress.perf.marks) || []).map((m) => ({ ...m }));
    const metricRows = S.perfMetrics;
    const lastT = frameRows.length ? frameRows[frameRows.length - 1][0] : 0;
    const timeShift = frameRows.length ? performance.now() + 1 - lastT : 0;
    session.perf.frames = frameRows.map((r) => { if (r.length) r[0] += timeShift; return r; });
    session.perf.marks = markRows.map((m) => ({ ...m, ...(Number.isFinite(m.t) ? { t: m.t + timeShift } : {}) }));
    S.perfMetrics = metricRows.map((m) => { if (m.length) m[0] += timeShift; return m; });
  }
  if (progress.minutes != null) S.minutes = Math.max(S.minutes || 0, progress.minutes || 0);
  if (progress.psnrTrain != null) S.psnrTrain = progress.psnrTrain;
  if (progress.psnrHold != null) S.psnrHold = progress.psnrHold;
}

// ── crash-safe checkpoint ───────────────────────────────────────────────────
// The raw trainer state — a straight GPU readback, no PLY text, no SOG
// k-means — goes into the run record when the user leaves the run (Back,
// in-app or the phone's), and (desktop only) when the tab goes hidden
// mid-training. Pausing writes nothing: it is just a breather. One slot per
// run, overwritten: leaving means the page may be closed, and the Yours tile
// continues the run from where it stopped (float16-close — training
// re-converges the quantization noise). The blob is dropped again the
// moment the run finishes properly — the sog takes over.
/** Crash-safe checkpoint. Returns the secured { state, recon } on success
 *  (finish() adopts them into the in-memory run record — no second GPU
 *  readback, no store round-trip), null when there was nothing to keep. */
async function checkpointRun(reason) {
  if (!S.runId || !S.session || !S.session.trainer) return null;
  // 'finish' runs in the done state, BEFORE the export chain: the result is
  // secured even if sog compression or the device dies right after training
  if (reason === 'finish' ? S.state !== 'done' : S.state !== 'train') return null;
  const it = S.session.trainer.iter | 0;
  if (S._ckptBusy || (reason === 'hidden' && it < 200)) return null;
  // hidden fires on every tab switch — only rewrite after real progress
  if (reason === 'hidden' && it - (S._ckptIter || 0) < 500) return null;
  S._ckptBusy = true;
  const gen = S.gen;
  try {
    const { packState, buildReconJson, trainBudget } = await import('./session_io.js');
    const state = new Blob([await packState(S.session)]);
    if (S.gen !== gen) return null;
    const recon = buildReconJson(S);
    const detailProgress = captureRunProgressDetails();
    const patch = {
      iter: it, splats: S.splats || S.session.trainer.n,
      psnr: S.psnrHold ?? S.psnrTrain ?? null,
      cap: trainBudget(S.session),   // the configured budget carries over, not the allocation (seed x capMult under-allocates — the continue card must see the limit training runs under)
      state, recon, ckptAt: Date.now(), minutes: detailProgress.minutes,
      detailSolve: captureRunSolveDetails(),
      detailProgress,
      detailGpu: gpuFacts(false),
    };
    // Android app data is the durable fallback and must not depend on an IDB
    // write succeeding: quota/process-profile loss was hiding completed runs.
    const fileSaved = await writeDurableRunCheckpoint(state, recon, {
      status: reason === 'finish' ? 'finished' : (S.session.training ? 'active' : 'paused'),
      owner: 'activity', iter: it, at: Date.now(),
    }).catch((e) => { console.warn('[checkpoint] Android mirror failed', e); return false; });
    let idbSaved = false;
    try {
      const { patchRun, saveRun } = await import('./store.js');
      idbSaved = await patchRun(S.runId, patch);
      if (!idbSaved) {
        await saveRun({ id: S.runId, name: localRunName(S.preset), status: 'training',
          createdAt: Date.now(), updatedAt: Date.now(), maxIters: S.maxIters,
          frames: (S.loadedFiles || []).length, ownSrc: true, ...patch });
        idbSaved = true;
      }
    } catch (e) { console.warn('[checkpoint] IndexedDB mirror failed', e); }
    if (!fileSaved && !idbSaved) throw new Error('neither Android app data nor IndexedDB accepted the checkpoint');
    S._ckptIter = it;
    if (reason === 'back') flash('Progress saved on this device — safe to close, continue anytime from Yours.', 6000);
    return { state, recon };
  } catch (e) {
    console.warn('checkpoint failed', e);   // best-effort: never disturb a run
  } finally { S._ckptBusy = false; }
  return null;
}

function formatTrainingScore(psnr) {
  if (psnr == null || Number.isNaN(psnr)) return '—';
  return (S.scoreScale === 'similarity' ? psnrToApproxSsimPercent(psnr) : psnr).toFixed(2);
}

/** The score as shown: the ZLEMA curve's end once a chart has drawn one,
 *  else the latest raw sample. Raw stays stored everywhere — dots, hover
 *  and history all read it — only the headline number is filtered. */
function shownScore(which) {
  const raw = which === 'hold' ? S.psnrHold : S.psnrTrain;
  for (const c of [chart, dchart]) {
    if (!c) continue;
    const f = which === 'hold' ? c.lastHoldF : c.lastTrainF;
    if (f != null) return f;
  }
  return raw;
}

function renderTrainingScore() {
  const similarity = S.scoreScale === 'similarity';
  const unit = similarity ? '%' : 'dB';
  const trainEl = $('t-ptrain');
  if (trainEl) trainEl.textContent = formatTrainingScore(shownScore('train'));
  const trainUnit = $('t-punit');
  if (trainUnit) trainUnit.textContent = unit;
  const holdEl = $('t-phold');
  if (holdEl) holdEl.textContent = formatTrainingScore(shownScore('hold'));
  const holdUnit = $('t-hunit');
  if (holdUnit) holdUnit.textContent = unit;
}

// Score plotting rides the next animation frame: metrics arrive mid-frame
// from the training loop, so painting the chart + details panels
// synchronously freezes the frame that delivered them. Data pushes stay
// synchronous — only the painting defers, coalesced so a burst of readouts
// still paints once. The train-length gate preserves the chart's
// first-samples rule (pushes only happen past the ramp, so a non-empty
// series means the gate already cleared).
let metricsRaf = 0;
function queueMetricsPaint() {
  if (metricsRaf) return;
  metricsRaf = requestAnimationFrame(() => {
    metricsRaf = 0;
    if (chart && chart.train.length) chart.draw();
    if ($('details').hidden) return;
    if (S.detailTab === 'score') {
      if (dchart) {
        const history = (S.session && S.session.lossHistory) || [];
        dchart.train = history.map(([i, v]) => [i, v]);
        dchart.hold = (S.holdHist || []).slice();
        dchart.maxIter = S.maxIters;
        dchart.events = (S.chartEvents || []).map((e) => ({ ...e, at: e.iter / S.maxIters }));
        dchart.draw();
      }
    } else if (S.detailTab === 'perf') {
      $('d-perf').textContent = buildPerfReport();
    }
  });
}

function onMetrics(m) {
  S.iter = m.iter;
  S.splats = m.splats;
  S.itersPerSec = m.itersPerSec;
  androidNotifTick();
  // LOD training: hold the model at each detail level for a polish window,
  // snapshot it, then raise the growth limit and move on
  const LP = S.lodPlan;
  if (LP && S.session?.trainer && LP.idx < LP.levels.length - 1) {
    const tr = S.session.trainer;
    if (LP.holdUntil == null && tr.n >= LP.levels[LP.idx]) {
      LP.holdUntil = m.iter + LP.polish;
    } else if (typeof LP.holdUntil === 'number' && m.iter >= LP.holdUntil) {
      LP.holdUntil = 'snapping';
      const lvl = LP.levels[LP.idx];
      S.session.exportPlyBlob().then((blob) => {
        LP.snaps.push({ n: lvl, blob });
        LP.idx++;
        LP.holdUntil = null;
        tr.growLimit = LP.levels[LP.idx];
        flash(`LOD level ${fmt(lvl)} snapshotted — growing on`, 3500);
      }).catch(() => { LP.holdUntil = null; });
    }
  }
  (S.perfMetrics ??= []).push([Math.round(performance.now()), m.iter, m.itersPerSec,
    m.psnrTrain != null ? m.psnrTrain.toFixed(2) : '', m.psnrHold != null ? m.psnrHold.toFixed(2) : '']);
  if (m.psnrTrain != null) S.psnrTrain = m.psnrTrain;
  if (m.psnrHold != null) S.psnrHold = m.psnrHold;
  if (m.psnrHold != null) (S.holdHist ??= []).push([m.iter, m.psnrHold]);
  // the first few hundred cycles rocket from near-nothing to the real curve —
  // charting that ramp squashes the whole run's y-scale into a corner
  if (chart && m.psnrTrain != null && m.iter >= Math.min(500, S.maxIters * 0.02)) {
    chart.push(m.iter, m.psnrTrain, m.psnrHold ?? null);
    chart.maxIter = S.maxIters;
    chart.events = S.chartEvents.map((e) => ({ ...e, at: e.iter / S.maxIters }));
    queueMetricsPaint();
  }
  // re-anchor the countdown on every fresh pace measurement
  if (m.itersPerSec > 0 && S.maxIters) {
    const est = performance.now() + Math.max(0, S.maxIters - m.iter) / m.itersPerSec * 1000;
    S.etaAt = S.etaAt == null ? est : S.etaAt + 0.35 * (est - S.etaAt);
  }
  if (S.state === 'train') {
    const el = $('t-iter');
    if (el) {
      el.textContent = pctOf(S.iter);
      $('t-splats').textContent = fmt(S.splats);
      $('t-eta').textContent = etaText();
      renderTrainingScore();
    }
  }
  // the details panels repaint on the same deferred, coalesced frame as the
  // chart — rebuilding them synchronously here is the same hitch
  if (!$('details').hidden && (S.detailTab === 'score' || S.detailTab === 'perf')) queueMetricsPaint();
  // Local records get a compact diagnostics update alongside the headline
  // progress every twenty seconds. The complete final snapshot is written on
  // pause, finish, or when leaving the model viewer.
  if (S.runId && (!S._runSaveT || performance.now() - S._runSaveT > 20000)) {
    S._runSaveT = performance.now();
    const runId = S.runId;
    const detailProgress = captureRunProgressDetails();
    import('./store.js').then(({ patchRun }) => patchRun(runId, {
      iter: m.iter, splats: m.splats, psnr: shownScore('train') ?? null,
      minutes: detailProgress.minutes, detailProgress,
    })).catch(() => {});
  }
}

/** Training progress as the visitor reads it: a percentage of the horizon. */
function pctOf(iter) {
  if (!S.maxIters) return '0%';
  return `${Math.min(100, Math.floor((iter || 0) / S.maxIters * 100))}%`;
}

/** Time remaining as a live 04:32-style countdown (digits only — the static
 *  "ETA" label lives in the template). Metrics re-anchor the estimate
 *  (blended, so a wobbling pace doesn't make the clock stutter); the main
 *  loop ticks the display while training runs. */
function etaText() {
  if (S.etaAt == null) return '—:—';
  const s = Math.max(0, (S.etaAt - performance.now()) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return `${h ? `${h}:` : ''}${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function onTrainEvent(e) {
  if (e.kind === 'refine' && e.grown > 0) {
    S.chartEvents.push({ iter: e.iter, kind: 'grow', label: `Capacity +${fmt(e.grown)}` });
    // shown right under the splat count in the dock, not as a HUD chip
    S.growNote = { text: `+${fmt(e.grown)} splats`, until: performance.now() + 2200 };
  }
  if (e.kind === 'refine' && e.grown === 0 && !S.growthStopped && e.iter > S.maxIters * 0.7) {
    S.growthStopped = true;
    S.chartEvents.push({ iter: e.iter, kind: 'stop', label: 'Growth stops' });
  }
  if (e.kind === 'overflow' && S._ovWarned !== (S.runId ?? 'live')) {
    S._ovWarned = S.runId ?? 'live';
    flash(`Tile-sort overflow — ${e.tiles} tiles dropped so far. Lower the splat budget for this device.`, 8000);
  }
  if (e.kind === 'soft-prune') {
    S.chartEvents.push({ iter: e.iter, kind: 'prune', label: `Pruned −${fmt(e.culled)}` });
    S.growNote = { text: `−${fmt(e.culled)} splats`, until: performance.now() + 2200 };
  }
  if (e.kind === 'dead-cull') {
    S.chartEvents.push({ iter: e.iter, kind: 'prune', label: `Dead −${fmt(e.culled)}` });
    S.growNote = { text: `−${fmt(e.culled)} dead splats`, until: performance.now() + 2200 };
  }
  if (e.kind === 'grow-now') {
    S.chartEvents.push({ iter: e.iter, kind: 'grow', label: `Grew +${fmt(e.grown || 0)}` });
    S.growNote = { text: `+${fmt(e.grown || 0)} splats`, until: performance.now() + 2200 };
  }
  if (e.kind === 'cap-hit') {
    if (S.state !== 'train') return;
    showCapCard(e);
  }
  if (e.kind === 'prune-continue') {
    // the cached exports go stale the moment the model reseeds (like trainMore)
    S.plyBlob = null; S.sogBlob = null;
    S.maxIters = e.maxIters;
    if (chart) chart.maxIter = S.maxIters;
    S.chartEvents.push({ iter: e.iter, kind: 'prune', label: `Pruned ${fmt(e.from)}→${fmt(e.to)}` });
    S.growNote = { text: `−${fmt(e.from - e.to)} splats`, until: performance.now() + 2200 };
  }
  if (e.kind === 'train-complete') finish();
  if (e.kind === 'device-lost') deviceLostRecovery();
  if (e.kind === 'worker-error') {
    // the worker realm faulted (canvas transfers died with it) — the same
    // reboot as a lost device, with its own message
    if (S.state !== 'train') return;
    deviceLostRecovery('The training worker crashed — restarting the run. Photos and the camera solve are kept.');
  }
  if (e.kind === 'train-error') {
    // the session already stopped the loop, so play resumes it — repair the
    // stale pause icon and say how to continue
    if (S.state !== 'train') return;
    const b = $('t-play');
    if (b) {
      b.dataset.state = 'play';
      b.innerHTML = '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M8 4.8v14.4l12.4-7.2z"/></svg>';
    }
    const f = $('t-finish');
    if (f) f.hidden = false;
    if (!notifSetPaused(true)) androidNotifStop(false);
    flash(`Training stopped: ${e.message || 'unknown error'} — press play to resume.`, 9000);
  }
  if (e.kind === 'view-error') {
    if (S.state !== 'train' && S.state !== 'done') return;
    remountModelCanvas(e.message);
  }
  if (e.kind === 'train-stall') {
    // non-fatal: the frame still owns the loop if it lands — but the user
    // must know the run is stuck instead of watching "Starting" forever
    if (S.state !== 'train') return;
    flash(`Training stalled: a frame has run ${Math.max(1, Math.round((e.frameMs || 60000) / 1000))}s+ without completing (iteration ${fmt(e.iter)}). The GPU may be overwhelmed — pause and try fewer/smaller photos.`, 12000);
    const b = trainBridge();
    if (b) {
      try {
        b.updateTrainingNotification(`Training ${S._notifName || 'model'}`,
          `Stalled at ${fmt(e.iter)} — GPU overwhelmed`, S.iter | 0, S.maxIters | 0);
      } catch { /* best-effort */ }
    }
  }
}

/** iOS (and crashing drivers) reclaim the WebGPU device from backgrounded
 *  tabs. The trained splats lived on it; photos + camera solve are CPU-side.
 *  Mid-training: rebuild and train again. Done: the .ply blob was cached at
 *  completion, so export and upload still work. */
async function deviceLostRecovery(msg) {
  if (S._recovering) return;
  const gen = S.gen;
  if (S.state === 'done') {
    flash(S.plyBlob || S.sogBlob
      ? 'The browser reclaimed the graphics device — the finished model is safe, export still works.'
      : S.runId
        ? 'The browser reclaimed the graphics device — the result is saved on this device under Yours.'
        : 'The browser reclaimed the graphics device.', 9000);
    return;
  }
  if (S.state !== 'train') return;
  S._recovering = true;
  document.getElementById('cv-model')?.remove();   // its context died with the device
  gpuCanvas = null;
  flash(msg || 'The system put the GPU to sleep while the tab was in the background — restarting training. Photos and the camera solve are kept.', 15000);
  try {
    // iOS won't hand out a new device while hidden — wait for the tab back
    if (document.visibilityState === 'hidden') {
      await new Promise((res) => {
        const h = () => {
          if (document.visibilityState !== 'visible') return;
          removeEventListener('visibilitychange', h);
          res();
        };
        addEventListener('visibilitychange', h);
      });
    }
    if (S.gen !== gen) return;
    // A file-backed service handback cannot use Session.recover(): Android's
    // low-memory worker owns (and has now lost) its transferred seed pixels.
    // Restart the visible WebView once, then reconstruct from state.bin.
    if (S._hlStaged) {
      const runId = S._hlStaged;
      const restarts = S._hlGpuRestarts | 0;
      const b = headlessBridge();
      if (b && typeof b.restartForGpuRecovery === 'function' && restarts < 1) {
        b.writeRunJson(runId, 'status.json', JSON.stringify({
          status: 'interrupted', stopped: true, owner: 'activity', iter: S.iter | 0,
          at: Date.now(), gpuRestarts: restarts + 1,
          error: 'the Activity GPU was reset while returning; reopening from the saved checkpoint',
        }));
        S._hlGpuRestarts = restarts + 1;
        b.restartForGpuRecovery(runId);
        return;
      }
      try { S.session && S.session.dispose(); } catch { /* already lost */ }
      S.session = null;
      showRecovery(runId, {
        status: 'interrupted', stopped: true, iter: S.iter | 0,
        error: 'the Activity graphics process could not reopen the saved checkpoint; no run data was deleted',
      });
      flash('The graphics process could not restart. The checkpoint and original AR frames are still stored; Continue here can retry.', 12000);
      return;
    }
    await S.session.recover();
    if (S.gen !== gen) return;
    S.iter = 0; S.psnrTrain = null; S.psnrHold = null;
    S.holdHist = []; S.chartEvents = [];  S.growthStopped = false;
    S.plyBlob = null; S.sogBlob = null;
    buildSceneFromSession();
    mountModelCanvas();
    quietView(S.session.view.resync && S.session.view.resync());
    startTraining();
  } catch (err) {
    console.error(err);
    if (S.gen === gen) {
      // A service handback is different from an ordinary tab loss: its exact
      // checkpoint and photos are still on disk. lowMem moved the continuation
      // seed into the failed worker, so Session.recover() cannot repair it in
      // place. Recreate Android's visible WebView once to get a genuinely new
      // renderer/GPU process, then boot again from the durable transaction.
      const runId = S._hlStaged;
      const restarts = S._hlGpuRestarts | 0;
      const b = headlessBridge();
      if (runId && b && typeof b.restartForGpuRecovery === 'function' && restarts < 1) {
        try {
          b.writeRunJson(runId, 'status.json', JSON.stringify({
            status: 'interrupted', stopped: true, owner: 'activity', iter: S.iter | 0,
            at: Date.now(), gpuRestarts: restarts + 1,
            error: 'the Activity GPU was reset while returning; reopening from the saved checkpoint',
          }));
          S._hlGpuRestarts = restarts + 1;
          b.restartForGpuRecovery(runId);
          return;
        } catch (restartErr) { console.error(restartErr); }
      }
      if (runId) {
        // Never let a failed handback fall through trainDeadEnd() to Home. The
        // marker, checkpoint, reconstruction and AR photos remain retryable.
        try { S.session && S.session.dispose(); } catch { /* already lost */ }
        S.session = null;
        showRecovery(runId, {
          status: 'interrupted', stopped: true, iter: S.iter | 0,
          error: 'the Activity graphics process could not reopen the saved checkpoint; no run data was deleted',
        });
        flash('The graphics process could not restart. The checkpoint and original AR frames are still stored; Continue here can retry.', 12000);
      } else {
        solveFailed('The graphics device was lost and could not be brought back — ' +
          'you are back at the setup screen. Your photos are kept; a lower Training resolution ' +
          '(the gear) or fewer photos usually train fine.');
        trainDeadEnd();
      }
    }
  } finally {
    S._recovering = false;
  }
}

// ── iOS background guard ────────────────────────────────────────────────────
// iOS purges WebGPU buffer contents from hidden tabs WITHOUT firing
// device-loss: the worker-tick loop then keeps training a zeroed/garbage
// model (seen in the wild: PSNR cliff mid-run, smeared splats, iter count
// intact). On iOS training PAUSES while hidden; on return a params probe
// decides between resuming and an honest restart. Desktop keeps its
// background-training behavior — the corruption is iOS-specific.
const IOS = /iP(hone|ad|od)/.test(navigator.userAgent) ||
  (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
if (IOS) document.addEventListener('visibilitychange', () => {
  if (!S.session || S.state !== 'train') return;
  if (document.visibilityState === 'hidden') {
    if (S.session.training) {
      S._bgPaused = true;
      S.session.pause();
    }
  } else if (S._bgPaused) {
    S._bgPaused = false;
    const gen = S.gen;
    (async () => {
      const tr = S.session && S.session.trainer;
      const ok = tr && tr.sanityProbe ? await tr.sanityProbe() : true;
      if (S.gen !== gen || S.state !== 'train') return;
      if (ok) {
        S.session.start();
        flash('Welcome back — training resumes.', 3500);
      } else {
        // the buffers didn't survive the background trip
        flash('iOS cleared the model while the tab was hidden — restarting the training run.', 9000);
        deviceLostRecovery();
      }
    })().catch(() => {});
  }
});

// Desktop counterpart: training keeps running while hidden (worker tick), but
// a hidden tab is also where runs die unseen — crash, GPU reset, closed
// window. Snapshot the state when the user switches away; exportRawState is
// serialized against the step loop, so the readback is a consistent frame.
// (iOS stays pause-only above — reading buffers that may already be purged
// could overwrite a good checkpoint with garbage.)
if (!IOS) document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden' && S.state === 'train') {
    checkpointRun('hidden').catch(() => {});
  }
});

/** The done-state viewer: land at the photographer's view, controls, tour.
 *  Shared by a run finishing live and a finished run reopened from its
 *  state checkpoint (viewOnly continueLocalRun). */
function enterDone() {
  S.state = 'done';
  syncTrainingVisualsControl();
  S.atFrame = -1; S.fadeTo = 0;
  vp.lock = null; vp.freeF = null;
  $('stage').dataset.cursor = 'grab';
  // land roughly where the first photograph was taken — the photographer's
  // view of the result, not an abstract overview
  const first = S.scene.cams.find((c) => c.R && c.state !== 'holdout') || S.scene.cams[0];
  if (first && first.R) {
    vp.syncTo(first);
    vp.dist *= 1.15;   // stepped back just enough for context
  } else {
    vp.frameScene();
  }
  renderControls();
  dock('');
  startTour();
  if (S._detailsFromUrl) { S._detailsFromUrl = false; openDetails(); }   // ?details=<tab> reopens the sheet
}

/** Avatar mode: the trained, isolated splat is the halfway point — joints,
 *  face, body model, binding and the account come after, on their own card
 *  (app/avatar/runner.js). The app only mounts the card and persists the
 *  manifest the stages write. */
/** Point the viewer at a model an avatar stage handed back. The cut session
 *  keeps the SAME world and the same camera poses (cut.js reuses src.recon),
 *  so the pose, the intro flight and the frustums all still hold — only the
 *  room's splats are gone. The source session stays resident and untouched;
 *  nothing else in the app reads it once this returns.
 *
 *  Everything downstream of the viewer follows too, which is the point: the
 *  splat count on the chip, an export, and a share of this run are the
 *  isolated person now. */
function adoptSession(ses, stage = '', res = null) {
  if (!ses || !ses.trainer || ses === S.session) return;
  S.session = ses;
  S.splats = ses.trainer.n;
  S.plyBlob = null; S.sogBlob = null; S._sogJob = null;   // the cached exports were the old model
  S._viewKey = '';                      // re-render at the same pose
  if (gpuCanvas && ses.view) {
    try { quietView(ses.view.attach(gpuCanvas)); } catch (e) { console.warn('view attach', e); }
  }
  // Isolate hands back the person's own extent (cut.js bounds). The intro
  // flight was orbiting at arm's length inside a room that no longer exists —
  // it is stopped and the whole figure framed instead.
  if (res && res.bounds && res.bounds.radius > 0 && S.scene) {
    stopTour();
    S.scene.center = res.bounds.center;
    S.scene.radius = res.bounds.radius;
    vp.setScene(S.scene);   // frames it: target = the figure, distance 2.5x its radius
    // a person is not a scene: eye level rather than looking down from above,
    // and a little further back so the head clears the card at the top
    vp.dist = res.bounds.radius * 3.2;
    vp.pitch = -0.12;
    vp.dirty = true;
  }
  renderControls();
  console.log(`[avatar] the viewer now shows the ${stage || 'new'} model — ${ses.trainer.n.toLocaleString()} splats`);
}

// ── the hand-off ──────────────────────────────────────────────────────────
// ?handoff=1: this tool was opened by another one (the splat rigger) to MAKE an
// avatar, nothing else. The finished package is posted back to whoever opened
// it and that is the end of the job here — no inspection, no animation, no
// account (the user, 2026-09-18: "the ui is just for the creation, the
// inspection is then done in splat rigger, including the animation attachment").
//
// The message vocabulary is the rigger's own embed protocol, so a host that
// already listens to the rigger needs one more `source` and nothing else:
//   { source: 'splat-js', type: 'splat-asset', assetType, name, buffer }
//   { source: 'splat-js', type: 'splat-done',  name }
//   { source: 'splat-js', type: 'splat-error', message }
// Same-origin only, and the buffers are transferred, not copied.
const HANDOFF = new URLSearchParams(location.search).get('handoff') === '1' && window.parent !== window;

function postToHost(msg, transfer) {
  window.parent.postMessage({ source: 'splat-js', ...msg }, location.origin, transfer || []);
}

/** hand the finished avatar to whoever opened this tool */
async function handOff({ name, assets }) {
  for (const a of assets) postToHost({ type: 'splat-asset', ...a }, [a.buffer]);
  postToHost({ type: 'splat-done', name });
  flash('Avatar sent back — finish it in the rigger', 8000);
}

// ── the walk preview ──────────────────────────────────────────────────
// A bound avatar can be watched walking before it is sent anywhere: the clip is
// the rigger's own walking_anim.glb and the skinning is app/avatar/walk.js. The
// trainer's parameter buffer is overwritten with the posed splats each frame,
// so the view (which re-sorts every render) just draws it.
async function startWalk({ binding, fit, rigGlb, session }) {
  try {
    stopWalk();
    const { createWalker } = await import('../avatar/walk.js');
    S.walker = await createWalker({ session, binding, fit, rigGlb, log: (m) => console.log('[avatar]', m) });
    S.walking = true;
    renderControls();
    console.log(`[avatar] walking: ${S.walker.clip}, ${S.walker.duration.toFixed(2)}s, ${S.walker.splats.toLocaleString()} bound splats`);
  } catch (e) {
    console.warn('[avatar] no walk preview: ' + (e.message || e));
    S.walker = null; S.walking = false;
  }
}

/** Stop the preview and put the trained pose back. Anything that READS the
 *  model (compress, export, share, a thumbnail) must call this first, or it
 *  captures whatever half-step the walk was on. */
function stopWalk() {
  if (!S.walker) return;
  if (S.walking) { try { S.walker.rest(); } catch (e) { /* the device may be gone */ } }
  S.walking = false;
  S._viewKey = '';
}

async function runAvatarStages() {
  if (!S.avatar || !S.session) return;
  const gen = S.gen;
  try {
    const av = await import('../avatar/index.js');
    await av.afterTraining({
      session: S.session, frames: S.preset.files || S.loadedFiles, manifest: S.avatar,
      capture: S._capRec,
      mount: (card) => $('stage').appendChild(card),
      // the stage sequence, its message and its meter live in the DOCK — the
      // top bar the camera solve used. A card over the stage covered the face
      // it was asking about (the user, 2026-09-18).
      mountDock: (el) => { const d = $('dock'); d.className = 'dock dock-av'; d.innerHTML = ''; d.appendChild(el); },
      log: (m) => console.log('[avatar]', m), flash,
      // a stage replaced the model (Isolate cuts the person out of the room,
      // the face pass sharpens it): the viewer follows the model that will be
      // shipped, so the end of the run shows the AVATAR, not the scene it came
      // out of (the user, 2026-09-18)
      onSession: (ses, stage, res) => { if (S.gen === gen) adoptSession(ses, stage, res); },
      // the binding is what makes the avatar movable: show it walking — but not
      // on the embedded route, where the rigger does the animating
      ...(HANDOFF ? {} : { onBound: (o) => { if (S.gen === gen) startWalk({ ...o, session: S.session }); } }),
      // ?handoff=1: the package goes back to the opener instead of onto the account
      ...(HANDOFF ? { handoff: handOff } : {}),
      persist: (m) => { if (S._capRec && S.gen === gen) { S._capRec.avatar = m; persistCapture(S._capRec); } },
      // the thumbnail is of the avatar as it ships — the trained pose, not
      // whatever half-stride the walk preview happens to be on
      thumb: async () => { const was = S.walking; stopWalk(); const t = await renderShareThumb(); S.walking = was; return t; },
    });
  } catch (e) {
    console.error(e);
    flash(`Avatar: ${e.message || e}`, 8000);
    if (HANDOFF) postToHost({ type: 'splat-error', message: String(e.message || e) });
  }
}

async function finish() {
  androidNotifStop(true);   // the shade keeps the final score
  S.iter = S.session.trainer.iter;   // honest count — the run may end early
  S.minutes = Math.max(1, Math.round((performance.now() - S.trainT0) / 60000));
  enterDone();
  if (S.avatar) runAvatarStages();
  const hold = S.psnrHold != null ? ` · ${S.psnrHold.toFixed(1)} dB on the photograph it never saw` : '';
  flash(`Done${hold}`, 6000);
  if (EVAL.on) {
    // the ?eval benchmark verdict: mean PSNR over every held-out photo
    S.session.evalTestPsnr().then((r) => {
      if (!r || S.state !== 'done') return;
      S.psnrTest = r;
      flash(`Test PSNR ${r.psnr.toFixed(2)} dB over ${r.frames.length} held-out photos`, 12000);
    }).catch(() => {});
  }
  S.plyBlob = null; S.sogBlob = null; S._sogJob = null;
  // The result is secured as a RAW STATE checkpoint (a plain readback,
  // seconds) — the same record a pause writes, marked finished. It is
  // viewable and resumable from Yours without anything else. Nothing heavy
  // runs here: the .ply export (an O(splats × cameras) bake on the main
  // thread), the SOG compression (k-means on a second GPU device) and the
  // per-photo scoring used to start right now and left the finished viewer
  // at ~1 fps for minutes — exactly while the creator wanted to look at the
  // result. They are on demand: Compress (or Share / Download .sog) runs the
  // ONE compression job (getSogBlob), scoring runs when Details opens.
  const gen = S.gen;
  // Securing is async (GPU readback + store writes) while the done screen
  // is already live: the Train button below awaits this before offering
  // the finished-run path, so an immediate tap still gets the config card
  // and the history instead of a blind live resume.
  S._securing = (async () => {
    while (S._ckptBusy) await new Promise((r) => setTimeout(r, 200));   // a 'hidden' write in flight
    let ckpt = null;
    try { ckpt = await checkpointRun('finish'); } catch (e) { console.warn('finish checkpoint failed', e); }
    if (!ckpt) {
      flash('The finished model is still open, but it could not be saved. Free storage before leaving.', 12000);
      return false;
    }
    // The final checkpoint is now in Android files and/or IndexedDB; only then
    // release the live marker so Home can rediscover the finished model.
    headlessClear();
    if (S.gen !== gen) return true;
    if (S.runId && S.gen === gen) {
      try {
        const thumb = await renderShareThumb();
        if (S.gen !== gen) return;
        if (thumb) await writeRunBlob(S.runId, 'thumb.webp', thumb).catch(() => false);
        const { patchRun } = await import('./store.js');
        await patchRun(S.runId, {
          status: 'finished', iter: S.iter, splats: S.splats,
          // app runs train every photo (holdout -1): train PSNR is the number
          psnr: S.psnrHold ?? S.psnrTrain ?? null, minutes: S.minutes,
          detailSolve: captureRunSolveDetails(),
          detailProgress: captureRunProgressDetails(),
          detailGpu: gpuFacts(false),
          ...(thumb ? { thumb } : {}),
        });
        flash('Result saved on this device — it stays under Yours', 5000);
      } catch (e) { console.warn('local save failed', e); }
    }
    // adopted LAST, after every store write: the done screen's Train then
    // follows the same finished-run path as Yours (config card, history
    // restore, rebuild-continue) instead of the blind live resume — own
    // photos only, whose pixels stay reachable in the capture store.
    // (Adopting before the tail would let an immediate tap rebuild while
    // the finish patch is still in flight, landing 'finished' over live
    // training; adopted-after-tail, S._localRun existing means secured.)
    const pid = String((S.preset && S.preset.id) || '');
    if (ckpt && S.runId && S.gen === gen && (pid === '__own' || pid === '__last')) {
      S._localRun = {
        id: S.runId, name: localRunName(S.preset),
        status: 'finished', createdAt: Date.now(), updatedAt: Date.now(),
        iter: S.iter, maxIters: S.maxIters, splats: S.splats || 0,
        psnr: S.psnrHold ?? S.psnrTrain ?? null, minutes: S.minutes,
        frames: (S.photos || []).length, ownSrc: true,
        state: ckpt.state, recon: ckpt.recon,
        detailSolve: captureRunSolveDetails(),
        detailProgress: captureRunProgressDetails(),
        detailGpu: gpuFacts(false),
      };
      if (S.state === 'done') renderControls();   // swap the quick Train for the run's own
    }
    return true;
  })();
  try { await S._securing; } catch { /* best-effort */ }
  if (PERF.on) perfCard();
}

async function hydrateLocalRunPhotos(reconJson, run) {
  const source = reconJson && reconJson.source;
  if (!run || !run.ownSrc || !source || !Array.isArray(source.names) || !source.names.length) return;
  if (Array.isArray(source.urls) && source.urls.length === source.names.length && source.urls.every(Boolean)) return;
  try {
    const { loadLastCapture } = await import('./store.js');
    const capture = await loadLastCapture();
    const byName = new Map(((capture && capture.files) || []).map((f) => [f.name, f]));
    const urls = source.names.map((name) => {
      const file = byName.get(name);
      return file && file.blob ? URL.createObjectURL(file.blob) : null;
    });
    if (urls.every(Boolean)) {
      if (S.ownUrls) S.ownUrls.forEach((url) => URL.revokeObjectURL(url));
      S.ownUrls = urls;
      source.urls = urls;
    } else {
      urls.filter(Boolean).forEach((url) => URL.revokeObjectURL(url));
    }
  } catch { /* source imagery is optional; the saved diagnostics still render */ }
}

// ── restore: present a saved run without re-training ────────────────────────
// ?model=<url> (a session .zip, .ply or .sog; &recon=<url> adds the camera
// path to bare model files) or a file dropped on the start card. The model
// lands in the exact done-state viewer: tour, orbit, exports — no training.
async function restoreSession(src) {
  try {
    S._localRun = src.localRun || null; // set only by the local runs library
    S.runId = S._localRun ? S._localRun.id : null;
    S._resumeRunId = null;
    S.feats = new Map(); S.lastPairEv = null;
    S.solveStats = { pairsChecked: 0, pairsUsable: 0, solveSec: 0 };
    S.chartEvents = []; S.holdHist = [];  S.perfMetrics = []; S.itersPerSec = 0;
    S._savedGpuFacts = null; S._savedDetailSettings = null;
    // a run that was already shared knows its space: the viewer offers its
    // link and its listing, where a reload used to leave only a Download
    if (SHELL !== 'browser' && S._localRun && S._localRun.spaceId) S.share = { id: S._localRun.spaceId, title: S._localRun.name || '' };
    S._viewerOpen = true;
    // the viewer is a navigable state: Back returns to the wall, never to
    // whatever page happened to precede the app
    if (!(history.state && history.state.sj)) history.pushState({ sj: 'viewer' }, '');
    closeOverlays();
    $('start').hidden = true;
    $('detail').hidden = true;
    flash('Loading the model …', 120000);
    // SOG + recon: the lite viewer — the engine renders the compressed splat
    // directly (sorted, WebGL2), nothing decodes to float arrays, and phones
    // without WebGPU can still view. ?nopc forces the trainer view.
    const liteOk = !src.bytes && src.url && /\.sog($|\?)/.test(src.url) && src.reconUrl &&
      !new URLSearchParams(location.search).has('nopc');
    if (liteOk) {
      try {
        const r = await fetch(src.reconUrl);
        if (!r.ok) throw new Error(`recon fetch failed (${r.status})`);
        const buf = new Uint8Array(await r.arrayBuffer());
        let reconJson;
        if (buf[0] === 0x50 && buf[1] === 0x4b) {
          const { unzipStoreAsync } = await import('./session_io.js');
          reconJson = JSON.parse(new TextDecoder().decode((await unzipStoreAsync(buf)).get('recon.json')));
        } else {
          reconJson = JSON.parse(new TextDecoder().decode(buf));
        }
        await hydrateLocalRunPhotos(reconJson, S._localRun);
        const { createSogView } = await import('./pcview.js');
        const ses = await createSogView(src.url, { radius: reconJson.sceneRadius || 10, yUp: !!reconJson.knownCameras });
        ses.frames = (reconJson.frames || []).map((f) => ({ ...f }));
        // the Details sheet reads session.recon (cams/points) — same shape
        // the old view-only path built via useReconstruction()
        ses.recon = {
          known: !!reconJson.knownCameras,
          cams: (reconJson.cams || []).map((c) => ({ imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f, cx: c.cx, cy: c.cy })),
          // the Details header counts these; the recon carries a decimated
          // cloud, so at least report its size instead of a flat zero
          points: { length: Math.floor(((reconJson.cloud && reconJson.cloud.xyz) || []).length / 3) },
          k1: reconJson.k1 || 0, k2: reconJson.k2 || 0,
        };
        finishRestore(ses, reconJson, reconJson.splats || 0, false, null);
        return;
      } catch (e) {
        console.warn('sog viewer failed — falling back to the trainer view', e);
      }
    }
    const { decodeModel } = await import('./session_io.js');
    const got = src.bytes
      ? await decodeModel(src.bytes, src.reconUrl)
      : await fetchModel(src.url, src.reconUrl);
    const { gaussians, reconJson, state } = got;
    await hydrateLocalRunPhotos(reconJson, S._localRun);
    const iter = (state && state.iter) || (reconJson && reconJson.iter) || 0;
    const ses = createSession({ holdout: -1, maxIters: Math.max(1, iter), trainWorker: trainWorkerOpt() });
    if (reconJson) {
      ses.useReconstruction({
        known: !!reconJson.knownCameras,
        cams: reconJson.cams.map((c) => ({ imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f, cx: c.cx, cy: c.cy })),
        points: [],
        k1: reconJson.k1 || 0, k2: reconJson.k2 || 0,
        ...(reconJson.fFeat ? { fFeat: reconJson.fFeat } : {}),
      });
      ses.useFrames(reconJson.frames.map((f) => ({ ...f, sampleColor: () => [0.5, 0.5, 0.5] })));
    }
    // ?shup=x,y,z: the model was trained with horizontal-only SH about this up axis
    // (avatar runs); the viewer evaluates its SH the same way, else the bands that
    // depend on the vertical view angle show values they were never trained for
    const shupQ = (new URLSearchParams(location.search).get('shup') || '').split(',').map(Number);
    const shUp = shupQ.length === 3 && shupQ.every((v) => Number.isFinite(v)) ? shupQ : null;
    await ses.seedFrom(gaussians, {
      viewOnly: true,
      sceneRadius: reconJson ? reconJson.sceneRadius : undefined,
      iter,
      trainer: { maxSplats: gaussians.n, capMult: 1, ...(shUp ? { shUp } : {}) },
    });
    finishRestore(ses, reconJson, gaussians.n, !!state, gaussians);
  } catch (e) {
    console.error(e);
    loadFailed('That model didn\'t load', e.message);
  }
}

/** The shared back half of a restore: viewer state, scene, strip, tour —
 *  works for a real (view-only) session and for the SOG-lite stand-in. */
function finishRestore(ses, reconJson, nSplats, hasState, gaussians) {
  try {
    const iter = (reconJson && reconJson.iter) || 0;
    S.session = ses;
    const stats = (reconJson && reconJson.stats) || {};
    S.restored = { hasState, stats, reconJson };
    S.preset = { id: '__restored', name: (reconJson && reconJson.name) || 'shared splat' };
    S.splats = nSplats;
    S.maxIters = iter || 1;
    S.iter = iter;
    S.minutes = stats.minutes || 0;
    S.psnrTrain = stats.psnrTrain ?? null;
    S.psnrHold = stats.psnrHold ?? null;
    S.plyBlob = null; S.sogBlob = null;
    S.lodPlan = null;
    S.state = 'done';
    restoreRunDetails(S._localRun, ses);
    syncTrainingVisualsControl();
    const frames = (reconJson && reconJson.frames) || [];
    const source = (reconJson && reconJson.source) || {};
    // the photographs come back too when the session recorded where they
    // live (photo sets: one image per frame) — the strip and the per-photo
    // compare modes then work exactly like after a live run
    const photosBack = Array.isArray(source.urls) && source.urls.length &&
      source.urls.length === frames.length && source.urls.every(Boolean);
    S.isPanoSet = false;
    let cams;
    if (photosBack) {
      S.photos = source.names.map((n, i) => ({ url: source.urls[i], name: n }));
      // camMeta for lookThrough/error-render: train-res intrinsics per
      // registered camera (offset unused outside training)
      const metas = reconJson.cams.map((c) => {
        const fr = frames[c.imgIdx] || {};
        const s2 = (fr.tw || fr.fw || 1000) / (fr.fw || 1000);
        return { imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f * s2,
                 cx: (fr.tw || 1000) / 2, cy: (fr.th || 1000) / 2,
                 w: fr.tw || 1000, h: fr.th || 1000, offset: 0 };
      });
      if (ses.trainer) ses.trainer.camMeta = metas;
      const byImg = new Map(reconJson.cams.map((c, k) => [c.imgIdx, k]));
      cams = S.photos.map((p, i) => {
        const k = byImg.get(i);
        const fr = frames[i] || {};
        const w = fr.fw || 1000, h = fr.fh || 1000;
        if (k == null) return { i, ci: -1, R: null, t: null, url: p.url, name: p.name,
                                w, h, cx: w / 2, cy: h / 2, state: 'unplaced', feats: 0, psnr: null };
        const c = reconJson.cams[k];
        return { i, ci: k, R: c.R, t: c.t, f: c.f, w, h, cx: w / 2, cy: h / 2,
                 url: p.url, name: p.name, state: 'placed', feats: 0, psnr: null };
      });
    } else {
      // pano rigs: frames are sliced faces (name_fN) of the source
      // panoramas — the strip shows ONE card per pano, and clicking flies
      // to that station (no photo overlay: equirect vs pinhole never match)
      const panoish = Array.isArray(source.urls) && source.urls.length &&
        frames.length > source.names.length && source.urls.every(Boolean) &&
        /_f\d+$/.test(String((frames[0] || {}).name || ''));
      S.isPanoSet = !!panoish;
      if (panoish && reconJson) {
        const baseName = (n) => String(n).replace(/\.[^.]+$/, '');
        const idxOf = new Map(source.names.map((n, i) => [baseName(n), i]));
        const best = new Array(source.names.length).fill(null);   // {k, face}
        reconJson.cams.forEach((c, k) => {
          const m = String((frames[c.imgIdx] || {}).name || '').match(/^(.*)_f(\d+)$/);
          if (!m) return;
          const si = idxOf.get(m[1]);
          if (si == null) return;
          if (!best[si] || +m[2] < best[si].face) best[si] = { k, face: +m[2] };
        });
        S.photos = source.names.map((n, i) => ({ url: source.urls[i], name: n }));
        cams = S.photos.map((p, i) => {
          const b = best[i];
          if (!b) return { i, ci: -1, R: null, t: null, url: p.url, name: p.name,
                           w: 4, h: 3, cx: 2, cy: 1.5, state: 'unplaced', feats: 0, psnr: null };
          const c = reconJson.cams[b.k];
          const fr = frames[c.imgIdx] || {};
          const w = fr.fw || 1000, h = fr.fh || 1000;
          return { i, ci: b.k, R: c.R, t: c.t, f: c.f, w, h, cx: w / 2, cy: h / 2,
                   url: p.url, name: p.name, state: 'placed', feats: 0, psnr: null, pano: true };
        });
      } else {
        S.photos = [];      // bare models: no photographs at all
        cams = (reconJson ? reconJson.cams : []).map((c, ci) => {
          const fr = frames[c.imgIdx] || {};
          const w = fr.fw || 1000, h = fr.fh || 1000;
          return { i: ci, ci, R: c.R, t: c.t, f: c.f, w, h, cx: w / 2, cy: h / 2,
                   state: 'placed', feats: 0, name: c.name };
        });
      }
    }
    // viewer mode: the strip is visible UI — thumbs lazy-load for the tiles
    // in view (deferral is for the card flows, where the strip is covered).
    // A share may carry a packed thumbnail zip: one small fetch feeds the
    // whole strip instead of N full training photographs.
    S._thumbsUrl = (source && source.thumbs) || null;
    S._thumbPack = null;
    buildStrip();
    const cl = (reconJson && reconJson.cloud) || { xyz: [], rgb: [] };
    let center = reconJson && reconJson.center;
    let radius = (reconJson && reconJson.sceneRadius) || ses.model.radius;
    if (!center && gaussians) {
      // bare model files carry no scene metadata — read it off the splats
      const d = gaussians.data, step = Math.max(1, Math.floor(gaussians.n / 20000)) * 16;
      let sx = 0, sy = 0, sz = 0, m = 0;
      for (let o = 0; o < gaussians.n * 16; o += step) { sx += d[o]; sy += d[o + 1]; sz += d[o + 2]; m++; }
      center = [sx / m, sy / m, sz / m];
      // median splat distance ~= the subject's scale; the max is always some
      // far background shell and would frame the camera way outside it
      const dists = [];
      for (let o = 0; o < gaussians.n * 16; o += step) {
        dists.push((d[o] - center[0]) ** 2 + (d[o + 1] - center[1]) ** 2 + (d[o + 2] - center[2]) ** 2);
      }
      dists.sort((a, b) => a - b);
      radius = Math.sqrt(dists[dists.length >> 1]) * 2;
    } else if (!center) {
      center = [0, 0, 0];
    }
    cams.forEach((cam) => {
      const imgIdx = cam.ci >= 0 && reconJson && reconJson.cams[cam.ci]
        ? reconJson.cams[cam.ci].imgIdx : cam.i;
      cam.feats = (S.feats.get(imgIdx) || {}).n || 0;
    });
    S.scene = {
      cams,
      xyz: Float32Array.from(cl.xyz), rgb: Uint8Array.from(cl.rgb),
      center, radius,
    };
    vp.setScene(S.scene);
    // known-camera restores level to the capture world's gravity (persisted
    // per run); solver restores measure up from the cameras as before
    if (reconJson && reconJson.knownCameras) vp.setUp(gravityUpForKnownRun({ persisted: reconJson.levelingUp }));
    else if (cams.length) vp.detectUp(cams);
    mountModelCanvas();
    S.atFrame = -1; S.fadeTo = 0;
    vp.lock = null; vp.freeF = null;
    const first = cams.find((c) => c.R);
    if (first) { vp.syncTo(first); vp.dist *= 1.15; } else vp.frameScene();
    // first person: the set's flag (stamp or recon) — and the DEFAULT for
    // 360 captures, which are made from inside the scene by construction
    vp.fpv = !!(S.fpvSet || (reconJson && reconJson.fpv) || S.isPanoSet);
    if (vp.fpv) vp.dist = Math.max(0.3, ((S.scene && S.scene.radius) || 10) * 0.1);
    $('stage').dataset.cursor = 'grab';
    $('btn-new').hidden = false;   // the way back to the front page
    renderControls();
    dock('');
    // ?frame=<i>(&cmp=): the link was shared from a frame focus — land
    // exactly there instead of flying the intro
    const fq = new URLSearchParams(location.search);
    const fi = parseInt(fq.get('frame'), 10);
    if (Number.isFinite(fi) && S.photos.length && fi >= 0 && fi < S.photos.length) {
      const cmp = fq.get('cmp');
      if (['swipe', 'loupe', 'error'].includes(cmp)) S.compare = cmp;
      select(fi);
    } else if (cams.length > 2) {
      startTour();
    }
    if (S._detailsFromUrl) { S._detailsFromUrl = false; openDetails(); }   // ?details=<tab> reopens the sheet
    // the stand-in hero image fades once the model actually renders — for
    // the SOG-lite viewer that is when the asset finishes streaming
    const hero = document.getElementById('share-hero');
    if (hero) {
      Promise.resolve(ses._boot).then(() => {
        hero.classList.add('gone');
        setTimeout(() => hero.remove(), 700);
      });
    }
    flash(`Loaded ${fmt(nSplats)} splats — ${S.preset.name}.`, 5000);
  } catch (e) {
    console.error(e);
    loadFailed('That model couldn\'t be shown', e.message);
  }
}

// ── ?perf: the timing log as a downloadable text file ───────────────────────
function pctl(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

function buildPerfReport() {
  const ses = S.session;
  const gi = (ses.gpu && ses.gpu.info) || S._savedGpuFacts || {};
  const rows = (ses.perf && ses.perf.frames) || [];
  const marks = (ses.perf && ses.perf.marks) || [];
  const fr0 = (ses.frames && ses.frames[0]) || {};
  const settings = S._savedDetailSettings || S.settings;
  const tileGrad = ses.trainer ? ses.trainer.tileGrad
    : (S._localRun && S._localRun.detailProgress && S._localRun.detailProgress.tileGrad) ?? '?';
  const L = [];
  L.push(`splat.js perf log — ${new Date().toISOString()}`);
  L.push(`url: ${location.href}`);
  L.push(`ua: ${(S._savedGpuFacts && S._savedGpuFacts.ua) || navigator.userAgent}`);
  L.push(`gpu: ${[gi.vendor, gi.architecture, gi.device].filter(Boolean).join(' ') || 'unknown'}`);
  L.push(`screen: ${(S._savedGpuFacts && S._savedGpuFacts.screen) || `${screen.width}x${screen.height} @dpr ${devicePixelRatio}`}`);
  L.push(`photos: ${S.photos.length || (ses.frames && ses.frames.length) || 0} · placed ${ses.recon ? ses.recon.cams.length : '?'} · training res ${fr0.tw || '?'}x${fr0.th || '?'}`);
  L.push(`settings: ${JSON.stringify(settings)} · preset ${S.preset ? S.preset.id : '?'}`);
  L.push(`splats: ${fmt(S.splats)} · holdout psnr ${S.psnrHold != null ? S.psnrHold.toFixed(2) : '—'} dB`);
  L.push(`tileGrad: ${tileGrad} · maxIters ${S.maxIters}`);
  if (rows.length > 1) {
    const t0 = rows[0][0], t1 = rows[rows.length - 1][0];
    const iters = rows[rows.length - 1][1] - rows[0][1];
    L.push(`wall: ${((t1 - t0) / 1000).toFixed(1)}s for ${iters} iters -> ${(iters / Math.max(.001, (t1 - t0) / 1000)).toFixed(1)} it/s`);
    const col = (i) => rows.map((r) => r[i]);
    L.push(`per frame (batch med ${pctl(col(2), .5)}):`);
    const stat = (name, i) =>
      L.push(`  ${name} med ${pctl(col(i), .5).toFixed(1)}ms  p90 ${pctl(col(i), .9).toFixed(1)}ms  max ${Math.max(...col(i)).toFixed(1)}ms`);
    stat('encode ', 4);
    stat('view   ', 5);
    stat('fence  ', 6);
    stat('metrics', 7);
    stat('total  ', 8);
  }
  L.push('', 'frames: t_ms iter batch splats enc view fence met total');
  for (const r of rows) L.push('  ' + r.join(' '));
  L.push('', 'refines: t_ms iter ms moved grown');
  for (const m of marks) L.push(`  ${m.t} ${m.iter} ${m.ms} ${m.moved} ${m.grown}`);
  L.push('', 'metrics: t_ms iter it/s psnrTrain psnrHold');
  for (const m of S.perfMetrics || []) L.push('  ' + m.join(' '));
  return L.join('\n');
}

/** clipboard copy with feedback on the button itself — a flash would be
 *  hidden behind the details sheet */
async function copyPerfLog(btn) {
  const old = btn.textContent;
  try {
    await navigator.clipboard.writeText(buildPerfReport());
    btn.textContent = 'Copied ✓';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = old; }, 1600);
}

function downloadPerfLog() {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([buildPerfReport()], { type: 'text/plain' }));
  a.download = `splatjs_perf_${new Date().toISOString().replace(/\W/g, '').slice(0, 15)}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function perfCard() {
  document.getElementById('perfcard')?.remove();
  const txt = buildPerfReport();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'perfcard';
  card.innerHTML = `
    <b>Perf run complete</b>
    <pre class="perfpre">${txt.split('\nframes:')[0].replace(/</g, '&lt;')}</pre>
    <div class="upcard-row">
      <button class="btn btn-quiet" id="perf-close">Close</button>
      <button class="btn btn-quiet" id="perf-copy">Copy</button>
      <button class="btn btn-accent" id="perf-dl">Download log</button>
    </div>`;
  $('stage').appendChild(card);
  card.querySelector('#perf-close').addEventListener('click', () => card.remove());
  card.querySelector('#perf-copy').addEventListener('click', (e) => copyPerfLog(e.currentTarget));
  card.querySelector('#perf-dl').addEventListener('click', downloadPerfLog);
}

/** after the run: an honest per-photograph score, filled in the background */
async function scoreFrames() {
  const gen = S.gen;
  for (const c of S.scene.cams) {
    if (c.ci < 0 || S.gen !== gen || S.state !== 'done') return;
    try {
      c.psnr = await S.session.evalFramePsnr(c.ci);
      paintStrip();
      if (!$('details').hidden) renderDetails();
    } catch { return; }
  }
}

// ── stage controls (train/done) ─────────────────────────────────────────────
function seg(items, active, onPick) {
  const d = document.createElement('div');
  d.className = 'seg';
  items.forEach(([val, label]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(val === active));
    // pointerdown, not click: the mode switches the instant the finger lands.
    // click stays for keyboard activation (Enter/Space fire click, never
    // pointerdown) — the follow-up click after a real press re-picks the same
    // value, which onPick implementations must treat as a no-op.
    b.addEventListener('pointerdown', () => onPick(val));
    b.addEventListener('click', () => onPick(val));
    d.appendChild(b);
  });
  return d;
}

const cursorFor = (m) => (m === 'loupe' ? 'loupe' : m === 'swipe' ? 'swipe' : 'grab');

function renderControls() {
  const c = $('controls');
  c.innerHTML = '';
  const live = S.state === 'train' || S.state === 'done';
  c.hidden = !live;
  if (!live) return;

  if (S.atFrame >= 0) {
    c.appendChild(seg([['swipe', 'Swipe'], ['loupe', 'Loupe'], ['error', 'Error']],
      S.compare, (v) => {
        if (S.compare === v) return;   // swallow the click that follows pointerdown
        S.compare = v;
        $('stage').dataset.cursor = cursorFor(v);
        renderControls();
        syncFrameUrl();
      }));
  }

  if (S.state !== 'done') return;

  const play = document.createElement('button');
  play.className = 'iconbtn';
  play.id = 'c-play';
  play.title = 'Fly the capture path';
  play.setAttribute('aria-label', 'Fly the capture path');
  play.innerHTML = '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M8.5 5.5v13l10-6.5z"/></svg>';
  play.addEventListener('click', () => {
    if (S.tour) { stopTour(); return; }  // playing -> the same button stops
    if (S.atFrame >= 0) leaveFrame();    // play works from the compare modes too
    startTour(true);
  });
  c.appendChild(play);

  if (!S.restored && !S.sogBlob && S.session && S.session.trainer && S.session.trainer.device) {
    // the web-delivery format is optional and costs real GPU time (k-means
    // on a second device) — the creator starts it, never the finish
    const sog = document.createElement('button');
    sog.className = 'cbtn';
    sog.id = 'c-sog';
    sog.textContent = 'Compress';
    sog.title = 'Compress to .sog — 10-20x smaller, for sharing and the web';
    sog.addEventListener('click', async () => {
      try {
        const blob = await getSogBlob();
        flash(`${fmt(S.splats)} splats compressed to ${(blob.size / 1e6).toFixed(1)} MB — Share and Download .sog are instant now.`, 5000);
      } catch (e) {
        console.error(e);
        flash(`SOG compression failed: ${e.message}`, 6000);
      }
    });
    c.appendChild(sog);
  }

  if (S.walker) {
    const w = document.createElement('button');
    w.className = 'cbtn' + (S.walking ? ' btn-accent' : '');
    w.id = 'c-walk';
    w.textContent = S.walking ? 'Walking' : 'Walk';   // the state, not the click
    w.title = S.walking
      ? 'Stop the walk and put the trained pose back'
      : 'Play the walk cycle on the rigged avatar';
    w.addEventListener('click', () => {
      if (S.walking) { stopWalk(); } else { S.walking = true; }
      renderControls();
    });
    c.appendChild(w);
  }

  const stats = document.createElement('button');
  stats.className = 'statchip';
  stats.innerHTML = `<span><b>${fmt(S.splats)}</b> splats</span>` +
    (S.psnrHold != null ? `<span><b>${S.psnrHold.toFixed(1)}</b> dB</span>` : '') +
    '<i>Details ›</i>';
  stats.addEventListener('click', openDetails);
  c.appendChild(stats);
  if (!S.restored && !S._localRun) {
    // a restored model has no training targets in this tab — viewing and
    // exporting work, continuing the run does not (yet)
    const more = document.createElement('button');
    more.className = 'cbtn';
    more.textContent = 'Train';
    more.title = 'Continue training — the schedules stretch to the longer run';
    more.addEventListener('click', trainAfterDone);
    c.appendChild(more);
  }
  if (S.share) {
    // a shared creation is also a walkable arrival space — one click away
    const enter = document.createElement('a');
    enter.className = 'cbtn';
    enter.href = `https://arrival.space/${encodeURIComponent(S.share.id)}`;
    enter.target = '_blank';
    enter.rel = 'noopener';
    // the arrival mark from the top bar leads the label — same place, same sign
    const mark = document.querySelector('.brand-mark svg');
    if (mark) {
      const m = mark.cloneNode(true);
      m.classList.add('enter-mark');
      enter.appendChild(m);
    }
    enter.appendChild(document.createTextNode('Enter space'));
    enter.title = 'Walk this creation on arrival.space';
    c.appendChild(enter);
  }
  const rj = S.restored && S.restored.reconJson;
  if (S._localRun && S._localRun.ownSrc) {
    // a local run of the visitor's own photos: continue on the model, or
    // start fresh from the capture kept on this device
    const train = document.createElement('button');
    train.className = 'cbtn accent';
    train.textContent = 'Train';
    train.title = 'Keep training this model, or train new from the saved photos';
    train.addEventListener('click', trainLocalChoice);
    c.appendChild(train);
  } else if (rj && rj.source && rj.source.urls && rj.source.urls.length && rj.source.urls.every(Boolean)) {
    // the creation carries its photographs — the viewer can become the maker
    const train = document.createElement('button');
    train.className = 'cbtn accent';
    train.textContent = 'Train';
    train.title = `Solve and train the same ${rj.source.urls.length} photographs right here`;
    train.addEventListener('click', trainFromShare);
    c.appendChild(train);
  }
  c.appendChild(buildExport());
}

/** The local-run viewer's Train: continue refining the saved model, or a
 *  fresh solve from the photos in the capture store. */
function trainLocalChoice() {
  document.getElementById('ltchoice')?.remove();
  const r = S._localRun;
  const trainNew = async () => {
    const src = r && r.recon && r.recon.source;
    if (r && r.ownSrc) {
      const set = await openCaptureSet();
      if (!set) { flash('The photos are no longer stored on this device.', 6000); return; }
      stopTour();
      S.restored = null;
      await open(set);
      if (!WALL_FIRST) showDetail(set);
    } else if (src && Array.isArray(src.urls) && src.urls.length && src.urls.every(Boolean)) {
      // URL-backed run (a preset or a shared creation's photos): the set
      // rebuilds from the addresses kept with the record
      stopTour();
      S.restored = null;
      const n = src.urls.length;
      const set = {
        id: src.preset || '__url', name: r.name || 'Training run',
        kind: 'Saved run',
        origin: `${n} photographs kept with this run's record, fetched again to train from scratch.`,
        links: [], blurb: '', count: n,
        approx: n <= 20 ? '~3 min' : n <= 60 ? '~6 min' : '~10 min',
        urlList: src.names.map((nm, i) => ({ url: src.urls[i], name: nm })),
      };
      await open(set);
      if (!WALL_FIRST) showDetail(set);
    } else {
      flash('The photos are no longer stored on this device.', 6000);
    }
  };
  // nothing to continue (no checkpoint, no result): straight to the fresh solve
  if (!(r && (r.sog || (r.state && r.recon)))) { trainNew().catch((e) => { console.error(e); }); return; }
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'ltchoice';
  card.innerHTML = `
    <button class="card-x lt-x" id="lt-x" aria-label="Close">&times;</button>
    <b>Train these photos</b>
    <span class="lt-desc">Keep training refines the saved model further; Train new solves and
    trains from scratch with the photos kept on this device.</span>
    <div class="upcard-row">
      <button class="btn btn-accent" id="lt-cont">Keep training</button>
      <button class="btn btn-outline" id="lt-new">Train new</button>
    </div>`;
  $('stage').appendChild(card);
  $('lt-x').addEventListener('click', () => card.remove());
  $('lt-new').addEventListener('click', () => { card.remove(); trainNew().catch((e) => { console.error(e); }); });
  $('lt-cont').addEventListener('click', () => {
    card.remove();
    // a finished run is a new training decision — reselect its resolution,
    // budget and extra cycles first; a paused run simply resumes
    if (r && r.status === 'finished') { continueConfigCard(r); return; }
    continueLocalRun().catch((e) => {
      console.error(e);
      flash(`Could not continue this run: ${e.message}`, 8000);
    });
  });
}

/** Reselect the training configuration before refining a finished run —
 *  the same knobs a new training offers where they apply to a rebuild:
 *  training resolution, splat budget, and how many more cycles to train.
 *  The solved cameras are reused, never re-solved. Paused runs skip this
 *  card and resume their original configuration directly. */
function continueConfigCard(r) {
  document.getElementById('ltconfig')?.remove();
  const recon = r.recon || {};
  const fr0 = (recon.frames && recon.frames[0]) || {};
  const prevDim = Math.max(fr0.tw || 0, fr0.th || 0) || 0;
  const nativeDim = Math.max(fr0.nw || fr0.fw || 0, fr0.nh || fr0.fh || 0) || 0;
  const prevSplats = r.splats || 0;
  const baseIter = r.iter || recon.iter || 0;
  const defaultExtra = Math.max(10000, Math.round((r.maxIters || 20000) / 2));
  const recipe0 = (recon && recon.recipe) || null;
  const prevSh = recipe0 && Number.isInteger(recipe0.shDeg) ? recipe0.shDeg : 3;
  const prevMcmc = !!(recipe0 && recipe0.mcmcNoise === true);
  // the run's own input size and buffer (the recon, else its saved settings
  // when they reproduce the stored size): the default offers the INPUT in
  // start-panel language, not the combined input × buffer size
  const prevBuf = continuedBuf(recon.trainScale, r.detailProgress && r.detailProgress.settings, prevDim || undefined);
  const prevInputDim = continuedInputDim(prevDim || undefined, prevBuf);
  // resolution: every start-panel step, like a new training (Auto, the
  // fixed steps, Full); the default replays the run's own input size
  const resOpts = [`<option value="same" selected>${prevInputDim ? `${prevInputDim} px` : 'Run size'}</option>`,
    '<option value="auto">Auto</option>',
    ...[480, 640, 960, 1280, 1600, 1920].map((v) => `<option value="${v}">${v} px</option>`),
    `<option value="full">Full size${nativeDim ? ` (${nativeDim} px)` : ''}</option>`].join('');
  // the default keeps the run's ORIGINAL budget (the cap it trained under),
  // not its current count — else a run that finished below budget would
  // continue with no headroom to grow into. Records old enough to lack a
  // stored cap fall back to the current count. Explicit steps stay at or
  // above the finished size (shrinking a trained model is not offered).
  const prevCap = r.cap || recon.maxSplats || 0;
  const prevSplatLabel = prevCap ? `Original budget (${fmt(prevCap)})`
    : `Same as current${prevSplats ? ` (${fmt(prevSplats)})` : ''}`;
  const budgets = [300000, 600000, 1000000, 2000000, 4000000].filter((v) => v >= prevSplats);
  const splatOpts = [`<option value="" selected>${prevSplatLabel}</option>`,
    ...budgets.map((v) => `<option value="${v}">${fmt(v)}</option>`)].join('');
  // extra cycles mirror the start panel's fixed 10k–50k menu
  const cycleSteps = [10000, 20000, 30000, 40000, 50000];
  const nearest = cycleSteps.reduce((best, v) =>
    Math.abs(v - defaultExtra) < Math.abs(best - defaultExtra) ? v : best, 10000);
  const cycleOpts = cycleSteps.map((v) =>
    `<option value="${v}"${v === nearest ? ' selected' : ''}>+${fmt(v)}</option>`).join('');
  // spherical harmonics: the run's own degree unless reselected (the decoded
  // checkpoint is resampled to the chosen degree before training resumes)
  const shOpts = [`<option value="" selected>${prevSh}</option>`,
    ...[0, 1, 2, 3].map((v) => `<option value="${v}">${v}</option>`)].join('');
  // working buffer: the supervision grid scale, like the start panel
  const bufOpts = [`<option value="" selected>${prevBuf}×</option>`,
    ...[0.5, 1, 2].map((v) => `<option value="${v}">${v}×</option>`)].join('');
  // LOD: a fresh halving schedule above the model's current size (needs a
  // 1,000,000+ budget, like a new training)
  const lodOpts = '<option value="" selected>Off</option><option value="1">On (auto levels)</option>';
  // MCMC optimizer: the run's own set unless toggled
  const mcmcOpts = `<option value="" selected>${prevMcmc ? 'On' : 'Off'}</option><option value="1">On</option><option value="0">Off</option>`;
  // soft prune: the run's own trim unless reselected (the recipe carries
  // the trainer's softPrune block, so a run pruned at 10% replays 10%)
  const prevSoftRatio = recipe0 && recipe0.softPrune && Number.isFinite(recipe0.softPrune.ratio)
    ? recipe0.softPrune.ratio : 0;
  const prevSoft = prevSoftRatio > 0 ? `${Math.round(prevSoftRatio * 100)}%` : 'Off';
  const softOpts = [`<option value="" selected>${prevSoft}</option>`, '<option value="0">Off</option>',
    ...[0.01, 0.02, 0.05, 0.1].map((v) => `<option value="${v}">${Math.round(v * 100)}%</option>`),
    '<option value="custom">Custom…</option>'].join('');
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'ltconfig';
  card.innerHTML = `
    <button class="card-x lt-x" id="ltc-x" aria-label="Close">&times;</button>
    <b>Continue training</b>
    <span class="lt-desc">Refine <b>${esc(r.name || 'this model')}</b> further from ${fmt(baseIter)} cycles.
    Higher resolution and a larger splat budget sharpen detail; the solved cameras are reused.</span>
    <label class="upcard-opt"><span>Training resolution</span><select id="ltc-res">${resOpts}</select></label>
    <label class="upcard-opt"><span>Splat budget</span><select id="ltc-splats">${splatOpts}</select></label>
    <label class="upcard-opt"><span>Extra cycles</span><select id="ltc-iters">${cycleOpts}</select></label>
    <label class="upcard-opt"><span>Spherical harmonics</span><select id="ltc-sh">${shOpts}</select></label>
    <label class="upcard-opt"><span>Working buffer</span><select id="ltc-buf">${bufOpts}</select></label>
    <label class="upcard-opt"><span>LOD levels</span><select id="ltc-lod">${lodOpts}</select></label>
    <label class="upcard-opt"><span>MCMC optimizer</span><select id="ltc-mcmc">${mcmcOpts}</select></label>
    <label class="upcard-opt"><span>Soft prune</span><select id="ltc-softprune">${softOpts}</select><input id="ltc-softprune-custom" type="number" min="1" max="50" step="1" value="10" style="width:4.5em" hidden title="Custom trim percent (1–50)"></label>
    <div class="upcard-row">
      <button class="btn btn-outline" id="ltc-cancel">Cancel</button>
      <button class="btn btn-accent" id="ltc-go">Continue training</button>
    </div>`;
  $('stage').appendChild(card);
  $('ltc-softprune').addEventListener('change', () => {
    $('ltc-softprune-custom').hidden = $('ltc-softprune').value !== 'custom';
  });
  $('ltc-x').addEventListener('click', () => card.remove());
  $('ltc-cancel').addEventListener('click', () => card.remove());
  $('ltc-go').addEventListener('click', () => {
    const resRaw = $('ltc-res').value;
    const splatsRaw = parseInt($('ltc-splats').value, 10) || 0;
    const extraIters = parseInt($('ltc-iters').value, 10) || defaultExtra;
    const shRaw = $('ltc-sh').value;
    const bufRaw = $('ltc-buf').value;
    const lodRaw = $('ltc-lod').value;
    const mcmcRaw = $('ltc-mcmc').value;
    const softRaw = $('ltc-softprune').value;
    // Custom reads the percent input (clamped 1–50); garbage falls back to
    // untouched, so a typo never misconfigures the continuation
    const softCustomPct = softRaw === 'custom' ? Math.round(parseFloat($('ltc-softprune-custom').value) || 0) : 0;
    const softVal = softRaw === '' ? undefined : softRaw === 'custom'
      ? (softCustomPct >= 1 ? Math.min(50, softCustomPct) / 100 : undefined)
      : parseFloat(softRaw);
    card.remove();
    continueLocalRun(false, {
      res: resRaw === 'same' ? undefined : (resRaw === 'auto' ? 'auto' : (resRaw === 'full' ? 'full' : parseInt(resRaw, 10))),
      splats: splatsRaw || undefined,
      extraIters,
      sh: shRaw === '' ? undefined : parseInt(shRaw, 10),
      buf: bufRaw === '' ? undefined : parseFloat(bufRaw),
      lod: lodRaw === '1' ? true : undefined,
      mcmc: mcmcRaw === '' ? undefined : mcmcRaw === '1',
      softprune: softVal,
    }).catch((e) => {
      console.error(e);
      flash(`Could not continue this run: ${e.message}`, 8000);
    });
  });
}

/** Continue training a stored local run: decode its sog back into raw
 *  Gaussians (un-baking the Mip opacity compensation — the export bakes it
 *  for standard viewers, the trainer applies it itself), rebuild the session
 *  from the stored camera solve + the capture's photos, and run on. */
/** viewOnly: reopen a FINISHED run from its state checkpoint straight into
 *  the done viewer (same run record, so Compress patches the sog in)
 *  opts: { res, splats, extraIters, sh, buf, lod, mcmc, softprune } from the
 *  continue-config card — a reselected configuration for this continuation;
 *  unsettled knobs keep the run's own setup and automatic horizon extension. */
async function continueLocalRun(viewOnly = false, opts = {}) {
  const lr = S._localRun;
  if (!lr || !lr.recon || !(lr.state || lr.sog)) return;
  // LOD needs a 1M+ budget (the start-panel gating); reject before any teardown
  if (opts.lod === true && Math.max(lr.splats || 0, opts.splats || 0) < 1000000) {
    flash('LOD levels need a splat budget of at least 1,000,000 — raise the budget or leave LOD off.', 7000);
    return;
  }
  const recon = lr.recon;
  let files;
  const sourceNames = (recon.source && recon.source.names) || (recon.frames || []).map((f) => f.name);
  const savedCaptureId = recon.source?.captureId || String(lr.srcKey || '').replace(/^native:/, '') || null;
  let srcUrls = (recon.source && recon.source.urls) || [];
  // Records created before native URLs were stored still have a stable source
  // key and frame names. Repair those references instead of consulting the
  // mutable "last capture" slot, which can pair the model with wrong poses.
  if ((!srcUrls.length || !srcUrls.every(Boolean)) &&
      /^[A-Za-z0-9_-]{1,80}$/.test(savedCaptureId || '') && sourceNames.length) {
    srcUrls = sourceNames.map((name) =>
      `https://appassets.androidplatform.net/captures/${savedCaptureId}/${encodeURIComponent(name)}`);
    recon.source = { ...(recon.source || {}), names: sourceNames, urls: srcUrls, captureId: savedCaptureId };
    import('./store.js').then(({ patchRun }) => patchRun(lr.id,
      { srcKey: `native:${savedCaptureId}`, recon })).catch(() => {});
  }
  if (srcUrls.length && srcUrls.every(Boolean)) {
    // Stable URLs include presets and canonical native AR archives. Prefer
    // them even for own captures: loading "last capture" here can substitute
    // a newer scan and detach images from this model's saved camera poses.
    flash('Fetching the photos …', 60000);
    files = await Promise.all(srcUrls.map(async (u, i) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`photo fetch failed (${r.status})`);
      return new File([await r.blob()], recon.source.names[i], { type: 'image/jpeg' });
    }));
  } else {
    const rec = await loadLastCapture();
    if (!rec || !rec.files) { flash('The photos are no longer stored on this device.', 6000); return; }
    // photos matched BY NAME to the run's own frame order — re-sorting could
    // reorder against recon.cams imgIdx, and a newer capture must not sneak in
    const byName = new Map(rec.files.map((e) => [e.name, e]));
    files = (recon.frames || []).map((fr) => {
      const e = byName.get(fr.name);
      return e ? new File([e.blob], e.name, { type: e.blob.type || 'image/jpeg' }) : null;
    });
    if (!files.length || files.some((f) => !f)) {
      flash('The stored photos no longer match this run — Train new instead.', 7000);
      return;
    }
  }
  flash('Preparing to continue training …', 60000);
  let g, engine, stateIter = null, recipe = null;
  if (lr.state) {
    // leave checkpoint: the trainer's raw floats — nothing baked, nothing
    // quantized, tagged with the engine that wrote them. Bit-exact resume.
    const { parseState } = await import('./session_io.js');
    const st = parseState(new Uint8Array(await lr.state.arrayBuffer()));
    g = st.gaussians; engine = st.engine; stateIter = st.iter; recipe = st.recipe;
  } else {
    const { decodeModel } = await import('./session_io.js');
    ({ gaussians: g } = await decodeModel(new Uint8Array(await lr.sog.arrayBuffer()), null));
  }
  // the trainer recipe the run was made with (state header, else the recon
  // JSON); null = a record from before recipes = the legacy defaults
  recipe = recipe || (recon && recon.recipe) || null;
  // a reselected SH degree resamples the decoded harmonics (low bands kept,
  // growth zero-padded) so seedFrom rebuilds the chosen degree; unsettled,
  // the trainer replays the run's own degree from the checkpoint
  let contShDeg = null;
  if (opts.sh != null) {
    try {
      const { convertSHDegree } = await import('./session_io.js');
      const DEG_OF_SHK = { 0: 0, 3: 1, 8: 2, 15: 3 };
      const curDeg = g.shK != null ? DEG_OF_SHK[g.shK] : (recipe && recipe.shDeg);
      if (curDeg == null) throw new Error(`unfamiliar band count ${g.shK}`);
      if (curDeg !== opts.sh) convertSHDegree(g, opts.sh);
      contShDeg = opts.sh;
    } catch (e) {
      flash(`Kept the run's SH degree (${e.message}).`, 7000);
    }
  }
  // un-bake opacity compensation (pure fn of position, mean scale, focal,
  // camera centres — recompute and divide out in logit space); only sog/ply
  // exports carry the baking, the raw checkpoint never does. Under the
  // placement set the trainer has no Mip compensation, so its exports were
  // never baked and there is nothing to divide out (a comp-trained share
  // continued under the set keeps its baked opacities — experimental).
  // a model trained WITHOUT Mip compensation was exported unbaked — the
  // recipe says so; records without one are legacy (baked) unless the
  // placement set is on in this tab
  const wasBaked = recipe ? recipe.mipComp !== false : !PLACEMENT_ON;
  if (!lr.state && wasBaked) {
    const pos = [];
    for (const c of recon.cams) {
      const R = c.R, t = c.t;
      pos.push(
        -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
        -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
        -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]));
    }
    const fr0 = recon.frames[recon.cams[0].imgIdx];
    const f = recon.cams[0].f * ((fr0.tw || fr0.fw) / fr0.fw);
    const d = g.data;
    for (let i = 0; i < g.n; i++) {
      const b = i * 16;
      let z2 = Infinity;
      for (let c = 0; c < pos.length; c += 3) {
        const dx = d[b] - pos[c], dy = d[b + 1] - pos[c + 1], dz = d[b + 2] - pos[c + 2];
        const q = dx * dx + dy * dy + dz * dz;
        if (q < z2) z2 = q;
      }
      const z = Math.max(1e-3, Math.sqrt(z2));
      const sMean = Math.exp((d[b + 3] + d[b + 4] + d[b + 5]) / 3);
      const s2d = f * sMean / z;
      const comp = (s2d * s2d) / (s2d * s2d + 0.3);
      const opa = 1 / (1 + Math.exp(-d[b + 13]));
      const raw = Math.min(1 - 1e-6, Math.max(1e-6, opa / Math.max(comp, 1e-6)));
      d[b + 13] = Math.log(raw / (1 - raw));
    }
  }
  // tear down the viewer, build the training session (trainFromShare's reset)
  stopTour();
  $('start').hidden = true;   // a paused tile resumes straight from the wall
  try { S.session.dispose(); } catch (e) { /* view-only facade */ }
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  S.gen++;
  const gen = S.gen;
  try {
    S.session = null; S.share = null; S.restored = null;
    S._viewerOpen = false;
    S._fromSpace = null;
    S._thumbsUrl = null; S._thumbPack = null;
    if (history.state && history.state.sj) history.replaceState(null, '');
    S.plyBlob = null; S.sogBlob = null;
    // keep the source identity: own captures stay '__own' (the next resume goes
    // back to the capture store), URL-backed runs keep their preset/urls so a
    // leaving during THIS run checkpoints a resumable source list again
    const nativeCaptureId = savedCaptureId;
    S.preset = lr.ownSrc
      ? { id: '__own', name: `${lr.name || 'Your photos'}`,
          ...(nativeCaptureId ? { capture: { metadata: { nativeARCore: true, nativeCaptureId } } } : {}) }
      : { id: (recon.source && recon.source.preset) || '__url', name: `${lr.name || 'Your photos'}` };
    S.loadedFiles = files.map((f, i) => Object.assign(f, { url: srcUrls[i] || undefined }));
    S.photos = files.map((f, i) => ({ url: URL.createObjectURL(f), name: f.name, i }));
    buildStrip();
    const baseIter = stateIter ?? (lr.iter || 0);
    // a finished run continues PAST its old horizon — by the configured extra
    // cycles, or the automatic half-horizon when none were chosen; a paused
    // one resumes toward the horizon it was already aiming at
    const defaultExtra = Math.max(10000, Math.round((lr.maxIters || 20000) / 2));
    S.maxIters = lr.status !== 'finished' && lr.maxIters > baseIter
      ? lr.maxIters
      : baseIter + (Number.isSafeInteger(opts.extraIters) && opts.extraIters > 0 ? opts.extraIters : defaultExtra);
    const fr0 = recon.frames[0] || {};
    // the continuation may reselect the training size (higher resolution,
    // more splats) like a new training; unsettled it replays the run's own
    // input scale AND buffer. The stored size is the COMBINED size (input ×
    // buffer), so the buffer is divided back out before the value passes
    // through the multiplier again — else the buffer applies twice and the
    // continuation trains at (e.g.) 1/16 instead of the run's 1/4
    const prevDim = Math.max(fr0.tw || 0, fr0.th || 0) || undefined;
    const prevBuf = continuedBuf(recon.trainScale, lr.detailProgress && lr.detailProgress.settings, prevDim);
    const prevInputDim = continuedInputDim(prevDim, prevBuf);
    const nativeDim = Math.max(fr0.nw || fr0.fw || 0, fr0.nh || fr0.fh || 0) || undefined;
    const trainMaxDim = opts.res === 'auto' ? undefined
      : opts.res === 'full' ? nativeDim
      : Number.isFinite(opts.res) ? opts.res : prevInputDim;
    // the run's own budget carries over (finished runs included — it is the
    // cap they trained under, not their current count); the model size
    // floors it, an explicit card choice raises it
    const cap = Math.max(g.n, lr.splats || 0, lr.cap || 0, (recon && recon.maxSplats) || 0, opts.splats || 0);
    // LOD on a continuation starts a fresh halving schedule ABOVE the model's
    // current size (snapshotting below it would mislabel the levels); without
    // headroom there is nothing to schedule
    if (opts.lod === true) {
      if (cap < 1000000) {
        flash('LOD levels need a splat budget of at least 1,000,000 — continuing without them.', 7000);
      } else {
        const plan = buildLODPlan(cap, 0, Math.max(10000, S.maxIters - baseIter));
        const levels = plan ? plan.levels.filter((l) => l > g.n) : [];
        if (!levels.length) {
          flash('The model already fills its budget — LOD levels need headroom to grow into.', 7000);
        } else {
          S.lodPlan = { ...plan, levels, idx: 0, holdUntil: null, snaps: [] };
        }
      }
    }
    // rebuild the SAME trainer the run was made with: the recipe carries the
    // MCMC / needle / placement sets and the refine cadence; sizing is per run
    // and the card may reselect the SH degree or the MCMC set on top
    const { refineEvery: recipeRefine, ...recipeTrainer } = recipe || {};
    // the card's MCMC toggle rewrites the optimizer set the same way the
    // start panel builds it; untouched, the recipe replays the run's own set
    if (opts.mcmc === true) {
      Object.assign(recipeTrainer, { growRate: 0.05, mcmcNoise: true, scaleReg: 0.01, moveCap: 0.25 });
      if ((contShDeg ?? recipeTrainer.shDeg ?? 3) > 0) recipeTrainer.shLr = 3e-4;
      else delete recipeTrainer.shLr;
    } else if (opts.mcmc === false) {
      delete recipeTrainer.mcmcNoise; delete recipeTrainer.scaleReg;
      delete recipeTrainer.moveCap; delete recipeTrainer.shLr;
      // the placement set carries its own growth rate — classic restores that,
      // otherwise the trainer default applies
      if (recipeTrainer.refineV2) recipeTrainer.growRate = 0.1;
      else delete recipeTrainer.growRate;
    }
    // the card reselects soft-prune; untouched (and on resume/view paths,
    // which pass no overrides), the recipe replays the run's own setting —
    // a run pruned at 10% keeps pruning at 10%. The start panel stays out:
    // its value may have drifted since the run and must not leak in.
    if (opts.softprune !== undefined) {
      if (opts.softprune > 0) recipeTrainer.softPrune = { ratio: opts.softprune, every: 3000, until: 0.85 };
      else delete recipeTrainer.softPrune;
    }
    const contRefine = opts.mcmc === true ? 500 : opts.mcmc === false ? undefined : recipeRefine;
    const bufScale = opts.buf ?? (prevBuf !== 1 ? prevBuf : undefined);
    // same OOM warning as a fresh run: the continue card reselects the
    // buffer, so this decision point needs the message too
    if ((bufScale ?? 1) === 2 && isPhoneClass()) {
      flash('Working buffer 2× roughly quadruples GPU working memory — on this phone the run may abort for lack of memory. 1× trains the same photos safely.', 8000);
    }
    const ses = createSession({
      maxIters: S.maxIters, holdout: -1,
      // continuations obey the GPU-usage slider like fresh runs (st here is
      // the parsed checkpoint, so the setting comes from S.settings)
      gpuUsage: (S.settings.gpu ?? 100) / 100,
      ...(contRefine != null ? { refineEvery: contRefine } : {}),
      frames: { trainMaxDim, ...(bufScale != null ? { trainScale: bufScale } : {}) },
      trainer: {
        ...recipeTrainer,
        maxSplats: cap, shDeg: contShDeg ?? recipeTrainer.shDeg ?? 3,
        capMult: Math.max(2, Math.ceil(cap / Math.max(1, g.n))),
        ...(engine === 'v2' ? { engine: 'v2', refineEvery: 200 } : {}),
      },
      trainWorker: trainWorkerOpt(),
    });
    S.session = ses;
    ses.on('stage', (e) => { if (S.gen === gen) onStage(e); });
    ses.on('metrics', (e) => { if (S.gen === gen) onMetrics(e); });
    ses.on('event', (e) => { if (S.gen === gen) onTrainEvent(e); });
    await ses.load(files);
    if (S.gen !== gen) return;
    ses.useReconstruction({
      known: !!recon.knownCameras,
      cams: recon.cams.map((c) => ({ imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f, cx: c.cx, cy: c.cy })),
      points: [],
      k1: recon.k1 || 0, k2: recon.k2 || 0,
      ...(recon.fFeat ? { fFeat: recon.fFeat } : {}),
    });
    if (undistortFrames(ses.frames, ses.recon)) { /* targets match the original run */ }
    // the recon JSON carries the run's scene radius — without it seedFrom
    // falls back to 10 and every radius-scaled quantity (position lr, min/max
    // scale, Langevin noise) is wrong on resume (e2e resume caught it once the
    // recipe replayed the MCMC noise: positions doubled within a few steps)
    await ses.seedFrom(g, { iter: baseIter, sceneRadius: recon.sceneRadius || undefined });
    if (S.gen !== gen) return;
    if (S.lodPlan) ses.trainer.growLimit = S.lodPlan.levels[S.lodPlan.idx] || S.lodPlan.levels[0];
    S.iter = baseIter;
    S.psnrTrain = null; S.psnrHold = null;
    S.holdHist = []; S.chartEvents = [];  S.growthStopped = false;
    restoreRunDetails(lr, ses, viewOnly ? {} : { trimToIter: baseIter });
    buildSceneFromSession();
    mountModelCanvas();
    if (viewOnly) {
      S.maxIters = baseIter;
      S.runId = lr.id;
      S.minutes = lr.minutes || 0;
      S.psnrTrain = lr.psnr ?? null;
      S.splats = g.n;
      S.trainT0 = performance.now();
      S.restored = { hasState: true, stats: (recon && recon.stats) || {}, reconJson: recon };
      S._viewerOpen = true;
      $('btn-new').hidden = false;
      enterDone();
      flash(`${fmt(g.n)} splats — Compress turns this into a shareable .sog`, 6000);
      return;
    }
    S.minutes = Math.max(S.minutes || 0, lr.minutes || 0,
      (lr.detailProgress && lr.detailProgress.minutes) || 0);
    S._carryTrainElapsed = true;
    // every continuation — paused or finished — trains back into its own
    // record: the saved result replaces the previous trained state there, the
    // root camera poses it rebuilt from stay, and its photos are the capture
    // store's blobs matched by name above, reused by reference and never
    // re-saved, so continuing costs no extra image storage
    S._resumeRunId = lr.id;
    startTraining();
    flash(`Continuing from ${fmt(baseIter)} cycles — +${fmt(S.maxIters - baseIter)} more.`, 6000);
  } catch (e) {
    if (S.gen !== gen) throw e;   // superseded mid-flight: another screen owns the UI now
    trainDeadEnd();
    throw e;   // the caller flashes "Could not continue this run: …" on top
  }
}

/** Done screen's Train before the finished record adopts (or when the run has
 *  none — a restore, a demo, a sub-200-cycle stub): an immediate tap waits
 *  for the securing pass, then takes the finished-run path (config card +
 *  history) like Yours does; only a record-less session falls back to the
 *  blind live resume. */
async function trainAfterDone() {
  if (!S.session || S.state !== 'done') return;
  if (S._localRun) { trainLocalChoice(); return; }
  if (S._securing) {
    const gen = S.gen;
    flash('Securing the finished result first…', 4000);
    try { await S._securing; } catch { /* best-effort: fall through to live resume */ }
    if (S.gen !== gen) return;
    if (S._localRun && S.state === 'done' && S.session) { trainLocalChoice(); return; }
  }
  continueTraining();
}

/** Resume from done: raise the horizon, restore the curve, back to train. */
async function continueTraining() {
  if (!S.session || S.state !== 'done') return;
  const gen = S.gen;
  stopTour();
  S.plyBlob = null; S.sogBlob = null;   // the cached export goes stale the moment training resumes
  S.etaAt = null;
  // continueFor is async (the remote horizon crosses the worker first) —
  // awaiting it is what keeps S.maxIters a number: assigning the promise
  // used to NaN the chart's x-scale and blank the restored history
  S.maxIters = await S.session.continueFor(MORE_ITERS);
  if (S.gen !== gen || !S.session || S.state !== 'done') return;
  S.state = 'train';
  S.trainVisuals = defaultTrainingVisuals();
  syncTrainingVisualsControl();
  S.trainT0 = performance.now() - S.minutes * 60000;   // minutes stay cumulative
  S.growthStopped = false;
  S.atFrame = -1; S.fadeTo = 0;
  vp.lock = null;
  $('stage').dataset.cursor = 'grab';
  renderControls();
  dock('train');
  syncTrainingVisualsControl();
  if (chart) {
    chart.train = S.session.lossHistory.map(([i, v]) => [i, v]);
    chart.hold = (S.holdHist || []).slice();
    chart.maxIter = S.maxIters;
    chart.draw();
  }
  paintStrip();
}

const DL_ICON = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" class="dl">' +
  '<path d="M22 15.3333V19.7777C22 20.3671 21.7659 20.9323 21.3491 21.349C20.9324 21.7658 20.3671 21.9999 19.7778 21.9999H4.22222C3.63285 21.9999 3.06762 21.7658 2.65087 21.349C2.23413 20.9323 2 20.3671 2 19.7777V15.3333"/>' +
  '<path d="M6.44449 9.77745L12 15.333M12 15.333L17.5556 9.77745M12 15.333L12 1.99967"/></svg>';

/** The model as a compressed .sog Blob — converted once per run (cached),
 *  with a proper progress card: the compression takes real time on big
 *  models and the writer reports its stages. */
async function getSogBlob() {
  stopWalk();   // the posed buffer is not what anyone wants compressed
  if (S.sogBlob) return S.sogBlob;
  // ONE compression per run: a second request (Download .sog while Share is
  // compressing, or the old auto-compress at finish) used to start a second
  // k-means job on a second GPU device
  if (S._sogJob && S._sogJob.gen === S.gen) return S._sogJob.p;
  const gen = S.gen;
  const p = compressSog(gen).finally(() => { if (S._sogJob && S._sogJob.gen === gen) S._sogJob = null; });
  S._sogJob = { gen, p };
  return p;
}

async function compressSog(gen) {
  // the viewer is not navigable while the compressor owns the GPU — pause
  // the tour instead of rendering into the stall, resume it afterwards
  const wasTour = !!S.tour;
  stopTour();
  const ply = S.plyBlob || await S.session.exportPlyBlob();
  if (S.gen !== gen) throw new Error('run changed');
  S.plyBlob = ply;
  document.getElementById('sogcard')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'sogcard';
  card.innerHTML = `
    <b>Compressing to .sog</b>
    <span class="sog-status" id="sog-status">Reading the model …</span>
    <div class="prep-meter"><i id="sog-bar" style="width:0%"></i></div>`;
  $('stage').appendChild(card);
  try {
    const { plyToSog } = await import('./sog.js');
    const bytes = new Uint8Array(await ply.arrayBuffer());
    const blob = await plyToSog(bytes, {
      onProgress: ({ label, frac }) => {
        const st = document.getElementById('sog-status');
        const bar = document.getElementById('sog-bar');
        if (st && label) st.textContent = label;
        if (bar && frac != null) bar.style.width = `${(frac * 100).toFixed(1)}%`;
      },
    });
    if (S.gen !== gen) throw new Error('run changed');
    S.sogBlob = blob;
    renderControls();   // the Compress button has done its job
    // the library record: sog in, the raw state checkpoint out (10-20x
    // smaller, and the sog is what the wall tile opens)
    if (S.runId) {
      try {
        const { buildReconJson } = await import('./session_io.js');
        const { patchRun } = await import('./store.js');
        await patchRun(S.runId, {
          sog: blob, recon: buildReconJson(S), splats: S.splats,
          psnr: S.psnrHold ?? S.psnrTrain ?? null,
          detailSolve: captureRunSolveDetails(),
          detailProgress: captureRunProgressDetails(),
          detailGpu: gpuFacts(false),
          state: null, ckptAt: null,
        });
      } catch (e) { console.warn('local save failed', e); }
    }
    return blob;
  } finally {
    document.getElementById('sogcard')?.remove();
    if (wasTour && S.gen === gen && S.state === 'done' && !S.tour) startTour(true);
  }
}

/** The viewer link of a shared scene — index.html explicitly (the bare
 *  /splat-js URL 301s through the CDN and drops its query). */
const shareLinkOf = (spaceId) => {
  const path = location.pathname.endsWith('.html') ? location.pathname : location.pathname.replace(/\/$/, '') + '/index.html';
  return `${location.origin}${path}?space=${encodeURIComponent(spaceId)}`;
};
const SHARE_ICON = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" class="sh">' +
  '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>' +
  '<path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4"/></svg>';

/** Save a Blob under the run's name. */
async function downloadBlob(blob, ext) {
  const base = ((S.preset && S.preset.name) || 'splat').toLowerCase().replace(/\W+/g, '_');
  const filename = `${base}.${ext}`;
  const mime = /\.zip$/i.test(ext) ? 'application/zip'
    : ext === 'txt' ? 'text/plain' : 'application/octet-stream';
  return saveBlob(blob, filename, mime);
}

/** The download options for the current run — listed under Share in the
 *  share sheet. Each carries its own progress and messages. */
function exportActions() {
  const mb = (S.splats * 164 / 1e6).toFixed(1); // 41 float properties per splat (SH deg 2)
  const sogMb = (S.splats * 164 / 1e6 / 15).toFixed(1); // SOG lands around 1/15th
  const acts = [];
  acts.push({ act: 'sog', label: 'Download .sog', sub: `Compressed for the web · ~${sogMb} MB`, run: async () => {
    try {
      const blob = await getSogBlob();
      const saved = await downloadBlob(blob, 'sog');
      flash(saved ? `${fmt(S.splats)} splats compressed and saved (${(blob.size / 1e6).toFixed(1)} MB).` : 'Save cancelled.', 4000);
    } catch (e) {
      console.error(e);
      flash(`SOG export failed: ${e.message}`, 6000);
    }
  } });
  if (S.lodPlan && S.lodPlan.snaps.length) acts.push({ act: 'lod', label: 'Download LOD', sub: `Streamed SOG, ${S.lodPlan.snaps.length + 1} detail levels · zip`, run: async () => {
    try {
      document.getElementById('sogcard')?.remove();
      const card = document.createElement('div');
      card.className = 'upcard';
      card.id = 'sogcard';
      card.innerHTML = `
        <b>Building the streamed LOD</b>
        <span class="sog-status" id="sog-status">Reading the levels …</span>
        <div class="prep-meter"><i id="sog-bar" style="width:0%"></i></div>`;
      $('stage').appendChild(card);
      const { plysToLodEntries } = await import('./sog.js');
      // finest first (LOD 0 = the full model), then the snapshots descending
      const full = S.plyBlob || await S.session.exportPlyBlob();
      const levels = [new Uint8Array(await full.arrayBuffer())];
      for (const snap of [...S.lodPlan.snaps].reverse()) {
        levels.push(new Uint8Array(await snap.blob.arrayBuffer()));
      }
      const entries = await plysToLodEntries(levels, {
        onProgress: ({ label, frac }) => {
          const st2 = document.getElementById('sog-status');
          const bar = document.getElementById('sog-bar');
          if (st2 && label) st2.textContent = label;
          if (bar && frac != null) bar.style.width = `${(frac * 100).toFixed(1)}%`;
        },
      });
      const zip = zipStore(entries.map(([name, data]) => ({ name: name.replace(/^\//, ''), data })));
      document.getElementById('sogcard')?.remove();
      const saved = await downloadBlob(zip, 'lod.zip');
      flash(saved ? `Streamed LOD with ${levels.length} levels · ${(zip.size / 1e6).toFixed(1)} MB saved.` : 'Save cancelled.', 5000);
    } catch (e) {
      document.getElementById('sogcard')?.remove();
      console.error(e);
      flash(`LOD build failed: ${e.message}`, 6000);
    }
  } });
  acts.push({ act: 'ply', label: 'Download .ply', sub: `Standard splat file · ${mb} MB`, run: async () => {
    try {
      stopWalk();
      const saved = await downloadBlob(S.plyBlob || await S.session.exportPlyBlob(), 'ply');
      flash(saved ? `${fmt(S.splats)} splats saved to the selected location.` : 'Save cancelled.', 3500);
    } catch (e) {
      console.error(e);
      flash(`PLY save failed: ${e.message}`, 6000);
    }
  } });
  if (!S.restored) acts.push({ act: 'simplify', label: 'Download simplified .ply', sub: 'Fewer splats, same look · pick a size', run: async () => {
    try {
      stopWalk();
      document.getElementById('simpcard')?.remove();
      const card = document.createElement('div');
      card.className = 'upcard';
      card.id = 'simpcard';
      card.innerHTML = `
        <b>Shrink the model</b>
        <span class="sog-status" id="simp-status">Merging nearby splats keeps the look — pick a target size.</span>
        <div class="row" id="simp-pick">
          <button class="btn" data-ratio="0.75">Keep ¾</button>
          <button class="btn" data-ratio="0.5">Keep ½</button>
          <button class="btn" data-ratio="0.25">Keep ¼</button>
          <button class="btn linkish" data-ratio="">Cancel</button>
        </div>
        <div class="prep-meter"><i id="simp-bar" style="width:0%"></i></div>`;
      $('stage').appendChild(card);
      const ratio = await new Promise((resolve) => {
        card.querySelector('#simp-pick').addEventListener('click', (e) => {
          const r = e.target && e.target.dataset ? e.target.dataset.ratio : null;
          if (r == null) return;   // tapped between the buttons
          resolve(r === '' ? null : parseFloat(r));
        });
      });
      if (ratio == null) { card.remove(); return; }
      card.querySelector('#simp-pick').remove();
      const st = card.querySelector('#simp-status');
      const bar = card.querySelector('#simp-bar');
      const from = S.splats;
      let to = Math.ceil(from * ratio);
      const blob = await S.session.exportSimplifiedPlyBlob(ratio, {
        onStatus: (m) => { if (st) st.textContent = String(m).replace(/^simplify: /, ''); },
        onProgress: (p) => {
          to = p.currentCount;
          const span = from - p.targetCount;
          if (bar && span > 0) bar.style.width = `${(100 * (from - p.currentCount) / span).toFixed(1)}%`;
        },
      });
      card.remove();
      const saved = await downloadBlob(blob, 'ply');
      flash(saved ? `${fmt(from)} → ${fmt(to)} splats saved (${(blob.size / 1e6).toFixed(1)} MB).` : 'Save cancelled.', 5000);
    } catch (e) {
      document.getElementById('simpcard')?.remove();
      console.error(e);
      flash(`Simplify failed: ${e.message}`, 6000);
    }
  } });
  if (!S.restored) acts.push({ act: 'session', label: 'Download session', sub: 'Re-loadable + resumable · sog, poses, raw state', run: async () => {
    try {
      const sog = await getSogBlob();
      flash('Packing the session …', 60000);
      const zip = await buildSessionZip(S, sog);
      const saved = await downloadBlob(zip, 'session.zip');
      if (!saved) { flash('Save cancelled.', 4000); return; }
      flash(`Session saved · ${(zip.size / 1e6).toFixed(1)} MB. Load it back with ` +
        `?model=<url> or by dropping it on the app.`, 7000);
    } catch (e) {
      console.error(e);
      flash(`Session save failed: ${e.message}`, 6000);
    }
  } });
  if (S.loadedFiles && S.loadedFiles.length) acts.push({ act: 'imgs', label: 'Download photos', sub: `The ${S.loadedFiles ? S.loadedFiles.length : 0} training images · zip`, run: async () => {
    try {
      if (!S.loadedFiles || !S.loadedFiles.length) { flash('No source images in this run.'); return; }
      flash('Packing your photos …', 60000);
      const entries = [];
      for (const f of S.loadedFiles) {
        const blob = f.source || f;
        entries.push({ name: f.name, data: new Uint8Array(await blob.arrayBuffer()) });
      }
      const zip = zipStore(entries);
      const saved = await downloadBlob(zip, 'photos.zip');
      flash(saved ? `${entries.length} photos saved in a ZIP.` : 'Save cancelled.', 4000);
    } catch (e) {
      console.error(e);
      flash(`Photo ZIP save failed: ${e.message}`, 6000);
    }
  } });
  return acts;
}

/** The finished run's one call to action: Share (opens the share sheet,
 *  which also lists the downloads). A scene someone else shared cannot be
 *  re-shared, so it gets Download instead. */
function buildExport() {
  const wrap = document.createElement('div');
  wrap.className = 'exportwrap';
  if (S.share) {
    wrap.innerHTML = `<button class="cbtn accent btn-share" title="Copy this scene's link">${SHARE_ICON}Share</button>`;
    wrap.querySelector('button').addEventListener('click', () => shareDialog(null, { link: true }));
  } else {
    wrap.innerHTML = `<button class="cbtn btn-share" title="Download this model">${DL_ICON}Download</button>`;
    wrap.querySelector('button').addEventListener('click', () => shareDialog(null, { downloadsOnly: true }));
  }
  return wrap;
}

/** Local export sheet and link-copy sheet. Account login and cloud publishing
 *  have been removed; existing public links remain shareable. */
function shareDialog(rec = null, { downloadsOnly = false, link = false } = {}) {
  downloadsOnly ||= !link;
  document.getElementById('upcard')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'upcard';
  const acts = (S.session && (!rec || rec === S._localRun)) ? exportActions() : [];
  const url = link && S.share ? shareLinkOf(S.share.id) : '';
  card.innerHTML = `
    <button class="card-x" id="sh-x" aria-label="Close">&times;</button>
    <b>${link ? 'Share this scene' : 'Download'}</b>
    ${link && S.share ? `<div class="sh-link">
      <input id="sh-url" type="text" readonly value="${esc(url)}">
      <div class="upcard-row sh-linkrow">
        <a class="linkish sh-enter" href="https://arrival.space/${encodeURIComponent(S.share.id)}" target="_blank" rel="noopener">Enter the space ↗</a>
        ${navigator.share ? '<button type="button" class="btn btn-outline" id="sh-native">Share …</button>' : ''}
        <button type="button" class="btn btn-accent" id="sh-copy">Copy link</button>
      </div>
    </div>` : ''}
    ${acts.length ? `<div class="sh-dl">${downloadsOnly ? '' : '<span class="sh-dl-head">Or download</span>'}
      ${acts.map((a) => `<button type="button" data-dl="${a.act}"><b>${esc(a.label)}</b><span>${esc(a.sub)}</span></button>`).join('')}
    </div>` : '<p class="sh-hint">No downloadable files are available in this view.</p>'}`;
  $('stage').appendChild(card);
  for (const a of acts) card.querySelector(`[data-dl="${a.act}"]`).addEventListener('click', () => { card.remove(); a.run(); });
  card.querySelector('#sh-x').addEventListener('click', () => card.remove());
  if (link && S.share) {
    const input = card.querySelector('#sh-url');
    input.addEventListener('focus', () => input.select());
    card.querySelector('#sh-copy').addEventListener('click', async (e) => {
      try { await navigator.clipboard.writeText(url); e.currentTarget.textContent = 'Copied'; }
      catch { input.select(); flash('Press Ctrl+C to copy the link.', 4000); }
    });
    card.querySelector('#sh-native')?.addEventListener('click', async () => {
      try { await navigator.share({ title: S.share.title || 'Splat scene', url }); } catch { /* dismissed */ }
    });
  }
}

/** The gallery hero is the capture's FIRST PHOTOGRAPH, downscaled — the
 *  real image, not a model render. */
async function photoThumb() {
  try {
    const p = S.photos && S.photos[0];
    if (!p) return null;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = p.url;
    await img.decode();
    const W = Math.min(640, img.naturalWidth);
    const H = Math.max(1, Math.round(W * img.naturalHeight / img.naturalWidth));
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    cv.getContext('2d').drawImage(img, 0, 0, W, H);
    return await new Promise((res) => cv.toBlob(res, 'image/webp', 0.85));
  } catch (e) {
    return null; // tainted canvas / missing photo -> fall back to a render
  }
}

/** Fallback when no photograph is reachable: a ~640px view render from the
 *  first registered camera's pose. */
async function renderShareThumb() {
  try {
    const ses = S.session;
    // remote training owns the view (and a transferred canvas has no
    // main-side pixels to drawImage) — this scratch flow is local-only
    if (!ses || !ses.trainer || !ses.trainer.device) return null;
    // the hero is the first registered camera's own view — the shot the
    // creator opened the capture with
    const c = S.scene.cams.find((k) => k.R) || null;
    if (!c) return null;
    // the thumb wears the hero camera's own aspect — a portrait scene
    // rendered into a landscape frame pads itself with unreconstructed
    // darkness that no display-side crop can remove
    const ar = (c.w && c.h) ? c.w / c.h : 1.6;
    const W = ar >= 1 ? 640 : Math.max(240, Math.round(400 * ar));
    const H = Math.round(W / ar);
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    quietView(ses.view.attach(cv));
    quietView(ses.view.setCamera({ R: c.R, t: c.t, f: c.f * (W / c.w), cx: W / 2, cy: H / 2, w: W, h: H }));
    quietView(ses.view.renderNow(true));
    await ses.trainer.device.queue.onSubmittedWorkDone();
    const out = document.createElement('canvas');
    out.width = W; out.height = H;
    out.getContext('2d').drawImage(cv, 0, 0);
    if (gpuCanvas) { quietView(ses.view.attach(gpuCanvas)); S._viewKey = ''; }
    return await new Promise((res) => out.toBlob(res, 'image/webp', 0.85));
  } catch (e) {
    console.error(e);
    return null;
  }
}

/** ?space=<id>: a shared creation — resolve it (public endpoint, honors the
 *  space's privacy) and present it like a finished run. */
async function restoreShared(spaceId) {
  try {
    closeOverlays();
    $('start').hidden = true;
    $('detail').hidden = true;
    flash('Loading the shared creation …', 120000);
    const { resolveShare } = await import('./share.js');
    const sh = await resolveShare(spaceId);
    S.share = { id: spaceId, title: sh.title };
    // the tile's image is already in the browser cache — let it stand in
    // for the model while the SOG streams
    const heroSrc = (sh.splatjs && sh.splatjs.thumbUrl) || sh.screenshotUrl;
    if (heroSrc) {
      document.getElementById('share-hero')?.remove();
      const hero = Object.assign(new Image(), { src: heroSrc, id: 'share-hero', alt: '' });
      $('stage').insertBefore(hero, $('stage').firstChild);
    }
    S.fpvSet = !!sh.splatjs.fpv;   // per-set: first-person controls
    await restoreSession({ url: sh.splatjs.sogUrl, reconUrl: sh.splatjs.reconUrl });
    renderControls();
  } catch (e) {
    console.error(e);
    S.share = null;
    // boot skips mountWall while a share link loads — loadFailed mounts the
    // feed properly and says what happened on a real card
    loadFailed('That shared creation didn\'t load', e.message);
  }
}

/** Viewing is the detail card's primary action when a trained model exists;
 *  Start training steps back to the outline style there. Cards without a
 *  View button keep Start as the accent. */
function setStartStyle(primary) {
  const b = $('btn-go');
  b.classList.toggle('btn-accent', primary);
  b.classList.toggle('btn-cta', primary);
  b.classList.toggle('big', primary);
  b.classList.toggle('btn-outline', !primary);
  b.textContent = primary ? 'Start training' : 'Train locally';
}

/** The stat strip under the detail hero. Before a run starts it describes
 *  the INPUT — how many photographs, at what resolution — plus at most the
 *  train PSNR of the published model. Facts live in S.detailFacts and the
 *  strip repaints as they trickle in (hero decode, share resolve). */
function paintDetailStats() {
  const el = $('detail-stats');
  const f = S.detailFacts || {};
  const cells = [];
  if (f.frames) cells.push([fmt(f.frames), f.framesLabel || 'photographs']);
  if (f.res) cells.push([f.res, 'resolution']);
  if (f.trainRes) cells.push([f.trainRes, 'trained at']);
  if (f.splats) cells.push([fmt(f.splats), 'splats']);
  if (f.dB) cells.push([`${(+f.dB).toFixed(1)} dB`, f.dBLabel || 'train psnr']);
  el.innerHTML = cells.map(([v, l]) => `<div><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join('');
  el.hidden = !cells.length;
}

/** The focused detail card: one set, its description, and the only Start
 *  button. Back returns to the front page. Classic mode only. */
function showDetail(setOrPreset) {
  if (WALL_FIRST) return;
  document.getElementById('detail-view')?.remove();
  S.pendingShare = null;   // this card is a set with its photos in hand
  const hero = $('detail-hero');
  const src = (S.photos && S.photos[0] && S.photos[0].url) || null;
  hero.hidden = !src;
  if (src) hero.src = src;
  // input facts: the count is known; the resolution reads off the hero
  // photograph once it decodes (the same image the card displays anyway)
  const facts = S.detailFacts = { frames: (S.photos && S.photos.length) || 0 };
  paintDetailStats();
  if (src) {
    hero.onload = () => {
      if (S.detailFacts !== facts || !hero.naturalWidth) return;
      facts.res = `${hero.naturalWidth} × ${hero.naturalHeight}`;
      S.setMaxDim = Math.max(hero.naturalWidth, hero.naturalHeight);   // the Training resolution rows are built from this
      paintDetailStats();
    };
  }
  // the avatar route is offered on own captures only: the matte reads the
  // frames themselves, and a benchmark set of a truck is not a person
  {
    const own = !!(setOrPreset && Array.isArray(setOrPreset.files) && setOrPreset.files.length >= 2 && !setOrPreset.spaceId && !setOrPreset.capture?.transforms);
    $('av-choice').hidden = !own;
    $('set-avatar').checked = own && !!S.avatarWanted;
    if (!own) S.avatarWanted = false;
  }
  const capturedFiles = !!setOrPreset?.files?.length;
  $('xr-set-tools').hidden = !capturedFiles;
  $('xr-review-set').disabled = !capturedFiles || setOrPreset.files.length < 3;
  $('xr-export-set').disabled = !capturedFiles || setOrPreset.files.length < 3;
  $('xr-set-mode').textContent = setOrPreset?.capture?.transforms ? 'KNOWN POSES · NO CAMERA SOLVE' : 'PHOTO INPUT · ORIGINAL CAMERA SOLVE';
  $('set-solve').disabled = !!setOrPreset?.capture?.transforms;
  $('set-feat').disabled = !!setOrPreset?.capture?.transforms;
  setStartStyle(!(setOrPreset && setOrPreset.spaceId));
  if (setOrPreset && setOrPreset.spaceId && SHELL !== 'browser') {
    // a trained model of this benchmark is shared — View opens it, and the
    // published model's train PSNR joins the input facts once resolved
    const view = document.createElement('a');
    view.id = 'detail-view';
    view.className = 'btn btn-accent big btn-cta';
    view.href = `index.html?space=${encodeURIComponent(setOrPreset.spaceId)}`;
    view.textContent = 'View';
    document.querySelector('.startrow').prepend(view);
    import('./share.js').then(({ resolveShare }) => resolveShare(setOrPreset.spaceId))
      .then((sh) => {
        if (S.detailFacts !== facts || $('detail').hidden) return;
        const sj = sh.splatjs || {};
        facts.splats = sj.splats;
        facts.dB = sj.psnrTest ? sj.psnrTest.psnr : sj.psnrTrain;
        facts.dBLabel = sj.psnrTest ? 'holdout psnr' : 'train psnr';
        paintDetailStats();
      }).catch(() => {});
  }
  $('btn-go').disabled = !!S.noGpu;
  $('start').hidden = true;
  $('detail').hidden = false;
  $('btn-new').hidden = false;   // the setup card is not home
  // the card is a navigable UI state: Back must close IT, not leave the app
  if (!(history.state && history.state.sj === 'detail')) {
    history.pushState({ sj: 'detail' }, '');
  }
}

// Back closes pushed UI layers (About sheet, the viewer, the detail card)
// instead of navigating away — history walks the same steps the visitor sees
addEventListener('popstate', () => {
  // snapshot first: closing the overlays hides the sheets, and the pushed
  // states below still have to walk in the same order as before
  const aboutOpen = !$('about').hidden;
  const viewerOpen = S._viewerOpen;
  const detailOpen = !$('detail').hidden;
  closeOverlays();   // modals and score graphs never survive a Back
  if (aboutOpen) return;
  if (viewerOpen) { closeViewerToHome(); return; }
  if (detailOpen) { closeDetailCard(); return; }
  // back during a running solve/training: leave the same way the header
  // Back does — secure it, then a page navigation home (the wall is never
  // rendered OVER a live scene). No confirm: the checkpoint is the save.
  if ((S.state === 'prep' || S.state === 'train') && $('start').hidden) {
    if (S.state === 'prep') {
      S._leaving = true; location.href = 'index.html?back=1';
    } else {
      checkpointRun('back').catch(() => null).then((saved) => {
        if (!saved) {
          flash('Could not save this model yet. Training is still open; free storage and try Back again.', 9000);
          return;
        }
        S._leaving = true; location.href = 'index.html?back=1';
      });
    }
  }
});

/** The setup card's ‹ and ×: route through history when the card owns an
 *  entry so the phone's Back gesture stays in sync. */
function detailClose() {
  if (history.state && history.state.sj === 'detail') { history.back(); return; }
  closeDetailCard();
}

/** Close the setup card — back INTO the shared scene its Train came from,
 *  otherwise home to the wall. */
function closeDetailCard() {
  closeOverlays();   // a subset (failcard) used to be removed here
  S.pendingShare = null;
  $('detail').hidden = true;
  if (S.state === 'ready' && S._fromSpace) {
    const id = S._fromSpace;
    S._fromSpace = null;
    restoreShared(id);
    return;
  }
  showHome();   // fresh — a capture picked moments ago must show its tile
}

/** Tear the restored viewer down and land on the wall — the SPA counterpart
 *  of the old full-page navigation home. */
async function closeViewerToHome() {
  // Stop & keep enters the orbit viewer before the GPU readback has finished.
  // Never tear down that session (or bump S.gen) until the model checkpoint is
  // actually durable; this was the race that produced an empty Home.
  if (S.state === 'done' && S._securing) {
    flash('Saving the finished model …', 60000);
    const saved = await S._securing.catch(() => false);
    if (!saved) return;
  }
  stopTour();
  try { S.session && S.session.dispose(); } catch (e) { /* view-only facade */ }
  document.getElementById('cv-model')?.remove();
  document.getElementById('share-hero')?.remove();
  closeOverlays();   // modals and score graphs stay on the screen that opened them
  gpuCanvas = null;
  S.gen++;
  S.session = null; S.share = null; S.restored = null; S._localRun = null; S._securing = null;
  S.runId = null; S._resumeRunId = null;
  S._viewerOpen = false;
  S._fromSpace = null;
  S._thumbsUrl = null; S._thumbPack = null;
  S.plyBlob = null; S.sogBlob = null;
  S.state = 'ready'; S.preset = null; S.photos = [];
  syncTrainingVisualsControl();
  S.scene = null; S.tour = null;
  $('strip').innerHTML = '';
  $('controls').hidden = true;
  dock('');
  showHome(); // fresh tiles — runs may have finished or been added since
}

/** The creation's recon json (plain or store-zipped) — the photographs and
 *  cameras a "train this yourself" run needs. */
async function fetchShareRecon(it) {
  const { unzipStoreAsync } = await import('./session_io.js');
  const r = await fetch(it.splatjs.reconUrl);
  if (!r.ok) throw new Error('recon unavailable');
  const rb = new Uint8Array(await r.arrayBuffer());
  if (rb[0] === 0x50 && rb[1] === 0x4b) {
    return JSON.parse(new TextDecoder().decode((await unzipStoreAsync(rb)).get('recon.json')));
  }
  return JSON.parse(new TextDecoder().decode(rb));
}

/** One creation card, post-style: hero, badge, title, the FULL description
 *  and the key numbers. The whole card is the View action — clicking loads
 *  the creation directly (Train lives in the viewer). */
/** One field in the share sheet's chrome; resolves to the trimmed value, or
 *  null when the visitor backs out. (A browser prompt() would block the page
 *  and reads as a phishing box over someone else's scene.) */
function textCard(heading, value, { action = 'Save', max = 80 } = {}) {
  document.getElementById('upcard')?.remove();
  return new Promise((resolve) => {
    const card = document.createElement('div');
    card.className = 'upcard';
    card.id = 'upcard';
    card.innerHTML = `
      <b>${esc(heading)}</b>
      <input id="tc-val" type="text" spellcheck="false" maxlength="${max}">
      <div class="upcard-row">
        <button class="btn btn-quiet" id="tc-cancel">Cancel</button>
        <button class="btn btn-accent" id="tc-go">${esc(action)}</button>
      </div>`;
    $('stage').appendChild(card);
    const input = card.querySelector('#tc-val');
    input.value = value || '';
    input.focus();
    input.select();
    const done = (v) => { card.remove(); resolve(v); };
    card.querySelector('#tc-cancel').addEventListener('click', () => done(null));
    card.querySelector('#tc-go').addEventListener('click', () => done(input.value.trim() || null));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') done(input.value.trim() || null);
      if (e.key === 'Escape') done(null);
    });
  });
}

/** Rebuild the local library progressively. The main-branch behavior painted
 * IndexedDB immediately; Android archive discovery must enhance that wall,
 * never block every capture/model/training tile behind one slow source. */
let wallMountToken = 0;
async function mountWall() {
  const host = $('gallery');
  if (!host) return;
  const token = ++wallMountToken;
  const heading = document.createElement('p');
  heading.className = 'eyebrow';
  heading.textContent = 'SAVED ON THIS DEVICE';
  const grid = document.createElement('div');
  grid.className = 'galgrid';
  const slots = { scans: [], files: [], capture: [], local: [] };
  host.replaceChildren();
  host.hidden = true;

  const paint = () => {
    if (token !== wallMountToken) return;
    // Match main: IndexedDB run records are the model cards (thumbnail,
    // rename, saved poses, in-place continuation). Android run files are a
    // recovery fallback only when that same run record is absent.
    const localIds = new Set(slots.local.map((tile) => tile.dataset.runId).filter(Boolean));
    const recoveryFiles = slots.files.filter((tile) => !localIds.has(tile.dataset.runId));
    const tiles = [...slots.scans, ...slots.capture, ...slots.local, ...recoveryFiles];
    grid.replaceChildren(...tiles);
    if (!tiles.length) { host.hidden = true; return; }
    if (!heading.isConnected) host.append(heading, grid);
    host.hidden = false;
  };
  const load = (key, task, fallback = []) => {
    Promise.resolve(task).then((value) => {
      slots[key] = Array.isArray(value) ? value : (value ? [value] : []);
      paint();
    }, (e) => {
      console.warn(`[home] ${key} library unavailable`, e);
      slots[key] = fallback;
      paint();
    });
  };

  const native = globalThis.AndroidARBridge;
  if (native && typeof native.listHomeLibrary === 'function') {
    // Android has ONE catalog rooted in app data. Captures are canonical
    // image+pose archives; runs are model checkpoints referencing capture IDs.
    // IndexedDB is not a second Android Home library.
    let catalog = { captures: [], runs: [] };
    try { catalog = JSON.parse(native.listHomeLibrary()); } catch (e) {
      console.warn('[home] native catalog unavailable', e);
    }
    load('scans', nativeCaptureTiles(catalog.captures));
    load('files', androidRunTiles(catalog.runs));
    // Main stores model checkpoints in IndexedDB. Keep those model records in
    // the union (deduped by run ID); they reference the native capture archive
    // and are not a second image/pose source.
    load('local', localRunTiles(catalog.runs, catalog.captures));
    migrateLegacyNativeCapture().then((moved) => {
      if (moved && !legacyCaptureRemounted && token === wallMountToken) {
        legacyCaptureRemounted = true; mountWall();
      }
    });
  } else {
    // Browser/main behavior: its origin-local IndexedDB is the sole library.
    load('capture', lastCaptureTile());
    load('local', localRunTiles());
  }
}

window.addEventListener('pageshow', (e) => {
  if (e.persisted && document.getElementById('gallery')) mountWall().catch(() => {});
});

/** From the viewer into the studio: the shared creation's photographs become
 *  the training input — the moment of "I want to make this myself". */
function trainFromShare() {
  const rj = S.restored && S.restored.reconJson;
  if (!rj || !rj.source || !rj.source.urls || !rj.source.urls.every(Boolean)) return;
  stopTour();
  // remember where we came from: ‹ and × on the setup card return INTO this
  // scene instead of dumping the visitor on the front page
  S._fromSpace = S.share ? S.share.id : null;
  // the SHARE's title names the set — the recon carries the original
  // creator's local set name ("Your photos"), meaningless to this visitor
  const name = (S.share && S.share.title) || rj.name || 'Shared sample';
  try { S.session.dispose(); } catch (e) {}
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  S.gen++;
  S.session = null; S.share = null; S.restored = null;
  S._thumbsUrl = null; S._thumbPack = null;
  S._viewerOpen = false;
  S.plyBlob = null; S.sogBlob = null;
  S.state = 'ready';
  syncTrainingVisualsControl();
  S.preset = { id: '__sample', name };
  S.photos = rj.source.names.map((n, i) => ({ url: rj.source.urls[i], name: n }));
  S.sel = 0; S.atFrame = -1; S.fadeTo = 0;
  vp.lock = null; vp.pose = null; vp.scene = null; S.scene = null;
  renderControls();
  buildStrip(true);   // the card covers the strip — thumbs wait for the run
  $('set-desc').innerHTML = `<b>${esc(name)}</b> — ${S.photos.length} photographs from this creation, ready to train. The gear holds quality settings.`;
  $('set-desc').hidden = false;
  setStartStyle(true);
  $('btn-go').disabled = !!S.noGpu;
  $('btn-settings').disabled = false;   // open() normally arms the gear — this path skips it
  closeOverlays();   // the viewer's sheets and cards stay behind
  $('btn-new').hidden = false;   // the setup card is not home
  if (WALL_FIRST) {
    $('start').hidden = false;
  } else {
    // classic keeps Start on the focused detail card
    document.getElementById('detail-view')?.remove();
    const hero = $('detail-hero');
    hero.src = S.photos[0].url;
    hero.hidden = false;
    $('start').hidden = true;
    $('detail').hidden = false;
  }
  bmp(S.photos[0].url);
  flash('Photographs loaded — press Start training.', 6000);
}

/** The error map needs pixels of the render. The live WebGPU canvas can read
 *  back blank after presentation, so render the frame's exact camera into a
 *  scratch canvas and snapshot it right behind the fence — at training
 *  resolution, which also makes the comparison resolution-fair. */
async function ensureErrRender(key) {
  if ((S._errRender && S._errRender.key === key) || S._errRenderBusy) return;
  if (S.atFrame < 0) return;
  const cam = S.scene.cams[S.atFrame];
  if (!cam || cam.ci < 0) return;
  // remote training owns the view (and a transferred canvas has no
  // main-side pixels to snapshot) — this scratch flow is local-only
  if (!S.session || !S.session.trainer || !S.session.trainer.device) return;
  S._errRenderBusy = true;
  const gen = S.gen;
  try {
    const ses = S.session;
    const meta = ses.trainer.camMeta[cam.ci];
    const cv = (S._errScratch ??= document.createElement('canvas'));
    cv.width = meta.w; cv.height = meta.h;
    quietView(ses.view.attach(cv));
    quietView(ses.view.lookThrough(cam.ci));
    quietView(ses.view.renderNow(true));
    await ses.trainer.device.queue.onSubmittedWorkDone();
    if (S.gen !== gen) return;
    const snap = (S._errSnap ??= document.createElement('canvas'));
    snap.width = meta.w; snap.height = meta.h;
    snap.getContext('2d').drawImage(cv, 0, 0);
    S._errRender = { key, canvas: snap };
  } catch (e) {
    console.error(e);
  } finally {
    S._errRenderBusy = false;
    // hand the view back to the stage and force a fresh pose render
    if (S.gen === gen && gpuCanvas && S.session) {
      quietView(S.session.view.attach(gpuCanvas));
      S._viewKey = '';
    }
  }
}

/** put the camera exactly on a frame's pose and lay its photograph over the model */
/** In a restorable view (share / ?model=) the address mirrors the frame
 *  focus and compare mode — a refresh lands exactly here. */
function syncFrameUrl() {
  if (S.state !== 'done') return;
  const u = new URL(location.href);
  if (!u.searchParams.has('space') && !u.searchParams.has('model')) return;
  if (S.atFrame >= 0) {
    u.searchParams.set('frame', String(S.sel));
    u.searchParams.set('cmp', S.compare);
  } else {
    u.searchParams.delete('frame');
    u.searchParams.delete('cmp');
  }
  history.replaceState(null, '', u);
}

function goToFrame(i) {
  stopTour();
  const cam = S.scene.cams[i];
  if (!cam || !cam.R) { flash('That frame was never placed — there is no viewpoint to jump to.'); return; }
  if (cam.pano) {
    // a 360 station: fly to where it was taken — no photo overlay
    S.sel = i;
    vp.lock = null;
    vp.syncTo(cam);
    S.atFrame = -1; S.fadeTo = 0;
    $('stage').dataset.cursor = 'grab';
    renderControls(); paintStrip();
    return;
  }
  S.sel = i; S.atFrame = i;
  // gravity-leveled frame lock: the model keeps its upright orientation
  // (position and aim move to the photo, roll levels to gravity) and the
  // photograph turns to register — a cardinal snap, exact pixels. Untilted
  // frames keep the raw lock, pixel-identical to before.
  const snap = snapRollAngle(rollAngleToUp(cam.R, vp.up));
  let roll = null;
  if (snap) {
    const lv = rerollPose(cam.R, camCentre(cam), snap);
    const ex = exifIntrinsics(cam.w, cam.h, cam.cx, cam.cy, -snap);
    vp.lock = { R: lv.R, t: lv.t, f: cam.f, w: ex.w, h: ex.h, cx: ex.cx, cy: ex.cy };
    roll = { snap, camW: cam.w, camH: cam.h };
  } else {
    vp.lock = cam;
  }
  bmp(cam.url).then((b) => {
    if (b && S.scene.cams[S.atFrame]?.url === cam.url) dev.setBitmap(b, cam.url, roll);
  });
  S.fadeTo = 1;
  S.loupe.x = $('stage').clientWidth / 2;
  S.loupe.y = $('stage').clientHeight / 2;
  $('stage').dataset.cursor = cursorFor(S.compare);
  renderControls(); paintStrip();
  syncFrameUrl();
}

/** a drag pulls the camera off the frame — same position, same lens, now free */
function leaveFrame() {
  if (S.atFrame < 0) return;
  const cam = S.scene.cams[S.atFrame];
  vp.freeF = cam.f * Math.min(vp.w / cam.w, vp.h / cam.h);
  vp.lock = null;
  vp.syncTo(cam);
  S.atFrame = -1;
  S.fadeTo = 0;
  $('stage').dataset.cursor = 'grab';
  renderControls();
  syncFrameUrl();
}

function select(i) {
  if (!S.photos.length) return;
  S.sel = (i + S.photos.length) % S.photos.length;
  $('strip-scroll')?.children[S.sel]?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  if ((S.state === 'train' || S.state === 'done') && S.scene) goToFrame(S.sel);
  paintStrip(); renderControls();
  if (!$('details').hidden) renderDetails();
}

function wireStage() {
  const st = $('stage');
  // Touch-hold with a quiet window: while visuals show, the GPU belongs to
  // the view from the FIRST model touch until 500ms after the LAST one —
  // every new touch refreshes the window, so training resumes only after
  // 500ms of no interaction and never re-queues a backlog between quick
  // successive touches. Touches on UI (graph, controls hub, buttons) don't
  // count — model touches only. Up/cancel ride the window so a release
  // outside the stage still counts; a user-pressed pause clears the flag in
  // toggleTrain, so the window never resumes over it.
  const held = new Set();
  let resumeT = 0;
  const armResume = () => {
    clearTimeout(resumeT);
    resumeT = setTimeout(() => {
      resumeT = 0;
      if (!held.size && S._dragPaused) {
        S._dragPaused = false;
        if (S.state === 'train') S.session.start();
      }
    }, 500);
  };
  const noteModelTouch = () => {
    if (!S.session) { S._dragPaused = false; return; }
    clearTimeout(resumeT); resumeT = 0;
    if (S.state === 'train' && S.trainVisuals && S.session.training && !S._dragPaused) {
      S._dragPaused = true;
      S.session.pause();
    }
  };
  // the graph and the hub live inside the stage (the expanded chart even
  // covers it) — only canvas/model touches may hold the GPU
  const onModel = (e) => !e.target.closest('#controls, #hud, #start, .chartwrap, button');
  st.addEventListener('pointerdown', (e) => {
    if (!onModel(e)) return;
    held.add(e.pointerId);
    noteModelTouch();
  });
  window.addEventListener('pointerup', (e) => {
    if (held.delete(e.pointerId) && !held.size && S._dragPaused) armResume();
  });
  window.addEventListener('pointercancel', (e) => {
    if (held.delete(e.pointerId) && !held.size && S._dragPaused) armResume();
  });
  // a wheel tick adjusts the model without a touch — same hold, same window
  st.addEventListener('wheel', (e) => {
    if (!onModel(e)) return;
    noteModelTouch();
    if (S._dragPaused) armResume();
  }, { passive: true });
  st.addEventListener('pointermove', (e) => {
    if (S.atFrame < 0) return;
    const r = st.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (S.compare === 'loupe') { S.loupe.x = x; S.loupe.y = y; }
    if (S.compare === 'swipe' && S.rect) S.swipe = clamp((x - S.rect.x) / S.rect.w, 0, 1);
  });
  st.addEventListener('wheel', (e) => {
    if (S.atFrame < 0 || S.compare !== 'loupe') return;
    e.preventDefault();
    S.loupe.r = clamp(S.loupe.r - e.deltaY * .12, 40, 260);
  }, { passive: false });
}

// ── filmstrip ───────────────────────────────────────────────────────────────
function buildStrip(deferPhotos = false) {
  const strip = $('strip');
  strip.innerHTML = '<div class="strip-scroll" id="strip-scroll"></div>';
  const sc = $('strip-scroll');
  dragScroll(sc);
  const fill = async (el) => {
    if (el.dataset.filled) return;
    el.dataset.filled = '1';
    const i = +el.dataset.i;
    const b = (await thumbBmp(i)) || (await bmp(S.photos[i].url, 140));
    if (!b) return;
    const cv = document.createElement('canvas');
    // equirect panoramas crop to the card: the centre 4:3 window (the
    // pano's forward direction) instead of a squeezed 2:1 sliver
    const wide = b.width / b.height > 1.9;
    const sh = wide ? b.height * 0.72 : b.height;
    const sw = wide ? Math.min(b.width, sh * (4 / 3)) : b.width;
    const sx = (b.width - sw) / 2, sy = (b.height - sh) / 2;
    cv.width = Math.max(1, Math.round(sw));
    cv.height = Math.max(1, Math.round(sh));
    cv.getContext('2d').drawImage(b, sx, sy, sw, sh, 0, 0, cv.width, cv.height);
    el.querySelector('.ph')?.replaceWith(
      Object.assign(new Image(), { src: cv.toDataURL('image/jpeg', .7) }));
  };
  // viewing a share: the tiles stay placeholders until touched — a shared
  // creation must not pull hundreds of full-size training photos on load
  const io = deferPhotos ? null : new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      fill(e.target);
    });
  }, { root: sc, rootMargin: '250px' });

  S.photos.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'frame';
    b.dataset.i = i;
    b.innerHTML = `<div class="ph"></div>
      <span class="frame-tag" hidden></span><div class="frame-bar"><i></i></div>`;
    b.addEventListener('click', () => { fill(b); select(i); });
    sc.appendChild(b);
    if (io) io.observe(b);
  });
  paintStrip();
}

/** A shared creation's packed strip thumbnails: the zip is fetched once,
 *  lazily, and each card decodes its small JPEG — no full photographs are
 *  pulled just to draw 140px tiles. Null (fallback) when the share predates
 *  the pack or the fetch fails. */
async function thumbBmp(i) {
  if (!S._thumbsUrl) return null;
  try {
    S._thumbPack ??= (async () => {
      const r = await fetch(S._thumbsUrl);
      if (!r.ok) throw new Error(`thumbs fetch failed (${r.status})`);
      const { unzipStoreAsync } = await import('./session_io.js');
      return unzipStoreAsync(new Uint8Array(await r.arrayBuffer()));
    })();
    const data = (await S._thumbPack).get(`${i}.jpg`);
    return data ? await createImageBitmap(new Blob([data], { type: 'image/jpeg' })) : null;
  } catch {
    S._thumbsUrl = null;   // broken pack: every card falls back to the photo
    return null;
  }
}

function paintStrip() {
  const sc = $('strip-scroll');
  if (!sc) return;
  const active = (S.state === 'train' && S.session && S.session.training && S.scene)
    ? (S.scene.cams.find((c) => c.ci === S.session.activeCam) || {}).i : -1;
  S.photos.forEach((p, i) => {
    const b = sc.children[i];
    if (!b) return;
    const c = S.scene ? S.scene.cams[i] : null;
    b.dataset.sel = i === S.sel ? '1' : '0';
    b.dataset.live = i === active ? '1' : '0';
    b.dataset.state = c ? c.state : 'placed';
    const tag = b.querySelector('.frame-tag');
    const t = c && c.state === 'holdout' ? 'held' : c && c.state === 'unplaced' ? 'out' : null;
    tag.hidden = !t;
    if (t) { tag.dataset.t = c.state; tag.textContent = t; }
    const bar = b.querySelector('.frame-bar i');
    const score = c && (c.psnr != null ? c.psnr
      : (S.state !== 'ready' && c.state !== 'unplaced'
        ? (c.state === 'holdout' ? S.psnrHold : S.psnrTrain) : null));
    bar.style.width = score != null ? `${clamp((score - 12) / 22, 0, 1) * 100}%` : '0%';
    bar.style.background = c && c.state === 'holdout' ? '#f2a03f' : '#2fd4c1';
  });
}

// ── dock ────────────────────────────────────────────────────────────────────
function dock(kind) {
  const d = $('dock');
  // Hide-visuals reparents the score graph into stage. A recovered trainer
  // creates a new dock graph; remove every prior direct-stage graph first or
  // both canvases remain visible (the large translucent overlay in reports).
  document.querySelectorAll('#stage > .chartwrap, #stage > .chartwrap-expanded')
    .forEach((el) => el.remove());
  d.className = 'dock' + (kind ? ` dock-${kind}` : '');
  if (!kind) {
    d.innerHTML = '';
    chart = null;
    return;
  }

  if (kind === 'prep') {
    // the stage sequence IS the header: the current beat reads as the title,
    // the others wait in line around it
    d.innerHTML = `
      <div>
        <div class="prep-stages" id="p-steps">${BEATS.map((s, i) =>
          `<span data-k="${i}">${s.label}</span>`).join('')}</div>
        <div class="prep-sub" id="p-sub">—</div>
        <div class="prep-meter"><i id="p-bar" style="width:0%"></i><span class="prep-live" aria-hidden="true"></span></div>
      </div>`;
    return;
  }

  if (kind === 'train') {
    d.innerHTML = `
      <div class="tcontrols">
        <span class="playwrap"><button class="play" id="t-play" data-state="pause"><svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z"/></svg></button><button class="tbtn-sm" id="t-finish" hidden title="End the run here — the model is kept as it is and ready to export">Stop &amp; keep</button><button class="tbtn-sm" id="t-prune" title="Prune the model down or trigger a growth pass, then keep training">Prune/Grow</button></span>
        <div class="tmeta">
          <span class="tmeta-1"><span id="t-iter">${pctOf(S.iter)}</span></span>
          <span class="tmeta-2">ETA <span id="t-eta">${etaText()}</span></span>
          <span class="tmeta-gpu" title="Share of wall time training runs the GPU — live, no restart needed"><label for="t-gpu">GPU</label><input id="t-gpu" type="range" min="10" max="100" step="5" value="${S.settings.gpu ?? 100}"><span id="t-gpu-v">${S.settings.gpu ?? 100}%</span></span>
        </div>
      </div>
      <div class="chartwrap"><canvas id="chart"></canvas><div class="chart-tip" id="chart-tip" hidden></div></div>
      <div class="tscores">
        <div class="score" data-tone="accent">
          <div class="score-1"><span class="score-v"><span id="t-ptrain">${formatTrainingScore(shownScore('train'))}</span><small class="unit" id="t-punit">${S.scoreScale === 'similarity' ? '%' : 'dB'}</small></span></div>
          <div class="score-sub"><span id="t-splats">${S.splats ? fmt(S.splats) : '—'}</span> splats</div>
          <button class="chart-scale score-scale" id="chart-scale" type="button">Use %</button>
          <span class="tmeta-grow" id="t-grow">&nbsp;</span>
        </div>
        ${S.session && S.session.holdout >= 0 ? `<div class="score" data-tone="alt"><div class="score-1"><span class="score-v"><span id="t-phold">${formatTrainingScore(shownScore('hold'))}</span><small class="unit" id="t-hunit">${S.scoreScale === 'similarity' ? '%' : 'dB'}</small></span></div><div class="score-sub">held out</div></div>` : ''}
      </div>`;
    $('t-play').addEventListener('click', toggleTrain);
    $('t-finish').addEventListener('click', async () => {
      $('t-finish').disabled = true;
      await S.session.finish();   // emits train-complete -> finish()
    });
    $('t-prune').addEventListener('click', showPrunePicker);
    // the GPU usage slider, mid-training edition: the same persisted knob
    // as the settings panel, mirrored both ways, live on every input
    const tGpu = $('t-gpu');
    if (tGpu) {
      tGpu.addEventListener('input', () => {
        const v = Math.min(100, Math.max(10, parseInt(tGpu.value, 10) || 100));
        S.settings.gpu = v;
        $('t-gpu-v').textContent = `${tGpu.value}%`;
        if (S.session) S.session.opts.gpuUsage = v / 100;
        updateHeadlessGpu(v);
        const panel = $('set-gpu');
        if (panel) { panel.value = String(v); $('set-gpu-v').textContent = `${v}%`; }
      });
      tGpu.addEventListener('change', () => saveSettings());
    }
    chart = new Chart($('chart'), { onHover: chartTip });
    chart.scale = S.scoreScale;
    chart.maxIter = S.maxIters;
    $('chart-scale').addEventListener('click', toggleScoreScale);
    syncScoreScaleControls();
    chart.resize();
  }
}

function syncScoreScaleControls() {
  const next = S.scoreScale === 'db' ? 'Use %' : 'Use dB';
  const title = S.scoreScale === 'db'
    ? 'Switch graph to percentage'
    : 'Switch graph to PSNR in decibels';
  for (const id of ['chart-scale', 'd-chart-scale']) {
    const button = $(id);
    if (!button) continue;
    button.textContent = next;
    button.title = title;
    button.setAttribute('aria-label', title);
  }
}

function toggleScoreScale() {
  S.scoreScale = S.scoreScale === 'db' ? 'similarity' : 'db';
  if (chart) { chart.scale = S.scoreScale; chart.draw(); }
  if (dchart) { dchart.scale = S.scoreScale; dchart.draw(); }
  syncScoreScaleControls();
  renderTrainingScore();
}

function chartTip(h) {
  const tip = $('chart-tip');
  if (!tip) return;
  if (!h) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.innerHTML = `${fmt(h.iter)} · <b style="color:#2fd4c1">${h.train.toFixed(1)} ${h.unit}</b>` +
    (h.hold != null ? ` / <b style="color:#f2a03f">${h.hold.toFixed(1)} ${h.unit}</b>` : '') +
    (h.event ? `<br><span style="color:#93a1a0">${h.event}</span>` : '');
  // the readout rides the cursor INSIDE the plot: pinned to the top edge it
  // hung past the stage in the expanded graph and hid behind the dock.
  // Clamped sideways, floating 70px clear of the cursor (below it near the top).
  const wrap = tip.parentElement;
  const W = wrap ? wrap.clientWidth : 0, H = wrap ? wrap.clientHeight : 0;
  if (!W || !H) { tip.style.left = `${h.xPct}%`; tip.style.top = '4px'; return; }
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  const x = Math.max(tw / 2 + 4, Math.min(W - tw / 2 - 4, (h.xPct / 100) * W));
  const y = (Math.max(0, Math.min(100, h.yPct ?? 0)) / 100) * H;
  tip.style.left = `${(x / W) * 100}%`;
  tip.style.top = `${(y / H) * 100}%`;
  tip.style.transform = y > th + 86 ? 'translate(-50%, calc(-100% - 70px))' : 'translate(-50%, 70px)';
}

// ── flash ───────────────────────────────────────────────────────────────────
function flash(msg, ms = 2800, links = []) {
  S.flash = { msg: String(msg), links, until: performance.now() + ms };
}

function renderHud() {
  const hud = $('hud');
  const key = S.flash ? JSON.stringify(S.flash) : '';
  if (hud.dataset.k === key) return;
  hud.dataset.k = key;
  const row = document.createElement('div');
  row.className = 'chip-row';
  if (S.flash) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.dataset.tone = 'accent';
    chip.textContent = S.flash.msg;
    for (const link of S.flash.links) {
      const a = document.createElement('a');
      a.textContent = link.label;
      if (link.copy != null) {
        a.href = '#';
        a.addEventListener('click', async (e) => {
          e.preventDefault();
          try { await navigator.clipboard.writeText(link.copy); a.textContent = 'Copied'; }
          catch { a.textContent = 'Copy failed'; }
        });
      } else {
        let url;
        try { url = new URL(link.href, location.href); } catch { continue; }
        if (url.protocol !== 'http:' && url.protocol !== 'https:') continue;
        a.href = url.href;
        if (link.blank) { a.target = '_blank'; a.rel = 'noopener'; }
      }
      chip.append(' · ', a);
    }
    row.appendChild(chip);
  }
  hud.replaceChildren(row);
}

// ── Android foreground-service handoff ──────────────────────────────────────
// This Activity trains locally (live session, live viewer) and stages
// a run directory under files/runs/<runId>/ from the first cycle — manifest,
// photos and an exact transition checkpoint. When the Activity backgrounds (or
// is swiped away), the :training foreground service resumes that checkpoint in
// its headless WebView; the reopened
// app monitors (snapshot viewer + progress when Show-visuals is on), adopts
// the finish, or recovers. File shapes + cadence live in ./headless.js (shared
// with the driver, unit-pinned); this side only stages, checkpoints, watches.
//
// Every entry here is best-effort and bridge-gated: without the files-dir
// bridge (desktop, old app builds) each function returns silently and the run
// continues exactly as without the feature.
/** The files-dir bridge (MainActivity, new enough to stage runs). */
const headlessBridge = () => {
  const b = globalThis.AndroidARBridge;
  return b && typeof b.writeRunJson === 'function' && typeof b.writeRunChunk === 'function' ? b : null;
};
/** A files-dir read must never own the Start button forever. Android's asset
 *  handler is local, so seconds of silence means a wedged request, not useful
 *  evidence that another trainer exists. */
async function headlessFetch(url, timeoutMs = 2500) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try { return await fetch(url, { signal: ctl.signal }); }
  finally { clearTimeout(timer); }
}

/** True when a run directory claims a live run (staging/booting/active/paused). */
async function headlessMarkerLive() {
  if (!headlessBridge()) return false;
  try {
    const r = await headlessFetch(Headless.activeUrl());
    if (!r.ok) return false;
    const { runId } = await r.json();
    if (!Headless.validRunId(runId)) return false;
    const st = await headlessFetch(Headless.fresh(Headless.runUrl(runId, 'status.json')));
    if (!st.ok) return false;
    const status = await st.json();
    if (!Headless.isLiveStatus(status)) return false;
    // This guard is reached only by a new Start request. An activity-owned
    // live marker cannot own GPU work here (this page is still in setup), so
    // it is debris from a failed/old local run. Do not let it brick Start.
    if (status.owner === 'activity' && S.state === 'ready') {
      try {
        headlessBridge().writeRunJson(runId, 'status.json', JSON.stringify({
          ...status, status: 'dismissed', at: Date.now(),
          error: 'the previous local trainer was deliberately replaced',
        }));
        headlessBridge().clearActiveRun(runId);
      } catch { /* the new staging pass will replace an un-clearable marker */ }
      return false;
    }
    // a live run owned by a service that answers no ping died with its
    // process (killed, crashed, updated): release it to recovery instead of
    // blocking every new start behind a run that will never finish
    if (status.owner === 'service' && !(await headlessServiceAlive(runId))) {
      try {
        headlessBridge().writeRunJson(runId, 'status.json', JSON.stringify({
          ...status, status: 'interrupted', at: Date.now(),
          error: 'the background trainer was gone when a new run started',
        }));
      } catch { /* best-effort: the new run's own staging overwrites the marker */ }
      return false;
    }
    return true;
  } catch { return false; }
}

/** Liveness of a service-owned run: a fresh heartbeat proves it outright; a
 *  quiet one gets a ping round-trip (answered from Java, which no page
 *  throttle can silence) before it is declared dead. Conservative throughout:
 *  any doubt blocks the new start, as before. */
async function headlessServiceAlive(runId) {
  const b = headlessBridge();
  try {
    const s = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'service.json')))).json().catch(() => null);
    if (Headless.heartbeatFresh(s)) return true;
  } catch { /* fall through to the ping */ }
  if (!b || typeof b.headlessCommand !== 'function') return true;
  try { b.headlessCommand('headless-ping', runId); } catch { return true; }
  await new Promise((r) => setTimeout(r, 1200));
  try {
    const s = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'service.json')))).json();
    return Headless.heartbeatFresh(s);
  } catch { return false; }
}
/** Keep the staged service recipe in sync with the live GPU duty slider.
 * The service reads this manifest when it boots after backgrounding. */
function updateHeadlessGpu(value) {
  const b = headlessBridge();
  const v = Math.min(100, Math.max(10, Math.round(+value || 100)));
  if (!b || !S._hlManifest || S._hlManifest.runId !== S.runId || S._hlStaged !== S.runId) return;
  S._hlManifest = {
    ...S._hlManifest, gpu: v,
    settings: { ...(S._hlManifest.settings || {}), gpu: v },
  };
  try { b.writeRunJson(S.runId, 'manifest.json', JSON.stringify(S._hlManifest)); }
  catch (e) { console.warn('[headless] GPU duty update failed', e); }
}

/** Stage the handoff for this run: manifest + marker now, photos in the
 *  background (native captures retain their disk files, everything else
 *  uploads its bytes), status flips to live when the photos land. */
function stageHeadlessRun() {
  const b = headlessBridge();
  // the avatar route never stages (its post-training stages have no headless twin)
  if (!b || !S.runId || S.avatar || S._hlStaged) return;
  S._hlStaged = S.runId;
  S._hlPhotos = false; S._hlCkptBusy = false;
  S._hlStagePromise = (async () => {
    const runId = S.runId, gen = S.gen;
    try {
      const files = (S.loadedFiles || []).filter((f) => f && f.name);
      if (files.length < 3) throw new Error('no photos in hand');
      const names = files.map((f) => f.name);
      if (!names.every(Headless.validName)) throw new Error('photo names outside the handoff set');
      const cap = S.preset && S.preset.capture;
      const native = !!(cap && cap.metadata && cap.metadata.nativeARCore &&
        cap.metadata.nativeCaptureId && files.every((f) => f.url));
      const photoSource = native ? 'urls' : 'files';
      const manifest = Headless.buildManifest({
        runId, name: (S._localRun && S._localRun.name) || (S.preset && S.preset.name) || 'model',
        maxIters: S.maxIters, gpu: S.settings.gpu, srcKey: cap?.metadata?.nativeCaptureId || null,
        photos: names, photoSource,
        ...(native ? { photoUrls: files.map((f) => f.url) } : {}), settings: S.settings,
      });
      S._hlManifest = manifest;
      b.writeRunJson(runId, 'manifest.json', JSON.stringify(manifest));
      b.writeRunJson(runId, 'status.json', JSON.stringify(
        { status: 'staging', owner: 'activity', iter: 0, at: Date.now() }));
      b.setActiveRun(runId);
      // Never delete older run directories here. Their state.bin, recon and
      // retained photos are durable recovery sources backing Home tiles; only
      // an explicit user deletion may remove them.
      if (native) {
        // ARCore archives are the ground truth. The run stores references only;
        // both WebViews read the same immutable native JPEGs.
        b.writeRunJson(runId, 'photos.json', JSON.stringify(
          { names, source: 'urls', complete: true, at: Date.now() }));
      } else {
        // Browser/file-picker blobs have no native archive, so only those need
        // a service-readable copy in the run directory.
        await stagePhotoBlobs(runId, files);
      }
      if (S.gen !== gen || S.runId !== runId) return;   // the run moved on without us
      S._hlPhotos = true;
      headlessStatusWrite(S.session && S.session.training ? 'active' : 'paused');
    } catch (e) {
      // staging failed: the handoff stays disarmed (no photos.json, no live
      // status) and the run continues locally exactly as without the feature
      console.warn('[headless] staging failed — local training unaffected', e);
    }
  })().catch((e) => { console.warn('[headless] staging task failed', e); });
}
/** Upload photo bytes into the run dir (uploads, picker shots, video frames —
 *  everything without disk files). Chunked base64, sequential, throttled past
 *  the training loop's own hunger: one photo per macrotask at most. */
async function stagePhotoBlobs(runId, files) {
  const b = headlessBridge();
  const names = [];
  for (const f of files) {
    const bytes = new Uint8Array(await f.arrayBuffer());
    for (let o = 0; o < bytes.length; o += 262144) {
      b.writeRunChunk(runId, `photos/${f.name}`, Headless.encodeB64(bytes.subarray(o, o + 262144)),
        o + 262144 >= bytes.length);
    }
    names.push(f.name);
    await new Promise((r) => setTimeout(r, 0));
  }
  b.writeRunJson(runId, 'photos.json', JSON.stringify({ names, complete: true, at: Date.now() }));
}
/** Store a small run-owned artifact such as its Home preview. */
async function writeRunBlob(runId, name, blob) {
  const b = headlessBridge();
  if (!b || !Headless.validRunId(runId) || !Headless.validName(name) || !blob) return false;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  for (let o = 0; o < bytes.length; o += 262144) {
    if (!b.writeRunChunk(runId, name, Headless.encodeB64(bytes.subarray(o, o + 262144)),
      o + 262144 >= bytes.length)) return false;
  }
  return true;
}

/** Mirror an already-packed checkpoint into Android app data. This is the
 * durable model record used by Home; source photos remain references to their
 * canonical scan archive. */
async function writeDurableRunCheckpoint(state, recon, status = null) {
  const b = headlessBridge();
  if (S._hlStagePromise) await S._hlStagePromise;
  if (!b || !S._hlStaged || S.runId !== S._hlStaged || !S._hlPhotos) return false;
  const bytes = state instanceof Blob ? new Uint8Array(await state.arrayBuffer()) : new Uint8Array(state);
  for (let o = 0; o < bytes.length; o += 262144) {
    if (!b.writeRunChunk(S.runId, 'state.bin', Headless.encodeB64(bytes.subarray(o, o + 262144)),
      o + 262144 >= bytes.length)) return false;
  }
  if (!b.writeRunJson(S.runId, 'recon.json', JSON.stringify(recon))) return false;
  if (status) b.writeRunJson(S.runId, 'status.json', JSON.stringify(status));
  return true;
}

/** One-shot file checkpoint used for Activity→service ownership transfer. */
async function secureFileCheckpoint() {
  const b = headlessBridge();
  if (!b || !S._hlStaged || S.runId !== S._hlStaged || !S._hlPhotos) return false;
  if (!S.session || !S.session.trainer || S._hlCkptBusy) return false;
  S._hlCkptBusy = true;
  try {
    const { packState, buildReconJson } = await import('./session_io.js');
    const state = await packState(S.session);
    const recon = buildReconJson(S);
    if (!(await writeDurableRunCheckpoint(state, recon))) return false;
    // Mirror the same exact checkpoint into IndexedDB before relinquishing the
    // Activity. The run then remains a visible, tappable Home tile even if the
    // service process or active.json disappears.
    try {
      const { patchRun, saveRun } = await import('./store.js');
      const detailProgress = captureRunProgressDetails();
      const localPatch = {
        status: 'training', iter: S.session.trainer.iter | 0,
        splats: S.session.trainer.n | 0, state: new Blob([state]), recon,
        ckptAt: Date.now(), minutes: detailProgress.minutes,
        detailSolve: captureRunSolveDetails(), detailProgress, detailGpu: gpuFacts(false),
      };
      if (!(await patchRun(S.runId, localPatch))) {
        await saveRun({
          id: S.runId, name: (S.preset && S.preset.name) || 'Local Scene',
          createdAt: Date.now(), updatedAt: Date.now(), maxIters: S.maxIters,
          psnr: S.psnrHold ?? S.psnrTrain ?? null, frames: (S.loadedFiles || []).length,
          ownSrc: !!(S.preset && (S.preset.id === '__own' || S.preset.id === '__last')),
          ...localPatch,
        });
      }
      if (S._localRun && S._localRun.id === S.runId) Object.assign(S._localRun,
        { state: new Blob([state]), recon, detailProgress });
    } catch (e) { console.warn('[headless] IndexedDB checkpoint mirror failed', e); }
    headlessStatusWrite(S.session.training ? 'active' : 'paused', S.session.trainer.iter | 0);
    return true;
  } catch (e) {
    console.warn('[headless] file checkpoint failed', e);   // never disturb the run
    return false;
  } finally { S._hlCkptBusy = false; }
}
/** Best-effort status write for the staged run (pause/active/iter). */
function headlessStatusWrite(status, iter = null) {
  const b = headlessBridge();
  if (!b || !S._hlStaged || S.runId !== S._hlStaged) return;
  try {
    b.writeRunJson(S.runId, 'status.json', JSON.stringify({
      status, owner: 'activity',
      iter: iter ?? (S.session && S.session.trainer ? S.session.trainer.iter | 0 : 0),
      at: Date.now(),
    }));
  } catch { /* best-effort */ }
}
/** The Activity's liveness beat: the service refuses the takeover while this
 *  is fresh (a fast reopen + Continue-here must win over a booting twin). */
function headlessActivityBeat(now) {
  const b = headlessBridge();
  if (!b || !S._hlStaged || S.runId !== S._hlStaged) return;
  if (now - (S._hlBeatAt || 0) < 5000) return;
  S._hlBeatAt = now;
  try {
    b.writeRunJson(S.runId, 'activity.json', JSON.stringify({ alive: true, at: Date.now() }));
  } catch { /* best-effort */ }
}
/** Deliberate leave (Back): the IDB checkpoint is the resume path (Yours),
 *  so the files-dir handoff stands down instead of hijacking the fresh boot
 *  into recovery. Only ever clears the live marker, if one exists. */
async function headlessClearMarker() {
  const b = headlessBridge();
  if (!b) return;
  try {
    const r = await fetch(Headless.activeUrl());
    if (!r.ok) return;
    const { runId } = await r.json();
    if (Headless.validRunId(runId)) {
      try {
        b.writeRunJson(runId, 'status.json', JSON.stringify(
          { status: 'dismissed', owner: 'activity', iter: S.iter || 0, at: Date.now() }));
        b.clearActiveRun(runId);
      } catch { /* best-effort */ }
    }
  } catch { /* best-effort */ }
}
/** Release the marker (local finish, adopt, discard). Only clears when the
 *  marker still names this run — a newer run's claim is never touched. */
function headlessClear(runId = S.runId) {
  const b = headlessBridge();
  S._hlStaged = null; S._hlPhotos = false; S._hlManifest = null;
  if (!b || !runId) return;
  try { b.clearActiveRun(runId); } catch { /* best-effort */ }
}
/** Convert the file-backed reconstruction's score history into the same
 * IndexedDB detail shape used by local runs. */
function headlessProgressDetails(recon, manifest, status = {}) {
  const stats = (recon && recon.stats) || {};
  return {
    chart: Array.isArray(stats.chart) ? stats.chart.map((r) => Array.from(r || [])) : [],
    holds: Array.isArray(stats.holds) ? stats.holds.map((r) => Array.from(r || [])) : [],
    events: Array.isArray(stats.events) ? stats.events.map((e) => ({ ...e })) : [],
    perf: { frames: [], marks: [] }, metrics: [],
    settings: { ...((manifest && manifest.settings) || {}) },
    maxIters: (manifest && manifest.maxIters) || 0,
    iter: status.iter ?? recon.iter ?? 0,
    minutes: stats.minutes || 0,
    psnrTrain: status.psnr ?? stats.psnrTrain ?? null,
    psnrHold: stats.psnrHold ?? null,
  };
}

/** Mirror Android's durable run directory into the local library. This makes
 * a service checkpoint visible on Home even when IndexedDB was evicted before
 * handoff, and keeps a second independently discoverable copy of state.bin. */
async function mirrorHeadlessRun(runId, status) {
  const manifest = Headless.validateManifest(
    await (await fetch(Headless.fresh(Headless.runUrl(runId, 'manifest.json')))).json());
  const recon = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'recon.json')))).json();
  const stateBytes = new Uint8Array(
    await (await fetch(Headless.fresh(Headless.runUrl(runId, 'state.bin')))).arrayBuffer());
  const { parseState } = await import('./session_io.js');
  const parsed = parseState(stateBytes);
  recon.source = {
    preset: null, names: [...manifest.photos],
    urls: manifest.photoSource === 'urls'
      ? [...manifest.photoUrls]
      : manifest.photos.map((n) => Headless.photoUrl(runId, n)),
    ...(manifest.srcKey ? { captureId: manifest.srcKey } : {}),
  };
  let thumb = null;
  try {
    const r = await fetch(Headless.fresh(Headless.runUrl(runId, 'snapshot.png')));
    if (r.ok) thumb = await r.blob();
  } catch { /* optional */ }
  const detailProgress = headlessProgressDetails(recon, manifest, status);
  const rec = {
    id: runId, name: manifest.name,
    status: status.status === 'finished' ? 'finished' : 'training',
    createdAt: manifest.stagedAt || Date.now(), updatedAt: Date.now(),
    iter: parsed.iter, maxIters: manifest.maxIters,
    splats: status.splats ?? recon.splats ?? parsed.gaussians.n,
    psnr: status.psnr ?? recon.stats?.psnrTrain ?? null,
    frames: manifest.photos.length, ownSrc: !!manifest.srcKey,
    ...(manifest.srcKey ? { srcKey: `native:${manifest.srcKey}` } : {}),
    state: new Blob([stateBytes]), recon, detailProgress,
    ...(thumb ? { thumb } : {}),
  };
  try {
    const { saveRun, patchRun } = await import('./store.js');
    if (!(await patchRun(runId, rec))) await saveRun(rec);
  } catch (e) {
    // The Android checkpoint remains authoritative and openable even when
    // IndexedDB is full, unavailable, or belongs to another WebView profile.
    console.warn(`[headless] IndexedDB mirror failed for ${runId}`, e);
  }
  return rec;
}

let androidRunMirror = null;
async function mirrorAndroidRunDirs() {
  if (androidRunMirror) return androidRunMirror;
  const b = headlessBridge();
  if (!b || typeof b.listRecoverableRuns !== 'function') return 0;
  androidRunMirror = (async () => {
    let ids = [];
    try { ids = JSON.parse(b.listRecoverableRuns()); } catch { return 0; }
    if (!Array.isArray(ids) || !ids.length) return 0;
    const { listRuns } = await import('./store.js');
    const existing = new Set((await listRuns()).filter((r) => r && r.state && r.recon).map((r) => r.id));
    let restored = 0;
    for (const runId of ids) {
      if (!Headless.validRunId(runId) || existing.has(runId)) continue;
      try {
        let status = { status: 'interrupted', owner: 'activity', at: 0 };
        try {
          const response = await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')));
          if (response.ok) status = await response.json();
        } catch { /* a checkpoint is still a durable Home model without control metadata */ }
        await mirrorHeadlessRun(runId, status);
        restored++;
      } catch (e) { console.warn(`[headless] could not mirror ${runId}`, e); }
    }
    return restored;
  })();
  try { return await androidRunMirror; }
  finally { androidRunMirror = null; }
}

/** Boot hijack: a marker from an outlived run opens adopt / monitor /
 *  recover instead of home. True when it took over boot. */
async function headlessBootCheck() {
  if (!headlessBridge()) return false;
  let runId = null;
  try {
    const r = await fetch(Headless.activeUrl());
    if (r.ok) runId = (await r.json()).runId;
  } catch { /* an older broken handback may have removed only active.json */ }
  if (!Headless.validRunId(runId)) {
    const b = headlessBridge();
    try { runId = typeof b.recoverActiveRun === 'function' ? b.recoverActiveRun() : null; }
    catch { runId = null; }
  }
  if (!Headless.validRunId(runId)) return false;
  let status = null;
  try {
    status = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')))).json();
  } catch { status = null; }
  if (!status || typeof status.status !== 'string') { headlessDiscard(runId); return false; }
  // Populate Home before showing monitor/recovery. A failure here never blocks
  // the authoritative Android files from opening in the monitor.
  await mirrorHeadlessRun(runId, status).catch((e) => console.warn('[headless] Home mirror failed', e));
  if (status.status === 'finished') { await adoptHeadlessRun(runId, 'finished'); return true; }
  if (status.status === 'interrupted' && status.stopped) {
    showRecovery(runId, status);
    return true;
  }
  if (!Headless.isLiveStatus(status)) { showRecovery(runId, status); return true; }
  let service = null;
  try {
    service = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'service.json')))).json();
  } catch { service = null; }
  // owner activity + live status with no service heartbeat: the Activity died
  // before any takeover (a crash, or a swipe racing the service boot) —
  // recover from the files. A fresh service heartbeat means the takeover is
  // already under way: monitor it instead of fighting it.
  if (status.owner === 'service' || Headless.heartbeatFresh(service)) { enterMonitor(runId); return true; }
  showRecovery(runId, status);
  return true;
}
/** Notification tap into an existing Activity: reopen the file-backed
 *  monitor, but never adopt or stop the service trainer automatically. */
globalThis.__splatxrOpenMonitor = () => {
  if (S.state === 'monitor') return;
  headlessBootCheck().catch((e) => console.warn('[headless] monitor reopen failed', e));
};

/** The monitor: progress + snapshot viewer for a run owned by the service.
 *  No session, no GPU, no viewport — this process only watches files. */
function enterMonitor(runId) {
  // Cut away every surface from the Activity's former train/home state. The
  // monitor is a real screen, not a modal over a still-mounted chart or wall.
  closeOverlays();
  dock('');
  $('monitor').hidden = true;
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  window.scrollTo(0, 0);
  S.state = 'monitor';
  S._monRun = runId; S._monAt = 0; S._monShotBusy = false; S._monShotAt = 0;
  // off by default, like the in-app training visuals — snapshots render on demand
  try { S._monVisuals = localStorage.getItem('splatjs_monitor_visuals') === '1'; }
  catch { S._monVisuals = false; }
  $('start').hidden = true; $('detail').hidden = true; $('controls').hidden = true;
  if (vp && vp.ctx) vp.ctx.clearRect(0, 0, vp.w, vp.h);
  let mc = document.getElementById('cv-monitor');
  if (!mc) {
    mc = document.createElement('canvas');
    mc.id = 'cv-monitor';
    $('stage').insertBefore(mc, $('cv'));
  }
  mc.style.display = S._monVisuals ? '' : 'none';
  paintMonitorHead(runId);
  wireMonitorOnce();
  $('monitor').hidden = false;
  syncMonitorButtons();
  monitorPoll(0, true);
}
function wireMonitorOnce() {
  if (S._monWired) return;
  S._monWired = true;
  $('mon-visuals').addEventListener('click', () => {
    S._monVisuals = !S._monVisuals;
    try { localStorage.setItem('splatjs_monitor_visuals', S._monVisuals ? '1' : '0'); } catch { /* private mode */ }
    const mc = document.getElementById('cv-monitor');
    if (mc) mc.style.display = S._monVisuals ? '' : 'none';
    syncMonitorButtons();
    if (S._monVisuals) monitorPoll(0, true);   // visuals on: a snapshot now, not next poll
  });
  $('mon-pause').addEventListener('click', () => {
    const b = headlessBridge();
    if (!b || !S._monRun) return;
    try { b.headlessCommand(S._monPaused ? 'headless-resume' : 'headless-pause', S._monRun); } catch { /* best-effort */ }
    S._monPaused = !S._monPaused;
    syncMonitorButtons();
  });
  $('mon-stop').addEventListener('click', async () => {
    if (!S._monRun) return;
    $('mon-stop').disabled = true;
    flash('Securing the background checkpoint and transferring training here …', 60000);
    if (!(await returnHeadlessRun(S._monRun))) $('mon-stop').disabled = false;
  });
  $('mon-continue').addEventListener('click', async () => {
    if (!S._monRun) return;
    $('mon-continue').disabled = true;
    if (!(await returnHeadlessRun(S._monRun))) $('mon-continue').disabled = false;
  });
  $('mon-discard').addEventListener('click', () => {
    if (S._monRun) headlessDiscard(S._monRun);
    S._monRun = null;
    $('monitor').hidden = true;
    document.getElementById('cv-monitor')?.remove();
    S.state = 'ready';
    showHome();
  });
  $('mon-x').addEventListener('click', () => {
    // step aside to the wall (the run keeps training headless); Start stays
    // guarded by the live marker until adopted, recovered or discarded
    $('monitor').hidden = true;
    document.getElementById('cv-monitor')?.remove();
    S.state = 'ready';
    showHome();
  });
}
function syncMonitorButtons() {
  const v = $('mon-visuals'), p = $('mon-pause');
  if (v) v.textContent = S._monVisuals ? 'Hide visuals' : 'Show visuals';
  if (p) p.textContent = S._monPaused ? 'Resume' : 'Pause';
  // wireMonitorOnce persists across monitor visits. Reset the transfer buttons
  // from current state so a successful first handback cannot leave the second
  // visit permanently disabled.
  const transferring = !!S._handbackRun;
  if ($('mon-stop')) $('mon-stop').disabled = transferring;
  if ($('mon-continue')) $('mon-continue').disabled = transferring;
  for (const id of ['mon-live', 'mon-rec']) {
    const el = $(id);
    if (el) el.hidden = (id === 'mon-live') === !!S._monRecovery;
  }
}
function paintMonitorHead(runId) {
  S._monRecovery = false;
  $('mon-name').textContent = 'Background training';
  $('mon-sub').textContent = `run ${runId}`;
  syncMonitorButtons();
}
/** The monitor tick (5 s): status/progress paint, snapshot when visuals are
 *  on, and the transitions — finished adopts, dead/interrupted recovers. */
async function monitorPoll(now, force = false) {
  if (S.state !== 'monitor' || !S._monRun || S._monRecovery) return;
  // Stop & continue intentionally makes the service publish a final
  // interrupted/stopped checkpoint before releasing its GPU. That is a
  // transfer protocol state, not a failure modal.
  if (S._handbackRun === S._monRun) return;
  if (!force && now - S._monAt < 5000) return;
  S._monAt = now;
  const runId = S._monRun;
  try {
    const status = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')))).json().catch(() => null);
    if (!status) return;   // transient read failure — retry next poll
    // A poll can have started just before the button set _handbackRun. Do not
    // reinterpret the service's orderly stop acknowledgment as a crash.
    if (S._handbackRun === runId) return;
    if (status.status === 'finished') { adoptHeadlessRun(runId, 'finished'); return; }
    if (status.status === 'interrupted' && status.stopped) {
      showRecovery(runId, status);
      return;
    }
    if (!Headless.isLiveStatus(status)) { showRecovery(runId, status); return; }
    let service = null, progress = null;
    try { service = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'service.json')))).json(); } catch { /* quiet */ }
    try { progress = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'progress.json')))).json(); } catch { /* quiet */ }
    if (S._handbackRun === runId) return;
    if (status.owner === 'service' && !Headless.heartbeatFresh(service)) { showRecovery(runId, status); return; }
    S._monPaused = status.status === 'paused';
    paintMonitorProgress(status, progress);
    syncMonitorButtons();
    if (S._monVisuals) monitorSnapshot(runId);
  } catch (e) { console.warn('[headless] monitor poll failed', e); }
}
function paintMonitorProgress(status, progress) {
  const p = progress || {};
  const iter = p.iter ?? status.iter ?? 0, max = p.maxIters || 0;
  $('mon-state').textContent = status.status === 'paused' ? 'Paused' :
    status.status === 'staging' ? 'Preparing …' : status.status === 'booting' ? 'Taking over …' : 'Training';
  $('mon-iter').textContent = max > 0
    ? `${fmt(iter)} / ${fmt(max)} cycles (${Math.min(99, Math.round(100 * iter / max))}%)`
    : `${fmt(iter)} cycles`;
  $('mon-score').textContent = p.psnr != null ? `${(+p.psnr).toFixed(1)} dB` : '—';
  $('mon-splats').textContent = p.splats != null ? fmt(p.splats) : '—';
  const age = p.at ? Math.max(0, Math.round((Date.now() - p.at) / 1000)) : null;
  $('mon-updated').textContent = age == null ? '' : age < 8 ? 'live' : `updated ${age}s ago`;
}
/** Demand-driven snapshot: ask the headless twin for a fresh render, show the
 *  newest PNG on disk (it lands a poll or two after the request). */
async function monitorSnapshot(runId) {
  if (S._monShotBusy) return;
  S._monShotBusy = true;
  try {
    const b = headlessBridge();
    try { b && b.headlessCommand('headless-snapshot', runId); } catch { /* best-effort */ }
    const meta = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'snapshot.json')))).json().catch(() => null);
    if (!meta || !(meta.at > S._monShotAt)) return;
    const r = await fetch(Headless.fresh(Headless.runUrl(runId, 'snapshot.png')));
    if (!r.ok) return;
    const bmp = await createImageBitmap(await r.blob());
    if (S._monRun !== runId || !S._monVisuals) { bmp.close(); return; }
    const mc = document.getElementById('cv-monitor');
    if (!mc) { bmp.close(); return; }
    if (mc.width !== bmp.width || mc.height !== bmp.height) { mc.width = bmp.width; mc.height = bmp.height; }
    mc.getContext('2d').drawImage(bmp, 0, 0);
    bmp.close();
    S._monShotAt = meta.at;
  } catch (e) { console.warn('[headless] snapshot poll failed', e); }
  finally { S._monShotBusy = false; }
}
/** Take the run back from the service: stop any live twin, then verify it
 *  stood down (or was already dead) before local training touches the GPU.
 *  False = a takeover is still in flight, retry shortly. */
async function headlessTakeover(runId) {
  const b = headlessBridge();
  // Do not start a dead service merely to tell it to stop: that path rewrites
  // status.json and used to erase the renderer-restart transaction metadata.
  try {
    const [status, service] = await Promise.all([
      fetch(Headless.fresh(Headless.runUrl(runId, 'status.json'))).then((r) => r.ok ? r.json() : null),
      fetch(Headless.fresh(Headless.runUrl(runId, 'service.json'))).then((r) => r.ok ? r.json() : null),
    ]);
    if (status && !Headless.isLiveStatus(status) && service &&
        (service.alive === false || !Headless.heartbeatFresh(service))) return true;
  } catch { /* command + acknowledgment loop below remains authoritative */ }
  try { b && b.headlessCommand('headless-stop', runId); } catch { /* best-effort */ }
  // A non-live status is written before Java destroys the service WebView.
  // Wait for service.json alive:false (written AFTER teardown), or a stale
  // heartbeat, so the Activity never creates a second WebGPU owner.
  for (let i = 0; i < 90; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    let status = null, service = null;
    try { status = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')))).json(); }
    catch { /* a torn/transition read is not proof the GPU is gone */ }
    try { service = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'service.json')))).json(); }
    catch { service = null; }
    const stateReleased = !status || !Headless.isLiveStatus(status) || status.owner === 'activity';
    const serviceReleased = !!service &&
      (service.alive === false || !Headless.heartbeatFresh(service));
    if (stateReleased && serviceReleased) return true;
  }
  return false;
}

/** The one automatic service→Activity handback used by notification return,
 *  normal foregrounding, and the monitor button. */
async function returnHeadlessRun(runId) {
  if (!runId) return false;
  if (S._handbackRun === runId) return false;
  S._handbackRun = runId;
  flash('Returning training to the app …', 90000);
  try {
    if (!(await headlessTakeover(runId))) {
      flash('The background trainer has not released the GPU yet. It is still safe in the service; try again shortly.', 9000);
      return false;
    }
    return await adoptHeadlessRun(runId, 'continue');
  } finally {
    if (S._handbackRun === runId) S._handbackRun = null;
    if (S.state === 'monitor') syncMonitorButtons();
  }
}
/** Adopt a background run into this Activity: finished opens view-only in the
 *  done viewer, continue resumes training locally under the same run id (the
 *  photos are already staged, so the handoff re-arms without re-uploading). */
async function adoptHeadlessRun(runId, mode) {
  // Continue is the sole transition out of the monitor. Remove its canvas and
  // any stale wall/train dock before continuation creates a new chart/canvas.
  closeOverlays();
  dock('');
  $('start').hidden = true; $('detail').hidden = true; $('controls').hidden = true;
  $('monitor').hidden = true;
  document.getElementById('cv-monitor')?.remove();
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  window.scrollTo(0, 0);
  flash(mode === 'finished' ? 'Opening the finished model …' : 'Resuming training here …', 60000);
  let handbackStatus = {};
  try {
    const manifest = Headless.validateManifest(
      await (await fetch(Headless.fresh(Headless.runUrl(runId, 'manifest.json')))).json());
    S._hlManifest = manifest;
    const recon = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'recon.json')))).json();
    const stateBytes = new Uint8Array(
      await (await fetch(Headless.fresh(Headless.runUrl(runId, 'state.bin')))).arrayBuffer());
    const status = await (await fetch(Headless.fresh(Headless.runUrl(runId, 'status.json')))).json().catch(() => ({}));
    handbackStatus = status;
    S._hlGpuRestarts = status.gpuRestarts | 0;
    const { parseState } = await import('./session_io.js');
    parseState(stateBytes);   // torn writes refuse HERE, not mid-rebuild
    let thumb = null;
    try {
      const r = await fetch(Headless.fresh(Headless.runUrl(runId, 'snapshot.png')));
      if (r.ok) thumb = await r.blob();
    } catch { /* the tile just goes textless */ }
    // photos resolve as URL-backed: the runs handler serves the staged bytes
    recon.source = {
      preset: null,
      names: [...manifest.photos],
      urls: manifest.photoSource === 'urls'
      ? [...manifest.photoUrls]
      : manifest.photos.map((n) => Headless.photoUrl(runId, n)),
      ...(manifest.srcKey ? { captureId: manifest.srcKey } : {}),
    };
    const rec = {
      id: runId, name: manifest.name, status: 'finished',
      createdAt: manifest.stagedAt || Date.now(), updatedAt: Date.now(),
      iter: status.iter ?? recon.iter ?? 0, maxIters: manifest.maxIters,
      splats: status.splats ?? recon.splats ?? 0,
      psnr: status.psnr ?? (recon.stats && recon.stats.psnrTrain) ?? null,
      frames: manifest.photos.length, ownSrc: !!manifest.srcKey,
      ...(manifest.srcKey ? { srcKey: `native:${manifest.srcKey}` } : {}),
      state: new Blob([stateBytes]), recon,
      detailProgress: headlessProgressDetails(recon, manifest, status),
      ...(thumb ? { thumb } : {}),
    };
    const { saveRun, patchRun } = await import('./store.js');
    // same id as the run's own record: adopting patches it, never duplicates it
    if (!(await patchRun(runId, rec))) await saveRun(rec);
    S._localRun = { ...rec };
    $('monitor').hidden = true;
    document.getElementById('cv-monitor')?.remove();
    if (mode === 'finished') {
      await continueLocalRun(true);
      if (S.state !== 'done' || !S.session || !S.session.trainer) {
        throw new Error('the finished model did not rebuild');
      }
      // Finished/view-only has no live owner. Clear only AFTER its viewer is
      // proven live; until then active.json is the recovery transaction log.
      headlessClear(runId);
    } else {
      // The staged AR photos + active marker remain authoritative throughout
      // rebuild. Never publish owner=activity before that Activity has a live
      // trainer, and never clear the marker for a continuing run.
      S._hlStaged = runId; S._hlPhotos = true; S._hlCkptBusy = false;
      // WebView.destroy() ran before alive:false, but give the GPU process one
      // turn to retire its device before allocating the Activity's replacement.
      await new Promise((resolve) => setTimeout(resolve, 300));
      await continueLocalRun(false);
      // Worker setup can resolve just before device.lost is delivered. Give
      // that event a turn and reject a trainer whose GPU is already dead.
      await new Promise((resolve) => setTimeout(resolve, 750));
      if (S.state !== 'train' || !S.session || !S.session.trainer || S.session.deviceLost) {
        throw new Error('the local trainer did not rebuild on a live graphics device');
      }
      const b = headlessBridge();
      if (b) {
        try {
          b.setActiveRun(runId);
          b.writeRunJson(runId, 'status.json', JSON.stringify(
            { status: 'active', owner: 'activity', iter: S.session.trainer.iter | 0,
              at: Date.now(), gpuRestarts: S._hlGpuRestarts | 0 }));
          b.writeRunJson(runId, 'activity.json', JSON.stringify({ alive: true, at: Date.now() }));
        } catch { /* the next activity beat retries ownership evidence */ }
      }
    }
    return true;
  } catch (e) {
    console.error(e);
    // Transaction rollback: files, state and active.json were never cleared.
    // Put recovery in front of the user instead of trainDeadEnd's Home screen.
    S._hlStaged = runId; S._hlPhotos = true; S._hlCkptBusy = false;
    const reason = String((e && e.message) || e);
    const restarts = handbackStatus.gpuRestarts | 0;
    const b = headlessBridge();
    if (mode !== 'finished' && restarts < 1 &&
        /device|GPU|graphics|WebGPU|worker/i.test(reason) && b &&
        typeof b.restartForGpuRecovery === 'function') {
      try {
        b.writeRunJson(runId, 'status.json', JSON.stringify({
          ...handbackStatus, status: 'interrupted', stopped: true, owner: 'activity',
          at: Date.now(), gpuRestarts: restarts + 1,
          error: `reopening the Activity graphics process after handback failure: ${reason}`,
        }));
        S._hlGpuRestarts = restarts + 1;
        b.restartForGpuRecovery(runId);
        return false;
      } catch (restartErr) { console.error(restartErr); }
    }
    showRecovery(runId, { ...handbackStatus, status: 'interrupted', stopped: true,
      error: `return to app failed: ${reason}` });
    flash(`Could not resume in the app (${reason}) — the checkpoint and AR frames are still available here.`, 12000);
    return false;
  }
}
/** The run died out of reach (pressure kill, crash, pre-handoff swipe): offer
 *  Continue-here from the files, or Discard. */
function showRecovery(runId, status) {
  closeOverlays();
  dock('');
  $('monitor').hidden = true;
  document.getElementById('cv-model')?.remove();
  gpuCanvas = null;
  window.scrollTo(0, 0);
  S.state = 'monitor';
  S._monRun = runId; S._monRecovery = true;
  $('start').hidden = true; $('detail').hidden = true; $('controls').hidden = true;
  document.getElementById('cv-monitor')?.remove();
  $('mon-name').textContent = 'Background training interrupted';
  const iter = status && status.iter ? ` at cycle ${fmt(status.iter)}` : '';
  const why = status && status.error ? ` — ${status.error}` :
    ' — the system reclaimed it while the app was away.';
  $('mon-sub').textContent = `The run's last checkpoint${iter} is stored on this device${why}`;
  wireMonitorOnce();
  $('monitor').hidden = false;
  syncMonitorButtons();
}
/** Drop the marker (unreadable status, user discard). The files stay until the
 *  next staged run cleans them — a discard must never strand a booting twin
 *  by deleting the bytes under it. */
function headlessDiscard(runId) {
  const b = headlessBridge();
  try {
    b && b.writeRunJson(runId, 'status.json', JSON.stringify(
      { status: 'discarded', owner: 'activity', iter: 0, at: Date.now() }));
  } catch { /* clearing the marker still hides it for this boot */ }
  headlessClear(runId);
}
/** Foreground-service handoff: secure one exact checkpoint, then relinquish
 *  the Activity's GPU before the service creates its WebView. A failed save
 *  leaves the local trainer untouched; Java retries while the Activity stays
 *  backgrounded. Once successful this page becomes the file-backed monitor. */
globalThis.__splatxrHandoff = async () => {
  const b = headlessBridge();
  const runId = S.runId;
  let ok = false;
  try { ok = await secureFileCheckpoint(); } catch { ok = false; }
  if (ok && runId && S.runId === runId && S.state === 'train' && S.session) {
    try { S.session.pause(); S.session.dispose(); } catch { /* checkpoint is already safe */ }
    S.session = null;
    S.scene = null;
    document.getElementById('cv-model')?.remove();
    gpuCanvas = null;
    // bootHeadlessNow refuses a fresh Activity heartbeat. Retire it only after
    // the checkpoint and GPU disposal have both succeeded.
    try { b && b.writeRunJson(runId, 'activity.json', JSON.stringify({ alive: false, at: Date.now() })); }
    catch { /* the service also waits out stale beats */ }
    enterMonitor(runId);
  }
  try { b && b.handoffDone(runId, ok); } catch { /* the service's absence timer settles from the saved files */ }
};

// ── training notification (Android) ─────────────────────────────────────────
// A foreground-service notification carries the live score + splat count and
// keeps the process (and its GPU work) alive while the app runs in the
// background. Desktop browsers have no bridge — every call is feature-detected.
const trainBridge = () => {
  const b = globalThis.AndroidARBridge;
  return b && typeof b.updateTrainingNotification === 'function' ? b : null;
};
function androidNotifStart() {
  const b = trainBridge();
  if (!b) return;
  S._notifName = (S._localRun && S._localRun.name) || (S.preset && S.preset.name) || 'model';
  S._notifT = 0;
  try { b.startTrainingNotification(`Training ${S._notifName}`, 'Starting…'); } catch { /* best-effort */ }
}
/** The live score for the notification shade, in the user's chosen unit. */
function notifScore() {
  const psnr = S.psnrHold ?? S.psnrTrain;
  if (psnr == null) return '';
  return S.scoreScale === 'similarity'
    ? `${psnrToApproxSsimPercent(psnr).toFixed(1)} % · `
    : `${psnr.toFixed(1)} dB · `;
}
function androidNotifTick() {
  const b = trainBridge();
  if (!b || S.state !== 'train' || !S.session || !S.session.training) return;
  const now = performance.now();
  if (now - (S._notifT || 0) < 2500) return;   // the shade is not a 60 fps surface
  S._notifT = now;
  const text = `${notifScore()}${fmt(S.splats)} splats · ${fmt(S.iter)} / ${fmt(S.maxIters)}`;
  try { b.updateTrainingNotification(`Training ${S._notifName || 'model'}`, text, S.iter | 0, S.maxIters | 0); }
  catch { /* best-effort */ }
}
function androidNotifStop(finished) {
  const b = trainBridge();
  if (!b) return;
  try {
    if (finished) {
      b.finishTrainingNotification('Training complete',
        `${notifScore()}${fmt(S.splats)} splats · ${fmt(S.iter)} cycles`);
    } else {
      b.cancelTrainingNotification();
    }
  } catch { /* best-effort */ }
  S._notifT = 0;
}

/** Flip the notification's Pause/Resume button — false when the bridge is
 *  too old, so the caller falls back to shedding/restarting the service. */
function notifSetPaused(paused) {
  const b = trainBridge();
  if (!b || typeof b.setTrainingPaused !== 'function') return false;
  try { b.setTrainingPaused(paused); } catch { return false; }
  return true;
}
// The shade's Pause/Resume button calls in here (MainActivity turns the
// broadcast into this toggle) — the same path as the in-app button.
globalThis.__splatxrNotifToggle = () => { if (S.state === 'train') toggleTrain(); };

// ── screen wake lock ────────────────────────────────────────────────────────
// a long run gets no touches, and phones dim and lock the screen — hold a
// wake lock while the pipeline works (and while the done-tour is playing)
let wakeLock = null;
async function updateWakeLock() {
  const want = document.visibilityState === 'visible' &&
    (S.state === 'prep' || S.state === 'train' || (S.state === 'done' && !!S.tour));
  if (want && !wakeLock && navigator.wakeLock) {
    try {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    } catch { /* denied (battery saver etc.) — nothing to do */ }
  } else if (!want && wakeLock) {
    try { wakeLock.release(); } catch { /* already gone */ }
    wakeLock = null;
  }
}
document.addEventListener('visibilitychange', updateWakeLock);

// ── main loop ───────────────────────────────────────────────────────────────
let lastPulse = 0;
let lastLoopT = performance.now();

// ── done-state intro: glide along the capture path until the user acts ──────
// rotation interpolation: quaternions of the SOLVED camera matrices, so the
// replay carries the photographer's true roll (an orbit camera cannot)
function startTour(fromNearest = false) {
  let cams = S.scene ? S.scene.cams.filter((c) => c.R) : [];
  if (cams.length < 2) return;

  // ---- a stable path needs stable NODES first ----
  // 1) collapse co-located cameras (a pano rig contributes six orientations
  //    at one position — flying "through" them whips the view around)
  const centres = cams.map(camCentre);
  const gaps = [];
  for (let i = 1; i < centres.length; i++) {
    const d = Math.hypot(...[0, 1, 2].map((c) => centres[i][c] - centres[i - 1][c]));
    if (d > 1e-9) gaps.push(d);
  }
  gaps.sort((a, b) => a - b);
  const medGap = gaps[gaps.length >> 1] || 1e-3;
  const dedup = [];
  for (let i = 0; i < cams.length; i++) {
    const last = dedup[dedup.length - 1];
    if (!last || Math.hypot(...[0, 1, 2].map((c) => centres[i][c] - camCentre(last)[c])) > 0.15 * medGap) {
      dedup.push(cams[i]);
    }
  }
  // 2) never spline across a real break in the capture (a second walk, a
  //    registration gap): split into segments, fly the longest one
  const segs = [[dedup[0]]];
  for (let i = 1; i < dedup.length; i++) {
    const a = camCentre(dedup[i - 1]), b = camCentre(dedup[i]);
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (d > 4 * medGap) segs.push([]);
    segs[segs.length - 1].push(dedup[i]);
  }
  cams = segs.reduce((best, s) => (s.length > best.length ? s : best), []);
  const n = cams.length;
  if (n < 2) return;

  // pre-smooth positions over ±3 neighbours — a handheld capture path is
  // jittery, and a spline through jitter is jittery too
  const raw = cams.map(camCentre);
  const smooth3 = (arr) => arr.map((_, i) => {
    const acc = [0, 0, 0];
    let w = 0;
    for (let k = -3; k <= 3; k++) {
      const j = clamp(i + k, 0, arr.length - 1);
      const wt = 4 - Math.abs(k);
      for (let c = 0; c < 3; c++) acc[c] += arr[j][c] * wt;
      w += wt;
    }
    return [acc[0] / w, acc[1] / w, acc[2] / w];
  });
  const pts = smooth3(raw);

  // true rotations, sign-aligned then lightly smoothed towards the midpoint
  // of the neighbours — handheld roll jitter, not the roll itself, goes away
  const qs = cams.map((c) => quatFromR(c.R));
  for (let i = 1; i < qs.length; i++) {
    if (qs[i - 1][0] * qs[i][0] + qs[i - 1][1] * qs[i][1] + qs[i - 1][2] * qs[i][2] + qs[i - 1][3] * qs[i][3] < 0) {
      qs[i] = qs[i].map((v) => -v);
    }
  }
  // two smoothing passes: handheld roll/aim jitter goes, deliberate turns stay
  let sq = qs;
  for (let pass = 0; pass < 2; pass++) {
    sq = sq.map((q, i) => {
      if (i === 0 || i === sq.length - 1) return q;
      return qslerp(q, qslerp(sq[i - 1], sq[i + 1], 0.5), 0.5);
    });
  }

  // Catmull-Rom, densely resampled into an arc-length table: playback walks
  // the table at EXACTLY constant velocity, whatever the gap sizes
  const P = (k) => pts[clamp(k, 0, n - 1)];
  const cr = (i, f) => {
    const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
    const t2 = f * f, t3 = t2 * f;
    return [0, 1, 2].map((k) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * f +
      (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
      (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3));
  };
  const samples = [], us = [], cum = [0];
  const SUB = 8;
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < SUB; k++) { samples.push(cr(i, k / SUB)); us.push(i + k / SUB); }
  }
  samples.push(cr(n - 2, 1)); us.push(n - 1);
  for (let k = 1; k < samples.length; k++) {
    const a = samples[k - 1], b = samples[k];
    cum.push(cum[k - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  const total = cum[cum.length - 1] || 1e-6;
  const duration = clamp(1.4 * n, 10, 30);   // one full pass, 30 s at most

  S.tour = { cams, sq, samples, us, cum, total, speed: total / duration, s: 0, k: 0, dir: 1, pd: [] };

  // ALWAYS pick up from wherever the view is — the intro starts at the
  // landing pose (the first camera), replay from wherever the user flew to;
  // either way, no jump-cut into the path
  {
    const { fwd } = vp._basis();
    const eye = [vp.target[0] - fwd[0] * vp.dist, vp.target[1] - fwd[1] * vp.dist, vp.target[2] - fwd[2] * vp.dist];
    let best = 0, bd = Infinity;
    samples.forEach((p, k) => {
      const dd = (p[0] - eye[0]) ** 2 + (p[1] - eye[1]) ** 2 + (p[2] - eye[2]) ** 2;
      if (dd < bd) { bd = dd; best = k; }
    });
    S.tour.s = cum[best];
    S.tour.k = Math.min(best, cum.length - 2);
    // heading outward from a path end plays the return leg first
    if (!fromNearest && S.tour.s > total * 0.5) S.tour.dir = -1;
  }
}

function stopTour() {
  if (!S.tour) return;
  S.tour = null;
  vp.pose = null;   // hand the view back to the orbit (which has no roll)
  vp.dirty = true;
}

function tourStep(dt) {
  const T = S.tour;
  if (!T) return;
  if (S.state !== 'done' || S.atFrame >= 0 || S.picking || !$('details').hidden) return;
  if (S.keys.size) { stopTour(); return; }   // flying takes over

  T.s += dt * T.speed * T.dir;               // constant velocity, ping-pong
  if (T.s >= T.total) { T.s = T.total; T.dir = -1; }
  if (T.s <= 0) { T.s = 0; T.dir = 1; }
  while (T.k < T.cum.length - 2 && T.cum[T.k + 1] < T.s) T.k++;
  while (T.k > 0 && T.cum[T.k] > T.s) T.k--;

  const span = Math.max(1e-9, T.cum[T.k + 1] - T.cum[T.k]);
  const a = (T.s - T.cum[T.k]) / span;
  const A = T.samples[T.k], B = T.samples[T.k + 1];
  const pos = [A[0] + (B[0] - A[0]) * a, A[1] + (B[1] - A[1]) * a, A[2] + (B[2] - A[2]) * a];
  const u = T.us[T.k] + (T.us[T.k + 1] - T.us[T.k]) * a;
  const i = clamp(Math.floor(u), 0, T.cams.length - 2);
  const f = u - i;

  // gravity-leveled flight: position and aim follow the capture path, but
  // roll levels to the orbit's up axis through the pose override (the raw
  // sensor roll would swing the world sideways on portrait-held captures —
  // and the orbit view the tour starts from is leveled, so the tour must be
  // too). A straight up/down stare keeps the raw pose: no level roll exists.
  const Rq0 = quatToR(qslerp(T.sq[i], T.sq[i + 1], f));
  const level = rollAngleToUp(Rq0, vp.up);
  const Rq = level == null ? Rq0 : rerollPose(Rq0, pos, level).R;
  vp.pose = {
    R: Rq,
    t: [
      -(Rq[0] * pos[0] + Rq[1] * pos[1] + Rq[2] * pos[2]),
      -(Rq[3] * pos[0] + Rq[4] * pos[1] + Rq[5] * pos[2]),
      -(Rq[6] * pos[0] + Rq[7] * pos[1] + Rq[8] * pos[2]),
    ],
  };

  // keep the orbit tracking underneath (minus roll) so any user takeover
  // continues seamlessly from here
  const fwd = [Rq[6], Rq[7], Rq[8]];
  const da = (T.pd[i] ??= vp._pivotDist(T.cams[i]));
  const db = (T.pd[i + 1] ??= vp._pivotDist(T.cams[i + 1]));
  const d = da + (db - da) * f;
  vp.target = [pos[0] + fwd[0] * d, pos[1] + fwd[1] * d, pos[2] + fwd[2] * d];
  vp.dist = d;
  const ang = vp.anglesOf(fwd);
  vp.yaw = ang.yaw;
  vp.pitch = clamp(ang.pitch, -1.45, 1.45);
  vp.dirty = true;
}

/** WASD fly: move the orbit target along the camera's own axes */
function flyStep(dt) {
  if (!S.keys.size || !S.scene) return;
  if (S.state !== 'train' && S.state !== 'done') return;
  if (S.picking || !$('details').hidden) return;
  if (S.atFrame >= 0) leaveFrame();   // like a drag, movement leaves the photo
  const { fwd, right, down } = vp._basis();
  const boost = (S.keys.has('ShiftLeft') || S.keys.has('ShiftRight')) ? 3 : 1;
  // speed follows the pivot distance (zoomed in = fine movement, zoomed out
  // = covering ground), floored so a extreme close-up can still move
  const sp = Math.max(vp.dist, S.scene.radius * 0.02) * 1.2 * dt * boost;
  let any = false;
  const move = (v, f) => { for (let i = 0; i < 3; i++) vp.target[i] += v[i] * f; any = true; };
  if (S.keys.has('KeyW')) move(fwd, sp);
  if (S.keys.has('KeyS')) move(fwd, -sp);
  if (S.keys.has('KeyA')) move(right, -sp);
  if (S.keys.has('KeyD')) move(right, sp);
  if (S.keys.has('KeyE')) move(down, -sp);   // up
  if (S.keys.has('KeyQ')) move(down, sp);    // down
  if (any) vp.dirty = true;
}

function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastLoopT) / 1000);
  lastLoopT = now;
  if (S.flash && now > S.flash.until) S.flash = null;
  const wlWant = `${S.state}:${!!S.tour}`;
  if (wlWant !== S._wlKey) { S._wlKey = wlWant; updateWakeLock(); }
  // the tour can stop from any interaction — keep the play button honest
  const tourBtn = $('c-play');
  if (tourBtn && tourBtn.dataset.on !== String(!!S.tour)) {
    tourBtn.dataset.on = String(!!S.tour);
    tourBtn.title = S.tour ? 'Stop the flight' : 'Fly the capture path';
    tourBtn.innerHTML = S.tour
      ? '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><rect x="7.5" y="7.5" width="9" height="9" rx="1.5"/></svg>'
      : '<svg viewBox="0 0 24 24" class="pl" aria-hidden="true"><path d="M8.5 5.5v13l10-6.5z"/></svg>';
  }
  const grow = $('t-grow');
  if (grow) {
    // nbsp when quiet: the slot keeps its baseline, so nothing below shifts
    const txt = (S.growNote && now < S.growNote.until) ? S.growNote.text : ' ';
    if (grow.textContent !== txt) grow.textContent = txt;
  }
  // the countdown ticks between metrics events; paused = frozen, honest
  if (S.state === 'train' && S.session && S.session.training && now - (S._etaTick || 0) > 250) {
    S._etaTick = now;
    const eta = $('t-eta');
    if (eta) {
      const txt = etaText();
      if (eta.textContent !== txt) eta.textContent = txt;
    }
  }
  // no fade on the photo overlay — it reads as lag on slow devices
  S.fade = S.fadeTo;
  flyStep(dt);
  tourStep(dt);

  if (S.walking && S.walker) {
    try { S.walker.step(performance.now() / 1000); }
    catch (e) { console.warn('[avatar] walk stopped: ' + (e.message || e)); S.walking = false; }
  }

  if (S.state === 'prep') paintPrepDock();

  if (S.state === 'train' && now - lastPulse > 300) {
    lastPulse = now;
    paintStrip();          // the pulse on the frame being trained on
  }
  if (S.state === 'train') headlessActivityBeat(now);
  if (S.state === 'monitor') monitorPoll(now);

  renderHud();
  draw();
  if (!$('details').hidden) drawDetail();
}

function paintPrepDock() {
  const bar = $('p-bar');
  if (!bar || !S.prep) return;
  const bi = beatIndex(S.prep.stage);
  const frac = S.prep.total ? S.prep.done / S.prep.total : 0;
  bar.style.width = `${((bi + Math.min(1, frac)) / BEATS.length) * 100}%`;
  $('p-sub').textContent = prepSub();
  [...$('p-steps').children].forEach((el, k) =>
    el.dataset.on = k < bi ? 'done' : k === bi ? '1' : '0');
}

function prepSub() {
  const e = S.prep;
  if (!e) return '—';
  const n = S.photos.length;
  if (e.stage === 'known') return 'using pose-synchronized cameras and per-frame intrinsics — no structure-from-motion';
  if (e.stage === 'matte') return `frame ${e.done} of ${e.total} · the room removed, the person kept`;
  if (e.stage === 'decode') return `photo ${e.done} of ${e.total}`;
  if (e.stage === 'features') {
    const total = [...S.feats.values()].reduce((a, f) => a + f.n, 0);
    return `${fmt(total)} spots · frame ${e.done} of ${e.total}`;
  }
  if (e.stage === 'matching') {
    return `${fmt(e.done)} of ${fmt(e.total)} pairs · ${fmt(e.detail?.usable ?? 0)} survived the geometry test`;
  }
  if (e.stage === 'focal') return `no lens data in the files — measuring the lens from the geometry · ${e.done + 1} of ${e.total}`;
  if (e.stage === 'register') {
    return `${e.done} of ${e.total} photos placed` +
      (S.regPtsCount ? ` · ${fmt(S.regPtsCount)} points triangulated` : '');
  }
  if (e.stage === 'ba') return 'polishing the geometry (bundle adjustment)';
  if (e.stage === 'solved' && e.detail.known) return `${e.detail.cams} known cameras · ${fmt(e.detail.points)} seed points · camera solve skipped`;
  if (e.stage === 'solved') return `${e.detail.cams} cameras · ${fmt(e.detail.points)} points · ${e.detail.rms ? e.detail.rms.toFixed(2) + 'px' : ''}`;
  if (e.stage === 'seed' && S.preset?.capture?.transforms) return 'seeding Gaussians with known poses; cameras and intrinsics remain fixed';
  if (e.stage === 'seed') return 'one splat per landmark, plus jittered copies';
  return '—';
}

// ── drawing ─────────────────────────────────────────────────────────────────
function draw() {
  // migrate-on-swipe monitor: the snapshot canvas paints itself on poll —
  // no viewport overlays over another process's training
  if (S.state === 'monitor') return;
  const cv = $('cv');
  const r = cv.getBoundingClientRect();
  if (Math.abs(r.width * (vp.dpr || 1) - cv.width) > 2 || Math.abs(r.height * (vp.dpr || 1) - cv.height) > 2) vp.resize();
  const ctx = vp.ctx, w = vp.w, h = vp.h, dpr = vp.dpr || 1;

  if (S.state === 'ready') { photoStage(ctx, w, h, dpr); return; }

  if (S.state === 'prep') {
    const st = S.prep && S.prep.stage;
    if (st === 'decode' || st === 'features') return photoStage(ctx, w, h, dpr, st === 'features');
    // the focal search draws nothing of its own — keep the last matching
    // frame on stage instead of cutting to black
    if (st === 'matching' || st === 'focal') return pairStage(ctx, w, h, dpr);
    // register / ba / solved / seed: the camera solve, live
    if (S.scene) {
      vp.draw({ points: true, cams: S.scene.cams, showCams: true, showPath: true, sel: S.sel, active: -1 });
    } else if (S.regCams.length) {
      vp.draw({ cams: S.regCams, showCams: true, bright: true, reveal: S.regCams.length, active: -1, sel: -1, cloud: S.regPts, cloudRgb: S.regRgb });
    } else {
      pairStage(ctx, w, h, dpr);   // nothing registered yet — hold the photos
    }
    return;
  }

  // During training the optimizer keeps its own compute/render path. These
  // app-level presentation passes (model viewport, camera overlays and image
  // comparison) are optional and sleep by default; the score chart is separate.
  const capView = S._capView && S._capView.active && !S._capView.dead ? S._capView : null;
  if (S.state === 'train' && (!S.trainVisuals || !$('details').hidden)) {
    if (gpuCanvas) gpuCanvas.style.display = 'none';
    if (capView && capView.canvas) capView.canvas.style.display = 'none';
    ctx.clearRect(0, 0, cv.width, cv.height);
    return;
  }
  if (capView && capView.canvas) {
    if (capView.canvas.style.display === 'none') capView.canvas.style.display = '';
    if (gpuCanvas) gpuCanvas.style.display = 'none';
  } else if (gpuCanvas && gpuCanvas.style.display === 'none') gpuCanvas.style.display = '';

  // train/done: the model, rendered by the trainer at this exact pose
  // (render capped at the session's view-buffer size, blit scales up)
  const onFrame = S.atFrame >= 0;
  const pose = vp.viewPose();
  if (capView && S.session && S.session.trainer) {
    drawCaptureView(capView, vp, w, h, pose);
  } else if (gpuCanvas && S.session && S.session.trainer) {
    const now = performance.now();
    if (vp.dirty) S._camMovedAt = now;
    const training = S.state === 'train' && S.session.training;
    const moving = vp.dirty || now - (S._camMovedAt || 0) < 250;
    // progressive resolution: reduced while the camera moves or training
    // runs (fluid), the FULL device-pixel canvas once it settles (true
    // retina) — always inside the allocated view buffers / tile-grid budget.
    // The SOG viewer renders far cheaper than the compute rasterizer, so it
    // affords a much higher moving budget (less visible softness while
    // rotating), highest where a mouse implies desktop-class GPU.
    // Remote training rasterizes on the worker with the SAME compute kernel
    // as local (RemoteTrainer just has no .device to read) — only the
    // SOG-lite stand-in earns the cheap-viewer budget (2026-11: the worker
    // view rendered up to 2x the pixels of main's on the same phone).
    const sogView = !(S.session.trainer && (S.session.trainer.device || S.session._remote));
    const moveBudget = sogView
      ? (matchMedia('(pointer: fine)').matches ? 2.8e6 : 1.8e6)
      : 1.3e6;
    const budgetPx = Math.min(S.viewPixBudget || 2560 * 1440,
      (training || moving) ? moveBudget : 1e9);
    const sc = Math.min(1, Math.sqrt(budgetPx / (w * h)));
    const gw = Math.max(2, Math.round(w * sc)), gh = Math.max(2, Math.round(h * sc));
    const key = `${gw}x${gh}|${Math.round(pose.f)}|` +
      pose.R.map((v) => Math.round(v * 8192)).join(',') + '|' +
      pose.t.map((v) => Math.round(v * 8192)).join(',');
    // re-render when the view actually changed; while training also refresh
    // the evolving model — 2/s at most, fewer on slow devices (~25
    // iterations' worth of time between refreshes). Each render here also
    // pushes back the session's own auto-refresh, so there is ONE timer.
    const refreshMs = Math.max(500, 25000 / Math.max(1, S.itersPerSec || 100));
    if (key !== S._viewKey || S.walking || (training && now - (S._lastViewAt || 0) > refreshMs)) {
      S._viewKey = key;
      S._lastViewAt = now;
      if (gpuCanvas.width !== gw || gpuCanvas.height !== gh) {
        gpuCanvas.width = gw; gpuCanvas.height = gh;
        quietView(S.session.view.attach(gpuCanvas));
      }
      quietView(S.session.view.setCamera({
        R: pose.R, t: pose.t,
        f: pose.f * sc, cx: pose.cx * sc, cy: pose.cy * sc, w: gw, h: gh,
      }));
      quietView(S.session.view.renderNow());
    }
  }

  vp.draw({
    model: !!gpuCanvas,
    cams: S.scene.cams,
    // on a photograph (compare modes) the overlays read as artefacts in the
    // image — frustums only while moving around freely, never during the
    // intro flight, and hidden entirely in the finished viewer (the drawing
    // path stays; flip this when frustums earn a place there again)
    showCams: S.state !== 'done' && !onFrame && !S.tour,
    showPath: S.state === 'train' && !onFrame,
    faint: S.state === 'done',
    skip: S.atFrame,
    active: S.state === 'train' && S.session.training
      ? (S.scene.cams.find((c) => c.ci === S.session.activeCam) || {}).i : -1,
    sel: S.sel,
    dimOthers: S.state === 'train' && S.session.training,
  });

  if (S.fade > .005 && dev.ready) {
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.globalAlpha = S.fade;
    const errKey = `${S.atFrame}:${S.iter}`;
    if (S.compare === 'error' && S.session && S.session.trainer) ensureErrRender(errKey);
    S.rect = dev.render(ctx, w / dpr, h / dpr, {
      mode: S.compare, loupe: S.loupe, swipe: S.swipe, dpr,
      model: (S._errRender && S._errRender.key === errKey) ? S._errRender.canvas : null,
      key: errKey,
    });
    ctx.restore();
  }
}

/** a photograph filling the stage (ready + the first prep beats) */
function photoStage(ctx, w, h, dpr, marks = false) {
  ctx.fillStyle = '#070909';
  ctx.fillRect(0, 0, w, h);
  if (!S.photos.length || !S.photos[S.sel] || !S.photos[S.sel].url) return; // intro: bare stage
  // fast machines outrun the decoder during the landmarks beat (the selected
  // frame changes every ~90ms, a full-res decode takes longer) — hold the
  // last DECODED photo instead of flashing black, and mark that one
  let img = readyBmp(S.photos[S.sel].url);
  let shownIdx = S.sel;
  if (img) {
    S._lastReady = { img, idx: S.sel };
  } else if (S._lastReady) {
    img = S._lastReady.img;
    shownIdx = S._lastReady.idx;
  }
  if (!img) return;
  const r = fitRect(img.width, img.height, w / dpr, h / dpr, 10);
  ctx.save(); ctx.scale(dpr, dpr);
  ctx.globalAlpha = S.state === 'ready' ? .42 : 1;
  ctx.drawImage(img, r.x, r.y, r.w, r.h);
  ctx.globalAlpha = 1;
  if (marks) drawRealMarks(ctx, r, shownIdx);
  ctx.restore();
}

/** the solver's actual keypoints, appearing as they are found */
function drawRealMarks(ctx, r, imgIdx) {
  const f = S.feats.get(imgIdx);
  const fr = S.session && S.session.frames && S.session.frames[imgIdx];
  if (!f || !fr || !f.x || !f.y) return;
  const sx = r.w / (fr.fw || 1), sy = r.h / (fr.fh || 1);
  ctx.fillStyle = 'rgba(47,212,193,.8)';
  const n = Math.min(f.n || 0, 1200);
  for (let k = 0; k < n; k++) {
    if (f.x[k] != null && f.y[k] != null) {
      ctx.fillRect(r.x + f.x[k] * sx - 1, r.y + f.y[k] * sy - 1, 2, 2);
    }
  }
  ctx.fillStyle = '#93a1a0';
  ctx.font = '500 10px "Spline Sans Mono", monospace';
  ctx.fillText(`${fmt(f.n)} LANDMARKS`, r.x + 4, r.y + r.h + 14);
  // capture-order debugging: the photo's EXIF time, bottom-right — a walk
  // must read as a monotonic clock here; n/a = no EXIF (name order decides)
  const name = (S.photos[imgIdx] || {}).name || '';
  const cap = S.capDates && S.capDates.get(name);
  const pad2 = (v) => String(v).padStart(2, '0');
  const label = cap
    ? `${pad2(new Date(cap).getHours())}:${pad2(new Date(cap).getMinutes())}:${pad2(new Date(cap).getSeconds())}`
    : name; // no EXIF: the file name IS the ordering key — show it instead
  ctx.textAlign = 'right';
  ctx.fillText(label, r.x + r.w - 4, r.y + r.h + 14);
  ctx.textAlign = 'left';
}

/** two photographs, and the matches that survived between them.
 *  The stage HOLDS what it shows: a newer pair replaces the current one only
 *  when both of its bitmaps are decoded and the current pair has had its
 *  PAIR_HOLD on screen. Matching emits many pairs a second and the display
 *  LRU rarely holds both photos of a fresh pair, so switching on every
 *  event was a black frame per decode and a photo swap per event (flicker,
 *  "hardly see the photos"). Evicted bitmaps are never closed (img.js), so
 *  keeping references here is safe. Before the first pair is ready, the
 *  single-photo stage stands in — never black. */
const PAIR_HOLD = 700;
function pairStage(ctx, w, h, dpr) {
  const now = performance.now();
  const cand = S.lastPairEv;
  let sp = S.shownPair;
  if (cand && (!sp || (cand !== sp.ev && now - sp.since >= PAIR_HOLD))) {
    const p1 = S.photos && S.photos[cand.i];
    const p2 = S.photos && S.photos[cand.j];
    const a = p1 && p1.url ? readyBmp(p1.url) : null;
    const b = p2 && p2.url ? readyBmp(p2.url) : null;
    if (a && b) sp = S.shownPair = { ev: cand, a, b, since: now };
  }
  if (!sp) return photoStage(ctx, w, h, dpr, true);
  ctx.fillStyle = '#070909';
  ctx.fillRect(0, 0, w, h);
  const { ev, a, b } = sp;
  const half = w / dpr / 2;
  ctx.save(); ctx.scale(dpr, dpr);
  const r1 = fitRect(a.width, a.height, half, h / dpr, 14);
  const r2 = fitRect(b.width, b.height, half, h / dpr, 14);
  r2.x += half;
  ctx.globalAlpha = .7;
  ctx.drawImage(a, r1.x, r1.y, r1.w, r1.h);
  ctx.drawImage(b, r2.x, r2.y, r2.w, r2.h);
  ctx.globalAlpha = 1;

  const fa = S.feats.get(ev.i), fb = S.feats.get(ev.j);
  const f1 = S.session && S.session.frames && S.session.frames[ev.i];
  const f2 = S.session && S.session.frames && S.session.frames[ev.j];
  if (fa && fb && f1 && f2 && fa.x && fb.x && ev.sample) {
    ctx.strokeStyle = 'rgba(47,212,193,.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k + 1 < ev.sample.length * 2 && k < 100; k += 2) {
      const pair = ev.sample[k / 2];
      if (!pair) continue;
      const [ia, ib] = pair;
      if (ia == null || ib == null || fa.x[ia] == null || fb.x[ib] == null) continue;
      const x1 = r1.x + fa.x[ia] * (r1.w / (f1.fw || 1)), y1 = r1.y + fa.y[ia] * (r1.h / (f1.fh || 1));
      const x2 = r2.x + fb.x[ib] * (r2.w / (f2.fw || 1)), y2 = r2.y + fb.y[ib] * (r2.h / (f2.fh || 1));
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    }
    ctx.stroke();
  }
  ctx.fillStyle = '#93a1a0';
  ctx.font = '500 10px "Spline Sans Mono", monospace';
  ctx.fillText(`FRAME ${ev.i + 1}`, r1.x + 4, r1.y - 6);
  ctx.fillText(`FRAME ${ev.j + 1}`, r2.x + 4, r2.y - 6);
  ctx.restore();
}

// ── details sheet ───────────────────────────────────────────────────────────
const DTABS = [['score', 'Score'], ['marks', 'Landmarks'], ['matches', 'Matching'], ['cams', 'Cameras'], ['perf', 'Timing'], ['gpu', 'GPU']];

// ── GPU facts: what this device is, for the Details sheet and bug reports ──
// The adapter probed at boot serves the wall (no session yet); a live
// session's own adapter/device replaces it (its limits are the ones granted).
/** the GPU's marketing name — WebGPU's adapter info stops at vendor +
 *  architecture ("nvidia blackwell"); the WebGL debug renderer string carries
 *  the model ("NVIDIA GeForce RTX 5080"). Read once, from a throwaway context. */
function webglName() {
  if (S._glName !== undefined) return S._glName;
  let name = '';
  try {
    const cv = document.createElement('canvas');
    const gl = cv.getContext('webgl2') || cv.getContext('webgl');
    const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
    let r = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : '');
    r = String(r || '');
    // "ANGLE (NVIDIA, NVIDIA GeForce RTX 5080 (0x00002C02) Direct3D11 vs_5_0 ps_5_0, D3D11)"
    // "ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)"
    // "ANGLE (Qualcomm, Adreno (TM) 740, OpenGL ES 3.2 V@0615.0 ...)"
    if (/^ANGLE \(/.test(r)) {
      const parts = r.slice(7, -1).split(', ');
      r = parts[1] || parts[0] || '';
    }
    name = r.replace(/^ANGLE \w+ Renderer: /, '').replace(/ \(0x[0-9A-Fa-f]+\)/, '')
      .replace(/ (Direct3D|OpenGL|Vulkan|Metal|vs_)\S*.*$/, '').trim();
    const lose = gl && gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch { /* no WebGL: the WebGPU words will do */ }
  S._glName = name;
  return name;
}

/** The machine a scene was trained on, in a few characters for the stats
 *  line: "PC · RTX 5080", "Mac · M2", "iPhone", "Android · Adreno 740". A
 *  phone's GPU string says nothing a reader wants ("Apple GPU"), so the
 *  handhelds are named by the device alone. */
function deviceLabel() {
  const ua = navigator.userAgent || '';
  const iPadOS = /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;   // iPadOS calls itself Macintosh
  const kind = /iPhone/.test(ua) ? 'iPhone'
    : (/iPad/.test(ua) || iPadOS) ? 'iPad'
    : /Android/.test(ua) ? 'Android'
    : /Macintosh|Mac OS X/.test(ua) ? 'Mac'
    : /Windows|Linux|CrOS/.test(ua) ? 'PC' : '';
  if (kind === 'iPhone' || kind === 'iPad') return kind;
  // the marketing name down to the part people recognise:
  // "NVIDIA GeForce RTX 5080" -> "RTX 5080", "Adreno (TM) 740" -> "Adreno 740"
  let gpu = webglName()
    .replace(/\((?:TM|R)\)/gi, '')
    .replace(/NVIDIA GeForce|NVIDIA|AMD Radeon|Radeon|Apple/gi, '')
    .replace(/Intel\(?R?\)?/gi, 'Intel')
    .replace(/\s+/g, ' ')
    .trim();
  if (gpu.length > 18) gpu = gpu.slice(0, 18).trim();
  if (!gpu || /^(WebKit|Apple GPU|Software)/i.test(gpu)) return kind;
  // a comma inside, because the stats line itself is separated by · already
  return kind ? `${kind}, ${gpu}` : gpu;
}

function gpuFacts(preferSaved = true) {
  if (preferSaved && S._savedGpuFacts && S.state !== 'train') {
    return { ...S._savedGpuFacts, ips: S.itersPerSec || S._savedGpuFacts.ips || 0 };
  }
  const g = S.session && S.session.gpu;
  const adapter = (g && g.adapter) || (S.gpuProbe && S.gpuProbe.adapter) || null;
  const device = (g && g.device) || null;
  const info = (g && g.info) || (adapter && adapter.info) || (S.gpuProbe && S.gpuProbe.info) || {};
  const lim = (device && device.limits) || (adapter && adapter.limits) || null;
  const feats = device ? [...device.features] : adapter ? [...adapter.features] : [];
  const gb = (v) => v ? `${(v / (1 << 30)).toFixed(v >= (1 << 30) ? 1 : 2)} GB` : '—';
  const nav = navigator;
  // model: the marketing name of the GPU Chrome's own process renders on
  // (WebGL). webgpu: the adapter the TRAINER gets. On a dual-GPU laptop these
  // can differ (Chrome on the Intel iGPU, WebGPU high-performance on the
  // NVIDIA) — then the training adapter leads and the browser GPU is named
  // separately, never pasted together into one false name.
  const model = webglName();
  const webgpu = [info.vendor, info.architecture, info.device].filter(Boolean).join(' ');
  const di = S.gpuProbe && S.gpuProbe.defaultInfo;
  const defaultGpu = di ? [di.vendor, di.architecture, di.device].filter(Boolean).join(' ') : '';
  const sameVendor = !!(model && info.vendor && model.toLowerCase().includes(info.vendor.toLowerCase()));
  const name = sameVendor ? `${model} (${info.architecture || 'WebGPU'})`
    : webgpu ? (model ? `${webgpu} (WebGPU) · browser on ${model}` : webgpu)
      : model || (S.gpuProbe && S.gpuProbe.failed ? 'no WebGPU adapter' : '…');
  return {
    ready: !!(adapter || device || S.gpuProbe),
    model, webgpu, sameVendor,
    defaultGpu: defaultGpu && defaultGpu !== webgpu ? defaultGpu : '',
    name,
    description: info.description || '',
    vendor: info.vendor || '—', architecture: info.architecture || '—', device: info.device || '—',
    granted: !!device,
    features: feats.filter((f) => /subgroups|shader-f16|timestamp-query|float32-filterable|bgra8unorm-storage/.test(f)),
    maxBufferSize: lim ? gb(lim.maxBufferSize) : '—',
    maxStorageBinding: lim ? gb(lim.maxStorageBufferBindingSize) : '—',
    workgroupStorage: lim ? `${(lim.maxComputeWorkgroupStorageSize / 1024).toFixed(0)} KB` : '—',
    invocations: lim ? String(lim.maxComputeInvocationsPerWorkgroup) : '—',
    texture2d: lim ? String(lim.maxTextureDimension2D) : '—',
    ua: nav.userAgent,
    platform: (nav.userAgentData && nav.userAgentData.platform) || nav.platform || '—',
    cores: nav.hardwareConcurrency || '—',
    memory: nav.deviceMemory ? `${nav.deviceMemory} GB` : '—',
    screen: `${screen.width}×${screen.height} @${devicePixelRatio}`,
    ips: S.itersPerSec || 0,
  };
}

/** the about card's "This device" line — filled when the card opens and
 *  again when the boot probe lands */
function renderAboutGpu() {
  const el = $('about-gpu');
  if (!el) return;
  const g = gpuFacts();
  el.hidden = false;
  const bits = [g.name];
  if (g.maxBufferSize !== '—') bits.push(`${g.maxBufferSize} buffers`);
  if (g.features.length) bits.push(g.features.filter((f) => /subgroups|shader-f16/.test(f)).join(', '));
  if (g.ips) bits.push(`${fmt(g.ips)} cycles/s`);
  $('about-gpu-txt').textContent = bits.filter(Boolean).join(' · ');
  const btn = $('about-gpu-copy');
  if (!btn.dataset.wired) {
    btn.dataset.wired = '1';
    // an icon button: the copied state shows as colour + title, not text
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(buildGpuReport());
        btn.dataset.done = '1'; btn.title = 'Copied ✓';
      } catch { btn.title = 'Copy failed'; }
      setTimeout(() => { btn.dataset.done = ''; btn.title = 'Copy GPU report'; }, 1600);
    });
  }
}

/** the GPU tab of the Details sheet (stat: the sheet's row builder) */
function gpuTab(stat) {
  const g = gpuFacts();
  return {
    cap: 'The graphics adapter this browser gives the trainer, and the limits it granted.',
    title: g.name,
    body: [
      g.description ? esc(g.description) : 'Vendor, architecture and device as WebGPU reports them; browsers deliberately keep this coarse.',
      'Training speed is mostly memory bandwidth: the same photos train at identical quality on ' +
      'a phone and a desktop card, dozens of times apart in cycles per second. Copy the report ' +
      'below when something trains slower than expected, or fails to start.',
    ],
    rows: [
      stat('Training adapter', esc(g.webgpu || '—'), 'accent'),
      stat(g.sameVendor ? 'Model' : 'Browser GPU (WebGL)', esc(g.model || '—')),
      g.defaultGpu ? stat('Default adapter', esc(g.defaultGpu)) : '',
      stat('Vendor', esc(g.vendor)),
      stat('Architecture', esc(g.architecture)),
      stat('Device', esc(g.device)),
      stat('Buffer limit', g.maxBufferSize, 'accent'),
      stat('Storage binding', g.maxStorageBinding),
      stat('Workgroup memory', g.workgroupStorage),
      stat('Features', g.features.length ? esc(g.features.join(', ')) : '—'),
      g.ips ? stat('Speed', `${fmt(g.ips)} <small>cycles/s</small>`) : '',
      stat('Screen', esc(g.screen)),
      stat('CPU cores', esc(String(g.cores))),
    ].filter(Boolean),
    btns: [{ label: 'Copy GPU report', fn: copyGpuReport }],
  };
}

function buildGpuReport() {
  const g = gpuFacts();
  const L = [];
  L.push(`splat.js gpu report — ${new Date().toISOString()}`);
  L.push(`url: ${location.href}`);
  L.push(`ua: ${g.ua}`);
  L.push(`platform: ${g.platform} · cores ${g.cores} · memory ${g.memory}`);
  L.push(`screen: ${g.screen}`);
  L.push(`training adapter (WebGPU, high-performance): ${g.webgpu || '—'}${g.description ? ` — ${g.description}` : ''}`);
  L.push(`browser gpu (WebGL): ${g.model || '?'}${g.sameVendor ? ' (same device)' : ''}`);
  if (g.defaultGpu) L.push(`default WebGPU adapter: ${g.defaultGpu} (training asks for high-performance instead)`);
  L.push(`limits: maxBufferSize ${g.maxBufferSize} · maxStorageBufferBindingSize ${g.maxStorageBinding} · workgroup storage ${g.workgroupStorage} · invocations ${g.invocations} · texture2d ${g.texture2d}`);
  L.push(`features: ${g.features.join(', ') || 'none of interest'}${g.granted ? ' (device granted)' : ' (adapter)'}`);
  if (S.session) {
    L.push(`session: ${S.state} · ${fmt(S.splats || 0)} splats · ${fmt(S.iter || 0)} cycles${g.ips ? ` · ${fmt(g.ips)} cycles/s` : ''}`);
    L.push(`settings: ${JSON.stringify(S._savedDetailSettings || S.settings)} · preset ${S.preset ? S.preset.id : (S._localRun && S._localRun.name) || '?'}`);
  }
  return L.join('\n');
}

async function copyGpuReport(btn) {
  const old = btn.textContent;
  try {
    await navigator.clipboard.writeText(buildGpuReport());
    btn.textContent = 'Copied ✓';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = old; }, 1600);
}

/** the sheet's state lives in the address: ?details=<tab> — a refresh keeps
 *  it open, and the link hands the same tab (GPU, say) to whoever it is sent
 *  to, on THEIR device */
function syncDetailsUrl() {
  const u = new URL(location.href);
  const was = u.searchParams.get('details');
  const now = $('details').hidden ? null : S.detailTab;
  if ((was || null) === now) return;
  if (now) u.searchParams.set('details', now); else u.searchParams.delete('details');
  history.replaceState(history.state, '', u);
}

function closeDetails() {
  $('details').hidden = true;
  syncTrainingVisualsControl();
  if (S.state === 'train') persistRunDetails(S.runId, { full: false }).catch(() => {});
  syncDetailsUrl();
}

function openDetails() {
  $('details').hidden = false;
  syncTrainingVisualsControl();
  if (!S.session) {
    // the wall: only the device itself has facts to show
    S.detailTab = 'gpu';
    renderDetails();
    syncDetailsUrl();
    return;
  }
  // per-photo scores: one render + readback per camera, started the first
  // time the sheet asks for them (not at finish, where they stole the GPU
  // from the viewer)
  if (S.state === 'done' && S.session && S.session.trainer && S.session.trainer.device &&
      S.scene && S._scoredGen !== S.gen) {
    S._scoredGen = S.gen;
    scoreFrames().catch(() => {});
  }
  $('d-export').replaceChildren(buildExport());
  renderDetails();
  syncDetailsUrl();
}

function renderDetails() {
  if (!S.session) return renderDetailsNoSession();
  const ses = S.session, recon = ses.recon;
  const n = S.photos.length || (ses.frames && ses.frames.length) || 0;
  const placed = recon.cams.length;
  const pointCount = S.scene && S.scene.xyz && S.scene.xyz.length
    ? Math.floor(S.scene.xyz.length / 3) : recon.points.length;
  $('d-sub').textContent = `${S.preset.name} · ` +
    // pano sets have no 1:1 photograph list — count placed views instead
    (n ? `${n} photographs · ${placed} placed · ` : `${placed} views placed · `) +
    `${fmt(pointCount)} points · ${fmt(S.splats)} splats`;

  // a restored share never ran the solve here — tabs whose stats are empty
  // (landmarks, matching, timing) simply don't appear
  const have = {
    marks: S.feats.size > 0,
    matches: S.solveStats.pairsChecked > 0,
    perf: !!(ses.perf && ses.perf.frames && ses.perf.frames.length),
  };
  const tabs = DTABS.filter(([id]) => have[id] !== false);
  if (!tabs.some(([id]) => id === S.detailTab)) S.detailTab = 'score';
  const segHost = $('d-seg');
  segHost.innerHTML = '';
  tabs.forEach(([id, label]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(S.detailTab === id));
    b.addEventListener('click', () => { S.detailTab = id; renderDetails(); syncDetailsUrl(); });
    segHost.appendChild(b);
  });

  const stat = (k, v, tone) =>
    `<div class="stat"><span class="stat-k">${k}</span>
     <span class="stat-v"${tone ? ` data-tone="${tone}"` : ''}>${v}</span></div>`;

  const featsTotal = [...S.feats.values()].reduce((a, f) => a + f.n, 0);
  const selFeats = (S.feats.get(S.sel) || {}).n;
  const survived = S.solveStats.pairsChecked
    ? Math.round(100 * S.solveStats.pairsUsable / S.solveStats.pairsChecked) : 0;

  const gap = (shownScore('train') != null && shownScore('hold') != null) ? shownScore('train') - shownScore('hold') : null;
  const pf = (ses.perf && ses.perf.frames) || [];
  const colv = (i) => pf.map((r) => r[i]);
  const ipsAll = pf.length > 1
    ? (pf[pf.length - 1][1] - pf[0][1]) / Math.max(.001, (pf[pf.length - 1][0] - pf[0][0]) / 1000) : 0;
  const metCosts = colv(7).filter((v) => v > 0);
  const bench = !!(S.session && S.session.holdout >= 0);   // ?eval mode

  const T = {
    score: {
      cap: bench
        ? 'Turquoise: the photos it trained on. Amber: the held-out test photos.'
        : 'How closely the splats match the photographs.',
      title: 'Score over the run',
      body: [
        'Each cycle renders the splats from one photo\'s viewpoint and nudges them to ' +
        'shrink the difference to that photograph. Higher dB is better; +3 dB halves the error.',
        bench
          ? 'The amber photos never trained the model, so their score only rises when the 3D is ' +
            'actually right — the turquoise curve can also rise by memorising.'
          : 'Every photo trains the model here. Add ?eval to the address to hold every 8th ' +
            'photo out of training and score those instead — the honest benchmark number.',
      ],
      rows: [
        stat('Cycles', fmt(S.iter)),
        shownScore('train') != null ? stat('Trained photos', `${shownScore('train').toFixed(1)} <small>dB</small>`, 'accent') : '',
        shownScore('hold') != null ? stat('Hidden photo', `${shownScore('hold').toFixed(1)} <small>dB</small>`, 'alt') : '',
        S.psnrTest != null ? stat(`Test photos (${S.psnrTest.frames.length})`, `${S.psnrTest.psnr.toFixed(2)} <small>dB</small>`, 'alt') : '',
        gap != null ? stat('Gap', `${gap.toFixed(1)} <small>dB</small>`, Math.abs(gap) < 1.5 ? 'accent' : undefined) : '',
        stat('Splats', fmt(S.splats)),
        stat('Exported file', `${(S.splats * 164 / 1e6).toFixed(1)} <small>MB</small>`),
        stat('Time', `${S.minutes} <small>min</small>`),
      ].filter(Boolean),
      btns: bench ? [{
        label: 'Look at a frame it never saw',
        fn: () => {
          const h = S.scene.cams.find((c) => c.state === 'holdout');
          closeDetails();
          S.compare = 'swipe';
          select(h ? h.i : S.sel);
        },
      }] : [],
    },
    marks: {
      cap: `Photo ${S.sel + 1} of ${n} — flat sky and plain walls stay empty.`,
      title: 'Spots worth remembering',
      body: [
        'Before there is any 3D, every photo is scanned for places that could be recognised ' +
        'again from another angle: corners, texture, edges. Each one gets a short numeric ' +
        'fingerprint of its surroundings.',
        'Smooth surfaces produce nothing, which is exactly why blank walls, water and sky are ' +
        'hard for this kind of reconstruction.',
      ],
      rows: [
        stat('Marks on this frame', selFeats != null ? fmt(selFeats) : '—'),
        stat('Average per photo', fmt(featsTotal / Math.max(1, S.feats.size))),
        stat('Across the set', fmt(featsTotal)),
      ],
    },
    matches: {
      cap: 'The pairings that survived the geometry test, drawn between two frames.',
      title: 'The same spot, twice',
      body: [
        'Fingerprints are compared photo against photo. Plenty of pairings are wrong, so every ' +
        'candidate set is tested against geometry: only pairings that could be explained by one ' +
        'rigid scene seen from two positions survive.',
        'What survives is a chain — a spot tracked through many photos at once — and that chain ' +
        'is what makes a position solvable.',
      ],
      rows: [
        stat('Pairs compared', fmt(S.solveStats.pairsChecked)),
        stat('Survived the test', `${fmt(S.solveStats.pairsUsable)} · ${survived}%`, 'accent'),
      ],
    },
    cams: {
      cap: 'The sparse cloud and the position of every photograph. Drag to orbit.',
      title: 'Where the camera was',
      body: [
        'A spot seen from two known directions fixes a point in space; a photo with enough known ' +
        'points fixes a camera. Solved together they give both — the positions, and a sparse ' +
        'cloud of a few thousand points.',
        'That cloud is far too coarse to look at. Its job is to say roughly where surfaces are, ' +
        'so the splats do not start from nothing.',
      ],
      rows: [
        stat('Placed', `${placed} <small>/ ${n}</small>`, placed === n ? 'accent' : 'red'),
        stat('Points', fmt(recon.points.length)),
        stat('Reprojection error', recon.rmsBA ? `${recon.rmsBA.toFixed(2)} <small>px</small>` : '—',
          recon.rmsBA && recon.rmsBA < 1 ? 'accent' : undefined),
        stat('Focal length', `${Math.round(recon.cams[0].f)} <small>px, solved — no lens data was read</small>`),
        stat('Solve time', `${Math.round(S.solveStats.solveSec)} <small>s</small>`),
      ],
    },
    perf: {
      cap: 'Every submitted batch: encode, view render, GPU wait, score readback — in milliseconds.',
      title: 'Where the time went',
      body: [
        'The loop times itself as it runs: how long each batch of cycles takes to encode, how ' +
        'long the GPU makes it wait, and what the score readbacks cost — a readback has to ' +
        'drain everything queued before it can measure.',
        'Speeds differ mostly by memory bandwidth: a phone GPU sits dozens of times below a ' +
        'desktop card, at identical quality. The downloaded log is the file to attach when ' +
        'something is slower than it should be.',
      ],
      rows: [
        stat('GPU', (() => {
          const gi = (S.session && S.session.gpu && S.session.gpu.info) || S._savedGpuFacts || {};
          return esc([gi.vendor, gi.architecture].filter(Boolean).join(' ')) || '—';
        })()),
        stat('Speed', ipsAll ? `${fmt(ipsAll)} <small>cycles/s</small>` : '—', 'accent'),
        stat('GPU per cycle', ipsAll ? `${(1000 / ipsAll).toFixed(1)} <small>ms</small>` : '—'),
        stat('Cycles per submit', pf.length ? fmt(pctl(colv(2), .5)) : '—'),
        stat('Score readback', metCosts.length ? `${Math.round(pctl(metCosts, .5))} <small>ms median</small>` : '—'),
        stat('GPU wait', pf.length ? `${Math.round(pctl(colv(6), .9))} <small>ms p90</small>` : '—'),
      ],
      btns: [
        { label: 'Download log', fn: downloadPerfLog },
        { label: 'Copy to clipboard', fn: copyPerfLog },
      ],
    },
    gpu: S.detailTab === 'gpu' ? gpuTab(stat) : null,
  }[S.detailTab];

  $('d-prev').hidden = $('d-next').hidden = S.detailTab !== 'marks' || !S.photos.length;

  // the visual slot: the photo/pair/cameras canvas, the score chart, or the log
  const vis = (S.detailTab === 'perf' || S.detailTab === 'gpu') ? 'perf' : S.detailTab === 'score' ? 'chart' : 'cv';
  // Mobile browsers otherwise claim one-finger drags as page scrolling and
  // cancel the pointer stream before OrbitController can keep tracking it.
  $('d-cv').classList.toggle('camera-orbit', S.detailTab === 'cams');
  $('d-cv').hidden = vis !== 'cv';
  $('d-chart-wrap').hidden = vis !== 'chart';
  $('d-chart').hidden = vis !== 'chart';
  $('d-perf').hidden = vis !== 'perf';
  if (vis === 'perf') $('d-perf').textContent = S.detailTab === 'gpu' ? buildGpuReport() : buildPerfReport();
  if (vis === 'chart') {
    if (!dchart) {
      dchart = new Chart($('d-chart'), {});
      $('d-chart-scale')?.addEventListener('click', toggleScoreScale);
    }
    dchart.maxIter = S.maxIters;
    // a restored share has no live loss history — its recorded training
    // curve (recon stats) is the same line, saved at share time
    const rstats = (S.restored && S.restored.stats) || {};
    const hist = (S.session.lossHistory && S.session.lossHistory.length)
      ? S.session.lossHistory : (rstats.chart || []);
    dchart.train = hist.map(([i, v]) => [i, v]);
    dchart.hold = chart ? chart.hold : ((S.holdHist && S.holdHist.length ? S.holdHist : rstats.holds) || []).slice();
    dchart.events = S.chartEvents.map((e) => ({ ...e, at: e.iter / S.maxIters }));
    dchart.resize();
    dchart.draw();
  }

  $('d-cap').textContent = T.cap;
  $('d-txt').innerHTML =
    `<h3>${T.title}</h3>${T.body.map((p) => `<p>${p}</p>`).join('')}<div class="grp">${T.rows.join('')}</div>` +
    (T.btns ? `<div class="tabbtns">${T.btns.map((b, i) =>
      `<button class="btn btn-quiet" data-bi="${i}">${b.label}</button>`).join('')}</div>` : '');
  if (T.btns) {
    $('d-txt').querySelectorAll('[data-bi]').forEach((el) =>
      el.addEventListener('click', () => T.btns[el.dataset.bi].fn(el)));
  }

  if (S.detailTab === 'cams' && !dvp) {
    dvp = new Viewport($('d-cv'));
    dvp.setScene(S.scene);
    dvp.setUp(vp.up);
    // frame the CAMERA PATH, not the whole cloud — the cloud's far shell
    // would push the rig to a speck
    const placed = S.scene.cams.filter((c) => c.R).map(camCentre);
    if (placed.length) {
      const ctr = [0, 1, 2].map((k) => placed.reduce((a, p) => a + p[k], 0) / placed.length);
      const rad = Math.sqrt(placed.reduce((a, p) =>
        Math.max(a, (p[0] - ctr[0]) ** 2 + (p[1] - ctr[1]) ** 2 + (p[2] - ctr[2]) ** 2), 0));
      dvp.target = ctr;
      dvp.dist = Math.max(0.5, rad * 2.6);
    }
  }
  if (dvp) dvp.resize();
}

/** the sheet on the wall, before any run: the GPU tab alone */
function renderDetailsNoSession() {
  S.detailTab = 'gpu';
  $('d-sub').textContent = 'This device · no run open';
  $('d-export').replaceChildren();
  const segHost = $('d-seg');
  segHost.innerHTML = '';
  const b = document.createElement('button');
  b.textContent = 'GPU';
  b.setAttribute('aria-pressed', 'true');
  segHost.appendChild(b);
  const stat = (k, v, tone) =>
    `<div class="stat"><span class="stat-k">${k}</span>
     <span class="stat-v"${tone ? ` data-tone="${tone}"` : ''}>${v}</span></div>`;
  const T = gpuTab(stat);
  $('d-prev').hidden = $('d-next').hidden = true;
  $('d-cv').hidden = true;
  $('d-chart').hidden = true;
  $('d-perf').hidden = false;
  $('d-perf').textContent = buildGpuReport();
  $('d-cap').textContent = T.cap;
  $('d-txt').innerHTML =
    `<h3>${T.title}</h3>${T.body.map((p) => `<p>${p}</p>`).join('')}<div class="grp">${T.rows.join('')}</div>` +
    `<div class="tabbtns">${T.btns.map((b2, i) => `<button class="btn btn-quiet" data-bi="${i}">${b2.label}</button>`).join('')}</div>`;
  $('d-txt').querySelectorAll('[data-bi]').forEach((el) =>
    el.addEventListener('click', () => T.btns[el.dataset.bi].fn(el)));
}

/** Landmarks tab: step through the photos (the filmstrip is under the sheet) */
function detailFlip(dir) {
  const n = S.photos.length;
  if (!n) return;
  S.sel = (S.sel + dir + n) % n;
  if (S.photos[S.sel]?.url) bmp(S.photos[S.sel].url);                                  // decode now
  const next = (S.sel + dir + n) % n;
  if (S.photos[next]?.url) bmp(S.photos[next].url);                  // prefetch onward
  renderDetails();
}

function drawDetail() {
  // score = the chart canvas, perf/gpu = the log pre — none repaints per frame
  if (S.detailTab === 'perf' || S.detailTab === 'gpu' || S.detailTab === 'score' || !S.session) return;
  const cv = $('d-cv');
  if (!cv.clientWidth) return;
  if (S.detailTab === 'cams') {
    if (!dvp) return;
    const r = cv.getBoundingClientRect();
    if (Math.abs(r.width * (dvp.dpr || 1) - cv.width) > 2) dvp.resize();
    dvp.draw({ points: true, cams: S.scene.cams, showCams: true, showPath: true, sel: S.sel, active: -1 });
    return;
  }
  const dpr = Math.min(2, devicePixelRatio || 1);
  if (cv.width !== Math.round(cv.clientWidth * dpr)) {
    cv.width = Math.round(cv.clientWidth * dpr);
    cv.height = Math.round(cv.clientHeight * dpr);
  }
  const ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  const sourceUnavailable = () => {
    ctx.fillStyle = '#070909'; ctx.fillRect(0, 0, w, h);
    ctx.save(); ctx.scale(dpr, dpr);
    ctx.fillStyle = '#93a1a0'; ctx.font = '500 12px "Spline Sans Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('The saved landmarks and match counts are available,', w / dpr / 2, h / dpr / 2 - 8);
    ctx.fillText('but the original photos are not on this device.', w / dpr / 2, h / dpr / 2 + 12);
    ctx.restore();
  };
  if (S.detailTab === 'marks') {
    ctx.fillStyle = '#070909'; ctx.fillRect(0, 0, w, h);
    const p = S.photos && S.photos[S.sel];
    if (!p || !p.url) { sourceUnavailable(); return; }
    const img = readyBmp(p.url);
    if (!img) return;
    const r = fitRect(img.width, img.height, w / dpr, h / dpr, 6);
    ctx.save(); ctx.scale(dpr, dpr);
    ctx.drawImage(img, r.x, r.y, r.w, r.h);
    drawRealMarks(ctx, r, S.sel);
    ctx.restore();
  } else if (!S.photos.length || !S.photos.some((p) => p && p.url)) {
    sourceUnavailable();
  } else {
    pairStage(ctx, w, h, dpr);
  }
}
