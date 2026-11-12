// browser/app.js — the browser-only trainer: photos, video, or an exported
// .zip in, a standard 3DGS .ply out. No capture, no device features — just
// the vendored engine (./src, byte-synced from the repo's src/ — see
// tools/sync-browser.sh) driven start to finish in one tab.
//
// Serve THIS FOLDER over http (ES modules + WebGPU forbid file://):
//   python3 -m http.server 8000   →   http://localhost:8000/

import {
  createSession, parseCameras, extractSharpFrames, isVideoFile, solveTierOpts,
  buildLODPlan, undistortFrames,
} from './src/index.js';
import { readZip } from './zip.js';
import { validateDataset, measuredSeeds, levelingNote } from './app/js/xr-dataset.js';
import { gravityUpForKnownRun, decodeModel, parseState, parsePlyGaussians,
  sogToGaussians, continuedInputDim, continuedBuf, buildReconJson, buildSessionZip,
  convertSHDegree } from './app/js/session_io.js';
import { plyToSog } from './app/js/sog.js';
import { zipStore } from './app/js/zip.js';
import { saveBlob } from './app/js/download.js';
import { Chart, psnrToApproxSsimPercent } from './app/js/chart.js';
import { saveRun, patchRun, listRuns, deleteRun,
  saveLastCapture, loadLastCapture, deleteLastCapture } from './app/js/store.js';

const $ = (id) => document.getElementById(id);
const ui = {
  nogl: $('nogl'), files: $('files'), drop: $('drop'),
  q: $('opt-q'), res: $('opt-res'), feat: $('opt-feat'), tier: $('opt-tier'),
  buf: $('opt-buf'), sh: $('opt-sh'), splats: $('opt-splats'),
  lod: $('opt-lod'), lodCount: $('opt-lod-count'), mcmc: $('opt-mcmc'), softprune: $('opt-softprune'), softpruneCustom: $('opt-softprune-custom'),
  iters: $('opt-iters'), gpu: $('opt-gpu'), gpuV: $('opt-gpu-v'), gap: $('opt-gap'),
  summary: $('input-summary'), start: $('btn-start'),
  run: $('run'), stage: $('stage-label'), bar: $('stage-bar'), metrics: $('metrics'),
  pause: $('btn-pause'), finish: $('btn-finish'), prune: $('btn-prune'), ply: $('btn-ply'), simplify: $('btn-simplify'), sess: $('btn-session'),
  imgs: $('btn-photos'), cont: $('btn-continue'), tour: $('btn-tour'), strip: $('strip'),
  needphotos: $('needphotos'), photosHint: $('photos-hint'), photosFile: $('photos-file'),
  lodlinks: $('lodlinks'), status: $('status'),
  viewer: $('viewer'), view: $('view'), recenter: $('btn-recenter'),
  logwrap: $('logwrap'), log: $('log'),
  chartCv: $('chart'), scale: $('btn-scale'),
  lastcap: $('lastcap'), lastcapText: $('lastcap-text'),
  lastcapUse: $('lastcap-use'), lastcapForget: $('lastcap-forget'),
  library: $('library'), runs: $('runs'),
  details: $('details'), detailBody: $('detail-body'),
  tabs: { score: $('tab-score'), perf: $('tab-perf'), cams: $('tab-cams'), gpu: $('tab-gpu') },
};

const S = {
  staged: null,     // { kind: 'photos'|'video'|'zip'|'model'|'model-url', ... } — validated at Start
  capture: null,    // validated AR/capture dataset (transforms + seeds + metadata), if any
  restored: null,   // { gaussians, reconJson, state, label } — a model opened for viewing / continue
  session: null,
  lodPlan: null,    // LOD training plan (levels stepped in the metrics handler)
  maxIters: 30000,
  phase: 'idle',    // idle → prep → solve → train → done | error
  orbit: null,
  lines: 0,
  // score + details state (mirrors the full app's dock)
  scoreScale: 'db', // 'db' | 'similarity'
  detailTab: 'score',
  chartEvents: [],  // [{ iter, kind, label }] — refine markers on the curve
  holdHist: [],     // [iter, psnrHold]
  perfMetrics: [],  // [t_ms, iter, it/s, psnrTrain, psnrHold]
  psnrTrain: null, psnrHold: null, itersPerSec: 0,
  etaAt: null, trainT0: 0, minutes: 0,
  runSettings: null, runName: '',
  gpuFacts: null,
  runId: null,      // the library record this run secures into (null = none yet)
  loadedFiles: [],  // [{ name, url|null }] — the recon's photo source list
  lastPersist: 0,   // last progress patch to the library record
  tour: null,       // the camera fly-through, while it plays
  stripUrls: [],    // object URLs behind the photo strip (revoked on reset)
  photoBlobs: [],   // the run's photo Files/sources (photos.zip, strip)
  sogBlob: null,    // the compressed model, built once per run on demand
  _sogJob: null,
};

const fmt = (n) => n == null || Number.isNaN(n) ? '—' : Math.round(n).toLocaleString('en-US');
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
// the full app's phone class (coarse pointer, small screen): it sizes the
// auto splat budget, the seed target, and the memory-saver decode path
const phoneClass = () => matchMedia('(any-pointer: coarse)').matches &&
  Math.min(screen.width, screen.height) <= 820;
/** Where the GPU training core lives: the worker keeps encodes/submits off
 *  the UI thread — but its OffscreenCanvas view path never presented
 *  reliably on phone WebViews (black + laggy where the main-thread canvas
 *  is smooth on the same phone), so phones train locally like the full app.
 *  ?trainworker=1 forces the worker back for A/B; ?trainworker=0 forces
 *  local anywhere. */
function trainWorkerOpt() {
  const q = new URLSearchParams(location.search).get('trainworker');
  if (q === '1' || q === 'true') return true;
  if (q === '0' || q === 'false') return false;
  return phoneClass() ? false : 'auto';
}

// ── settings: the same training knobs as the full app's start card ─────────
// select values mirror the app's settings shape ('' = Auto/off, -1 = full
// size, mcmc '' = on / '0' = off) so the session wiring below reads identically.
/** Effective soft-prune trim: '' off, else a decimal fraction string. A
 *  Custom selection reads the adjacent percent input (clamped 1–50%). */
function readSoftprune() {
  if (ui.softprune.value !== 'custom') return ui.softprune.value;
  const pct = Math.min(50, Math.max(1, Math.round(parseFloat(ui.softpruneCustom.value) || 0)));
  return String(pct / 100);
}
function readSettings() {
  return {
    res: ui.res.value === '' ? 0 : parseInt(ui.res.value, 10),
    feat: ui.feat.value === '' ? 0 : parseInt(ui.feat.value, 10),
    buf: parseFloat(ui.buf.value) || 1,
    sh: parseInt(ui.sh.value, 10),
    splats: ui.splats.value === '' ? 0 : parseInt(ui.splats.value, 10),
    solve: ui.tier.value,
    lod: ui.lod.value === '1',
    lodLevels: Math.max(0, parseInt(ui.lodCount.value, 10) || 0),
    mcmc: ui.mcmc.value,
    softprune: readSoftprune(),
    gpu: clamp(parseInt(ui.gpu.value, 10) || 100, 10, 100),
    iters: parseInt(ui.iters.value, 10) || 30000,
  };
}

// quality macros: the one-knob row that drives the individual rows. Same
// values as the full app; Standard is this panel's defaults. Showcase's
// 100k exceeds this menu — the browser caps at its 60k full run.
const QKEYS = ['res', 'buf', 'sh', 'iters', 'splats', 'solve'];
const QMACROS = {
  draft:    { res: 480,  buf: 1, sh: 0, iters: 10000, splats: 0, solve: 'quick' },
  standard: { res: 0,    buf: 1, sh: 3, iters: 30000, splats: 0, solve: 'standard' },
  high:     { res: 1280, buf: 1, sh: 3, iters: 40000, splats: 0, solve: 'standard' },
  showcase: { res: 1280, buf: 1, sh: 3, iters: 60000, splats: 0, solve: 'standard' },
};
function qualityOf(st) {
  if (st.lod || st.lodLevels) return 'custom';
  for (const [k, m] of Object.entries(QMACROS)) {
    if (QKEYS.every((f) => st[f] === m[f])) return k;
  }
  return 'custom';
}
function pickQuality(v) {
  const m = QMACROS[v];
  if (!m) return;   // Custom is a display state, not a choice
  ui.res.value = m.res === 0 ? '' : String(m.res);
  ui.buf.value = String(m.buf);
  ui.sh.value = String(m.sh);
  ui.iters.value = String(m.iters);
  ui.splats.value = m.splats === 0 ? '' : String(m.splats);
  ui.tier.value = m.solve;
  ui.lod.value = '';
  ui.lodCount.value = '0';
  syncQuality();
}
/** The macro mirrors the rows (anything unmatched shows Custom); the LOD
 *  snapshot count only enables where a plan can exist (1M+ budget). */
function syncQuality() {
  const st = readSettings();
  ui.q.value = qualityOf(st);
  ui.lodCount.disabled = !(st.lod && st.splats >= 1000000);
}

function log(m) {
  const at = new Date().toLocaleTimeString([], { hour12: false });
  ui.log.textContent += `[${at}] ${m}\n`;
  ui.logwrap.hidden = false;
  if (++S.lines > 600) {  // keep the tab light on long runs
    const cut = ui.log.textContent.indexOf('\n', ui.log.textContent.length >> 1);
    ui.log.textContent = ui.log.textContent.slice(cut + 1);
  }
  ui.log.scrollTop = ui.log.scrollHeight;
}

function setStatus(msg, kind = '') {
  ui.status.textContent = msg;
  ui.status.className = `status${kind ? ` ${kind}` : ''}`;
}

function setStage(label, frac = null) {
  ui.stage.textContent = label;
  if (frac != null) ui.bar.style.width = `${clamp(frac * 100, 0, 100).toFixed(1)}%`;
}

// ── intake ──────────────────────────────────────────────────────────────────

const IMG_EXT = /\.(jpe?g|png|webp|bmp|gif|avif|tif?f|heic|heif)$/i;
const MIME = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp',
  gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif', tif: 'image/tiff',
  tiff: 'image/tiff', heic: 'image/heic', heif: 'image/heif' };
const isZipFile = (f) => /\.zip$/i.test(f.name || '') || f.type === 'application/zip';
const baseName = (p) => String(p).split(/[\\/]/).pop();
const DEG_OF_SHK = { 0: 0, 3: 1, 8: 2, 15: 3 };

function stageFiles(list) {
  resetRun();
  const files = [...list];
  if (!files.length) { ui.summary.textContent = 'Nothing selected yet.'; return; }
  const zips = files.filter(isZipFile);
  if (zips.length) {
    if (files.length > 1) {
      ui.summary.textContent = 'Drop either one .zip or loose files — not both at once.';
      ui.start.disabled = true;
      S.staged = null;
      return;
    }
    S.staged = { kind: 'zip', file: zips[0] };
    ui.summary.textContent = `${zips[0].name} — will be unpacked when training starts.`;
    ui.start.disabled = false;
    rememberCapture('zip', [zips[0]]);
    return;
  }
  const images = files.filter((f) => (f.type || '').startsWith('image/') || IMG_EXT.test(f.name || ''));
  const videos = files.filter(isVideoFile);
  const jsons = files.filter((f) => /\.json$/i.test(f.name || ''));
  const models = files.filter((f) => /\.(ply|sog)$/i.test(f.name || ''));
  if (models.length) {
    // a bare model opens in the viewer; with its recon.json alongside it
    // also knows the camera path (tour, look-through) and can continue
    const rest = files.filter((f) => f !== models[0]);
    const reconFile = rest.length === 1 && /\.json$/i.test(rest[0].name || '') ? rest[0] : null;
    if (models.length > 1 || (rest.length && !reconFile)) {
      ui.summary.textContent = 'Drop either one model (.ply / .sog / session .zip, optionally + its recon.json) or training input — not both at once.';
      ui.start.disabled = true;
      S.staged = null;
      return;
    }
    S.staged = { kind: 'model', file: models[0], reconFile };
    ui.summary.textContent = `${models[0].name}${reconFile ? ` + ${reconFile.name}` : ''} — will open as a finished model.`;
    ui.start.disabled = false;
    return;
  }
  if (videos.length && images.length < 3) {
    S.staged = { kind: 'video', file: videos[0] };
    ui.summary.textContent = `${videos[0].name} — sharp frames will be picked from it automatically.`;
    ui.start.disabled = false;
    rememberCapture('video', [videos[0]]);
    return;
  }
  if (!images.length) {
    ui.summary.textContent = videos.length
      ? 'That video came with photos too — but fewer than 3 are usable. Use at least 3 overlapping views; 20–200 are recommended.'
      : 'No usable input: drop photos, a video, or a .zip.';
    ui.start.disabled = true;
    S.staged = null;
    return;
  }
  S.staged = { kind: 'photos', images, jsons, ignoredVideo: videos[0] || null };
  const bits = [`${images.length} photos`];
  if (jsons.length) bits.push(`${jsons.length} JSON sidecar${jsons.length > 1 ? 's' : ''}`);
  if (videos[0]) bits.push(`(video ${videos[0].name} ignored — photos take precedence)`);
  ui.summary.textContent = `${bits.join(' · ')}.`;
  ui.start.disabled = images.length < 3;
  rememberCapture('photos', images);
  if (images.length < 3) ui.summary.textContent = 'Use at least 3 overlapping views; 20–200 are recommended.';
}

// ── zip intake (reading lives in ./zip.js — DOM-free and unit-tested) ───────

function parseJsonEntry(entries, name) {
  const raw = entries.get(name);
  if (!raw) return null;
  try {
    return JSON.parse(new TextDecoder().decode(raw));
  } catch {
    throw new Error(`${baseName(name)} is not valid JSON`);
  }
}

/** A zip becomes the loose-input shape: images as Files, sidecar JSONs, or
 *  a lone video to extract from. Throws user-facing errors otherwise. */
async function unpackZip(file, onTick) {
  const entries = await readZip(new Uint8Array(await file.arrayBuffer()));
  const names = [...entries.keys()];
  const images = [], jsons = [], videos = [];
  let dupes = 0;
  const seen = new Set();
  let i = 0;
  for (const name of names) {
    const base = baseName(name);
    if (!base || base.startsWith('.')) continue;
    if (IMG_EXT.test(base)) {
      if (seen.has(base.toLowerCase())) { dupes++; continue; }
      seen.add(base.toLowerCase());
      const ext = base.split('.').pop().toLowerCase();
      images.push(new File([entries.get(name)], base, { type: MIME[ext] || 'image/jpeg' }));
    } else if (/\.json$/i.test(base)) {
      jsons.push({ name: base, json: parseJsonEntry(entries, name) });
    } else if (isVideoFile({ name: base, type: '' })) {
      videos.push(new File([entries.get(name)], base, { type: 'video/mp4' }));
    }
    if (++i % 25 === 0) onTick(i / names.length);
  }
  onTick(1);
  if (dupes) log(`zip: ${dupes} duplicate file name${dupes > 1 ? 's' : ''} skipped (first copy kept)`);
  // a trained-session export (recon.json / state.bin / model.sog / model.ply)
  // restores as a finished model — with its checkpoint it can also continue
  const hasModel = ['recon.json', 'state.bin', 'model.sog', 'model.ply'].some((n) =>
    names.some((e) => baseName(e).toLowerCase() === n));
  if (!images.length && !videos.length && !hasModel) {
    throw new Error(`No images in this zip (saw: ${names.slice(0, 6).map(baseName).join(', ') || 'nothing readable'}). ` +
      'A capture .zip holds the JPEGs next to transforms.json — unzip-and-drop also works.');
  }
  return { images, jsons, videos, modelEntries: hasModel ? entries : null };
}

// ── the run ─────────────────────────────────────────────────────────────────

/** Loose photos + their JSON sidecars become the app's capture shape
 *  ({ kind, files, transforms, points, pointsByFrame, metadata }) — the same
 *  object the phone builds for an ARCore capture, validated the same way
 *  (unique names, a camera per photo, zero skew, well-formed seeds).
 *  Returns null when no camera file is present (plain photo set). Throws
 *  user-facing errors otherwise. */
function assembleCapture(files, jsons) {
  let transforms = null;
  for (const j of jsons) {
    const low = j.name.toLowerCase();
    const doc = j.json;
    if (low === 'transforms.json' || low === 'cameras.json' ||
        (doc && (Array.isArray(doc.frames) || Array.isArray(doc.cameras)))) {
      if (transforms) throw new Error('Choose only one camera JSON file with the matching photos.');
      transforms = doc;
    }
  }
  if (!transforms) return null;
  const seedDoc = jsons.find((j) => j.name.toLowerCase() === 'seed_points.json')?.json;
  const metaDoc = jsons.find((j) => j.name.toLowerCase() === 'capture.json')?.json;
  return validateDataset({
    kind: 'ar', files, transforms,
    points: (seedDoc && Array.isArray(seedDoc.points)) ? seedDoc.points : [],
    pointsByFrame: (seedDoc && seedDoc.pointsByFrame) || (metaDoc && metaDoc.pointsByFrame) || null,
    metadata: (metaDoc && typeof metaDoc === 'object') ? metaDoc : {},
  });
}

// ── capture sequence: EXIF time rules when present ──────────────────────────
// The picker hands files over in SELECTION order — restore the capture
// sequence before anything consumes them (same rule as the full app: EXIF
// DateTimeOriginal, else numeric name order when the whole set has no EXIF).

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

/** Restore the capture sequence in place (see above). */
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
  return files;
}

async function gatherInputs(staged) {
  if (staged.kind === 'zip') {
    setStage(`Unpacking ${staged.file.name} …`, 0);
    const { images, jsons, videos, modelEntries } = await unpackZip(staged.file, (f) =>
      setStage(`Unpacking ${staged.file.name} …`, f));
    log(`zip: ${images.length} images, ${jsons.length} JSON, ${videos.length} video`);
    if (images.length >= 3) {
      await sortByCapture(images);
      return { files: images, capture: assembleCapture(images, jsons), video: false };
    }
    if (modelEntries) return { files: [], restoreEntries: modelEntries, video: false };
    if (videos.length) return gatherVideo(videos[0]);
    throw new Error(`Only ${images.length} usable image${images.length === 1 ? '' : 's'} in this zip — need at least 3.`);
  }
  if (staged.kind === 'video') return gatherVideo(staged.file);
  const jsons = [];
  for (const f of staged.jsons) {
    try { jsons.push({ name: f.name, json: JSON.parse(await f.text()) }); }
    catch { throw new Error(`${f.name} is not valid JSON`); }
  }
  await sortByCapture(staged.images);
  return { files: staged.images, capture: assembleCapture(staged.images, jsons), video: false };
}

async function gatherVideo(file) {
  const gap = parseFloat(ui.gap.value) || 0;
  log(`video: ${file.name} (frame spacing ${gap > 0 ? `~${gap}s` : 'auto'})`);
  const { frames, duration } = await extractSharpFrames(file, {
    ...(gap > 0 ? { targetGapSec: gap } : {}),
    log: (m) => log(`[video] ${m}`),
    onProgress: (e) => {
      const half = e.stage === 'scan' ? 0 : 0.5;
      const label = e.stage === 'scan' ? 'Scoring every frame' : 'Saving the winners';
      setStage(`${label} · ${e.done} / ${e.total}`, half + (e.done / Math.max(1, e.total)) * 0.5);
    },
  });
  if (frames.length < 12) {
    throw new Error(`Only ${frames.length} frames came out of that video — a slow 20+ second pass works best.`);
  }
  // the container's lens data rides on every frame as its EXIF, so the solve
  // starts from a focal prior instead of its search
  try {
    const { readVideoMeta } = await import('./src/io/qtmeta.js');
    const vm = await readVideoMeta(file);
    if (vm.focal35) {
      for (const f of frames) f.exif = { f35: vm.focal35, lens: vm.lensModel, video: true };
      log(`[video] lens ${vm.lensModel || '?'} · ${vm.focal35}mm equiv — focal prior for the solve`);
    }
  } catch (e) { log(`[video] lens metadata: ${e.message || e}`); }
  log(`video: ${frames.length} sharp frames from ${duration.toFixed(0)}s`);
  return { files: frames, jsons: [], video: true };
}

/** One wiring for every training session — fresh runs and continuations
 *  alike report stages, metrics, and completion through the same handlers. */
function wireSession(session) {
  session.on('log', (m) => log(m));
  session.on('stage', (e) => {
    if (e.stage === 'train') return;   // the metrics line owns the bar then
    const label = { decode: 'Decoding photos', solve: 'Solving cameras', seed: 'Seeding gaussians' }[e.stage] || e.stage;
    setStage(e.total > 1 ? `${label} · ${e.done} / ${e.total}` : label,
      e.total ? e.done / e.total : null);
  });
  session.on('metrics', (m) => {
    // LOD training: hold the model at each detail level for a polish
    // window, snapshot it, then raise the growth limit and move on
    const LP = S.lodPlan;
    if (LP && session.trainer && LP.idx < LP.levels.length - 1) {
      const tr = session.trainer;
      if (LP.holdUntil == null && tr.n >= LP.levels[LP.idx]) {
        LP.holdUntil = m.iter + LP.polish;
      } else if (typeof LP.holdUntil === 'number' && m.iter >= LP.holdUntil) {
        LP.holdUntil = 'snapping';
        const lvl = LP.levels[LP.idx];
        session.exportPlyBlob().then((blob) => {
          LP.snaps.push({ n: lvl, blob });
          LP.idx++;
          LP.holdUntil = null;
          tr.growLimit = LP.levels[LP.idx];
          log(`LOD level ${fmt(lvl)} snapshotted — growing on`);
        }).catch(() => { LP.holdUntil = null; });
      }
    }
    setStage(`Training · iteration ${fmt(m.iter)} / ${fmt(S.maxIters)}`, m.iter / S.maxIters);
      const bits = [`${fmt(m.splats)} splats`, `${fmt(m.itersPerSec)} it/s`];
      if (m.psnrTrain > 0) bits.push(`train PSNR ${m.psnrTrain.toFixed(2)}`);
      if (m.psnrHold > 0) bits.push(`holdout ${m.psnrHold.toFixed(2)}`);
      ui.metrics.textContent = bits.join(' · ');
      // the details sheet's raw material + the live curve (the first few
      // hundred cycles rocket from near-nothing to the real curve, so the
      // chart skips the ramp to keep the run's y-scale readable)
      (S.perfMetrics ??= []).push([Math.round(performance.now()), m.iter, m.itersPerSec,
        m.psnrTrain != null ? +m.psnrTrain.toFixed(2) : '', m.psnrHold != null ? +m.psnrHold.toFixed(2) : '']);
      if (m.psnrTrain != null) S.psnrTrain = m.psnrTrain;
      if (m.psnrHold != null) S.psnrHold = m.psnrHold;
      if (m.psnrHold != null) (S.holdHist ??= []).push([m.iter, m.psnrHold]);
      S.itersPerSec = m.itersPerSec;
      if (chart && m.psnrTrain != null && m.iter >= Math.min(500, S.maxIters * 0.02)) {
        chart.push(m.iter, m.psnrTrain, m.psnrHold ?? null);
        chart.maxIter = S.maxIters;
        chart.events = S.chartEvents.map((e) => ({ ...e, at: e.iter / S.maxIters }));
        chart.draw();
      }
      // re-anchor the countdown on every fresh pace measurement (blended,
      // so a wobbling pace doesn't make the clock stutter)
      if (m.itersPerSec > 0 && S.maxIters) {
        const est = performance.now() + Math.max(0, S.maxIters - m.iter) / m.itersPerSec * 1000;
        S.etaAt = S.etaAt == null ? est : S.etaAt + 0.35 * (est - S.etaAt);
      }
      if (!ui.details.hidden && ui.details.open) renderDetails();
      // the shelf's progress line stays fresh through long runs (a patch
      // every 30 s — storage must never pace the trainer)
      if (S.runId && performance.now() - (S.lastPersist || 0) > 30000) {
        S.lastPersist = performance.now();
        persistRunDetails();
      }
    });
    session.on('event', (e) => {
      if (e.kind === 'train-complete') completeRun(e);
      else if (e.kind === 'device-lost') failRun(new Error('The GPU was reclaimed mid-run (backgrounded tab or driver reset). Keep the tab visible and try again.'));
      else if (e.kind === 'refine') {
        S.chartEvents.push({ iter: e.iter, kind: 'refine', label: 'refine' });
        log(`refine @${fmt(e.iter)}: ${fmt(e.kept)} kept`);
      }
      else if (e.kind === 'soft-prune') {
        S.chartEvents.push({ iter: e.iter, kind: 'soft-prune', label: `−${fmt(e.culled)}` });
        log(`soft-prune @${fmt(e.iter)}: culled ${fmt(e.culled)} → ${fmt(e.n)} splats`);
      }
      else if (e.kind === 'tandem-regrow' || e.kind === 'grow-now') {
        S.chartEvents.push({ iter: e.iter, kind: 'grow', label: `+${fmt(e.grown || 0)}` });
        log(`${e.kind === 'grow-now' ? 'grow-now' : 'tandem regrow'} @${fmt(e.iter)}: +${fmt(e.grown || 0)} → ${fmt(e.n)} splats`);
      }
      else if (e.kind === 'cap-hit') showCapPrompt(e);
      else if (e.kind === 'prune-continue') {
        S.sogBlob = null;
        S.maxIters = e.maxIters;
        chart.maxIter = S.maxIters;
        S.chartEvents.push({ iter: e.iter, kind: 'prune', label: `${fmt(e.from)}→${fmt(e.to)}` });
        log(`prune-and-continue: ${fmt(e.from)} → ${fmt(e.to)} splats` +
          (e.delta == null ? '' : ` (${e.delta >= 0 ? '+' : ''}${e.delta.toFixed(2)} dB)`));
      }
      else if (e.kind === 'overflow' && S._ovWarned !== (S.runId ?? 'live')) {
        S._ovWarned = S.runId ?? 'live';
        log(`WARNING: tile-sort overflow — ${e.tiles} tiles dropped so far; lower the splat budget for this device.`);
      }
    });
}

async function run() {
  if (S.phase === 'prep' || S.phase === 'solve' || S.phase === 'train') return;
  S.phase = 'prep';
  ui.start.disabled = true;
  ui.run.hidden = false;
  ensureChart().resize();
  ui.ply.disabled = true;
  ui.prune.disabled = true;
  ui.simplify.disabled = true;
  ui.sess.disabled = true;
  ui.imgs.disabled = true;
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.pause.textContent = 'Pause';
  setStatus('');
  const st = readSettings();
  S.maxIters = st.iters;
  S.runSettings = { ...st };
  S.runName = runNameFor(S.staged);
  S.chartEvents = []; S.holdHist = []; S.perfMetrics = [];
  S.psnrTrain = null; S.psnrHold = null; S.itersPerSec = 0; S.etaAt = null; S.minutes = 0;
  S.trainT0 = 0;
  ui.details.hidden = false;
  ensureChart().reset();

  try {
    // model intakes skip the pipeline: they open in the viewer, and a
    // checkpoint-backed one can continue training from there
    if (S.staged.kind === 'model') return restoreModelFile(S.staged.file, S.staged.reconFile);
    if (S.staged.kind === 'model-url') return restoreModelUrl(S.staged.url, S.staged.reconUrl);
    const inputs = await gatherInputs(S.staged);
    if (inputs.restoreEntries) return restoreFromEntries(inputs.restoreEntries, S.staged.file.name);
    const { files, capture, video } = inputs;
    if (files.length < 3) throw new Error('Use at least 3 overlapping views; 20–200 are recommended.');
    S.capture = capture || null;
    if (capture) {
      const seeds = measuredSeeds(capture);
      log(`${files.length} images + known cameras (${capture.transforms.frames ? 'transforms.json' : 'cameras.json'})` +
        `${seeds.length ? ` + ${fmt(seeds.length)} measured seeds` : ' (random-volume seed)'}` +
        `${video ? ' (from video)' : ''}`);
    } else {
      log(`${files.length} images${video ? ' (from video)' : ''}`);
    }

    // settings -> session options, mirroring the full app: res caps the
    // input scale, the working buffer scales the supervision grid on top of
    // whatever that yields. Full size trains at the pictures' native long
    // side (probed from the first image — the engine's memory budget still
    // applies); Auto leaves the cap to the engine.
    let trainMaxDim;
    if (st.res === -1) {
      try {
        const probe = files[0] instanceof Blob ? files[0] : files[0] && files[0].source;
        if (probe) {
          const bmp = await createImageBitmap(probe);
          trainMaxDim = Math.max(bmp.width, bmp.height);
          bmp.close();
        }
      } catch { /* fall through to the engine default */ }
      log(trainMaxDim
        ? `full size: training at the pictures' native ${trainMaxDim}px (memory budget still applies)`
        : 'full size requested but the native size was unreadable — the engine picks the cap');
    } else if (st.res) {
      trainMaxDim = st.res;
    }
    const frames = (trainMaxDim || st.feat || st.buf !== 1) ? {
      ...(trainMaxDim ? { trainMaxDim } : {}),
      ...(st.buf !== 1 ? { trainScale: st.buf } : {}),
      ...(st.feat ? { featMaxDim: st.feat } : {}),
    } : undefined;
    const trainerOpts = {};
    if (st.sh != null) trainerOpts.shDeg = st.sh;
    if (st.splats) trainerOpts.maxSplats = st.splats;
    // big budgets: growth must be able to reach the cap even when the sparse
    // cloud can't seed budget/4
    if (st.splats >= 1000000) trainerOpts.capMult = 16;
    // LOD training (opt-in, >= 1M budgets): the model pauses at halving
    // detail levels for a polish-and-snapshot before growing on
    const lodOn = st.lod && st.splats >= 1000000;
    if (lodOn) {
      try {
        S.lodPlan = buildLODPlan(st.splats, st.lodLevels || 0, S.maxIters);
      } catch (e) {
        throw new Error(`LOD snapshots: ${e.message || e}`);
      }
      trainerOpts.capMult = Math.max(16, Math.ceil(st.splats / 250000) + 1);
    } else {
      S.lodPlan = null;
      if (st.lod) log('LOD needs a splat budget of 1,000,000+ — training straight through');
    }
    // MCMC is the default set (measured ahead of the classic schedule on
    // most scene/budget combinations); Off restores the classic schedule
    const mcmcActive = st.mcmc !== '0';
    if (mcmcActive) {
      Object.assign(trainerOpts, {
        growRate: 0.05, mcmcNoise: true, scaleReg: 0.01, moveCap: 0.25,
        ...(st.sh > 0 ? { shLr: 3e-4 } : {}),
      });
    }
    // Soft prune (opt-in): periodic importance culls during growth
    if (st.softprune) trainerOpts.softPrune = { ratio: parseFloat(st.softprune), every: 3000, until: 0.85 };
    // Auto splat budget: the measured capacity law (~15 splats/cycle
    // classic, ~35 under MCMC), clamped to the device class
    const phone = phoneClass();
    if (!st.splats) {
      // mobile ceiling is 1M like the full app (past it the SH binding at
      // deg 3 and the tile-sort entries bind instead — overflow warns)
      trainerOpts.maxSplats = Math.max(150000, Math.min(phone ? 1000000 : 2000000,
        Math.round(S.maxIters * (mcmcActive ? 35 : 15))));
      trainerOpts.capMult = 8; // growth must reach the computed cap from a lean seed
    }
    log(`config: ${trainMaxDim ? `${trainMaxDim}px` : 'auto res'} · sh ${st.sh} · ` +
      `${fmt(trainerOpts.maxSplats)} splats · ${mcmcActive ? 'mcmc' : 'classic'} · ` +
      `gpu ${st.gpu}%${lodOn ? ` · lod ${S.lodPlan.levels.map((l) => fmt(l)).join('→')}` : ''}`);
    const session = createSession({
      maxIters: S.maxIters,
      holdout: -1,
      evalHoldEvery: 2500,
      // the GPU-usage slider as a duty cycle: full batches with idle
      // between them (read live, so dragging mid-run needs no restart)
      gpuUsage: st.gpu / 100,
      ...(mcmcActive ? { refineEvery: 500 } : {}),
      initTarget: lodOn ? S.lodPlan.levels[0]
        : (st.splats ? Math.round(st.splats / 4) : (phone ? 60000 : undefined)),
      ...(phone ? { lowMem: true } : {}),
      sfm: {
        ...solveTierOpts(st.solve),
        ...(phone ? { workers: 3, uiYield: true } : {}),
        // a video is a chain: consecutive frames overlap by construction, so
        // pairs gate on absolute inlier counts, not the ratio photo sets use
        ...(video ? { pairMinInliers: 100, pairMinInliersAdj: 15 } : {}),
      },
      frames: phone ? { featMaxDim: 960, ...(frames || {}) } : frames,
      trainer: Object.keys(trainerOpts).length ? trainerOpts : undefined,
      trainWorker: trainWorkerOpt(),
    });
    S.session = session;
    wireSession(session);

    setStage('Decoding photos …', 0);
    await session.load(files);
    setPhotoSources(files);
    if (capture) {
      S.phase = 'solve';
      setStage('Using the known cameras (solve skipped) …', 0);
      const points = measuredSeeds(capture);
      const recon = session.useKnownCameras(parseCameras(capture.transforms),
        { ...(points.length ? { points } : {}), seedCount: phone ? 8000 : 20000 });
      if (recon.cams.length !== files.length) {
        throw new Error('Some images lost their known cameras during decoding. No estimated poses will be substituted.');
      }
      log(`${recon.cams.length} known camera poses loaded — camera solve skipped. ${points.length ? 'Measured seeds.' : 'Random-volume initialization.'}`);
      const lv = levelingNote(capture.metadata && capture.metadata.leveling);
      if (lv) log(`leveling: ${lv}`);
    } else {
      S.phase = 'solve';
      setStage(`Solving cameras (${ui.tier.value}) …`, 0);
      await session.solve();
      const cams = session.recon?.cams?.length || 0;
      log(`solve: ${cams} of ${files.length} frames registered`);
      if (cams < 2) throw new Error('The solve registered fewer than 2 frames — try more overlap, or the Precise tier.');
    }
    S.phase = 'seed';
    setStage('Seeding gaussians …', 0);
    await session.seed();
    // LOD runs start capped at the first level — the metrics handler raises
    // the limit as each level snapshots
    if (S.lodPlan) session.trainer.growLimit = S.lodPlan.levels[0];
    setupViewer(session);
    S.phase = 'train';
    ui.pause.disabled = false;
    ui.finish.disabled = false;
    ui.prune.disabled = false;
    S.trainT0 = performance.now();
    session.start();
    S.loadedFiles = files.map((f) => ({ name: f.name, url: null }));
    saveRunStart(files[0]);
    setStatus('Training — you can orbit the preview while it runs.');
  } catch (e) {
    failRun(e);
  }
}

// ── score: the live curve + the details sheet ─────────────────────────────────
let chart = null;
function ensureChart() {
  if (!chart) {
    chart = new Chart(ui.chartCv);
    chart.scale = S.scoreScale;
  }
  chart.maxIter = S.maxIters;
  return chart;
}

/** Training progress as the visitor reads it: a percentage of the horizon. */
function pctOf(iter) {
  if (!S.maxIters) return '0%';
  return `${Math.min(100, Math.floor((iter || 0) / S.maxIters * 100))}%`;
}

/** Time remaining as a live 04:32-style countdown. */
function etaText() {
  if (S.etaAt == null) return '—:—';
  const s = Math.max(0, (S.etaAt - performance.now()) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return `${h ? `${h}:` : ''}${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function pctl(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

/** A score in the visitor's chosen unit (dB, or the calibrated % estimate). */
function fmtScore(psnr) {
  if (psnr == null || Number.isNaN(psnr)) return '—';
  return S.scoreScale === 'similarity'
    ? `${psnrToApproxSsimPercent(psnr).toFixed(1)} %`
    : `${psnr.toFixed(2)} dB`;
}

function minutesLive() {
  const carry = S.minutes || 0;
  if (!S.trainT0) return carry;
  return Math.max(carry, Math.round((performance.now() - S.trainT0) / 60000));
}

function renderDetails() {
  for (const [k, el] of Object.entries(ui.tabs)) el.classList.toggle('active', k === S.detailTab);
  const ses = S.session;
  if (!ses) { ui.detailBody.textContent = 'No session yet.'; return; }
  if (S.detailTab === 'perf') return renderPerfTab();
  if (S.detailTab === 'cams') return renderCamsTab();
  if (S.detailTab === 'gpu') return renderGpuTab();
  // score: the headline numbers behind the curve
  const it = ses.trainer ? ses.trainer.iter : 0;
  const rows = [
    ['Iteration', `${fmt(it)} / ${fmt(S.maxIters)} · ${pctOf(it)}`],
    ['Splats', fmt(ses.trainer ? ses.trainer.n : 0)],
    ['Pace', `${fmt(S.itersPerSec)} it/s`],
    ['Remaining', S.phase === 'train' ? etaText() : '—'],
    ['Train score', fmtScore(S.psnrTrain)],
    ['Holdout score', fmtScore(S.psnrHold)],
    ['Wall time', `${minutesLive()} min`],
  ];
  ui.detailBody.replaceChildren();
  const dl = document.createElement('dl');
  dl.className = 'stats';
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    dl.append(dt, dd);
  }
  ui.detailBody.append(dl);
}

function buildPerfReport() {
  const ses = S.session;
  if (!ses) return 'no session';
  const gi = (ses.gpu && ses.gpu.info) || {};
  const rows = (ses.perf && ses.perf.frames) || [];
  const marks = (ses.perf && ses.perf.marks) || [];
  const fr0 = (ses.frames && ses.frames[0]) || {};
  const L = [];
  L.push(`splat.js perf log — ${new Date().toISOString()}`);
  L.push(`url: ${location.href}`);
  L.push(`ua: ${navigator.userAgent}`);
  L.push(`gpu: ${[gi.vendor, gi.architecture, gi.device].filter(Boolean).join(' ') || 'unknown'}`);
  L.push(`screen: ${screen.width}x${screen.height} @dpr ${devicePixelRatio}`);
  L.push(`photos: ${(ses.frames && ses.frames.length) || 0} · placed ${ses.recon ? ses.recon.cams.length : '?'} · training res ${fr0.tw || '?'}x${fr0.th || '?'}`);
  L.push(`settings: ${JSON.stringify(S.runSettings || {})} · run ${S.runName || '?'}`);
  L.push(`splats: ${fmt(ses.trainer ? ses.trainer.n : 0)} · holdout psnr ${S.psnrHold != null ? S.psnrHold.toFixed(2) : '—'} dB`);
  if (ses.trainer) L.push(`tileGrad: ${ses.trainer.tileGrad} · maxIters ${S.maxIters}`);
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

function renderPerfTab() {
  ui.detailBody.replaceChildren();
  const row = document.createElement('div');
  row.className = 'row';
  const copy = document.createElement('button');
  copy.className = 'btn'; copy.textContent = 'Copy';
  copy.onclick = () => {
    navigator.clipboard.writeText(buildPerfReport()).then(
      () => { copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy'; }, 1500); },
      () => { copy.textContent = 'Copy failed'; });
  };
  const dl = document.createElement('button');
  dl.className = 'btn'; dl.textContent = 'Download';
  dl.onclick = () => {
    const blob = new Blob([buildPerfReport()], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `splat-perf-${Date.now()}.log`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
  };
  row.append(copy, dl);
  const pre = document.createElement('pre');
  pre.className = 'perflog';
  pre.textContent = buildPerfReport();
  ui.detailBody.append(row, pre);
}

function renderCamsTab() {
  const ses = S.session;
  const cams = (ses.recon && ses.recon.cams) || [];
  const fr0 = (ses.frames && ses.frames[0]) || {};
  ui.detailBody.replaceChildren();
  const p = document.createElement('p');
  p.className = 'hint';
  p.textContent = `${(ses.frames || []).length} photos · ${cams.length} placed` +
    `${ses.recon && ses.recon.known ? ' (known poses)' : ''} · training res ${fr0.tw || '?'}x${fr0.th || '?'}`;
  ui.detailBody.append(p);
  if (!cams.length) return;
  const pre = document.createElement('pre');
  pre.className = 'perflog';
  const cap = 500;
  const lines = cams.slice(0, cap).map((c, i) => {
    const fr = (ses.frames || [])[c.imgIdx] || {};
    return `${String(i).padStart(3)} ${(fr.name || `#${c.imgIdx}`).padEnd(24)} f ${(c.f || 0).toFixed(1)}`;
  });
  if (cams.length > cap) lines.push(`…and ${cams.length - cap} more`);
  pre.textContent = lines.join('\n');
  ui.detailBody.append(pre);
}

async function ensureGpuFacts() {
  if (S.gpuFacts) return S.gpuFacts;
  const f = { screen: `${screen.width}x${screen.height} @dpr ${devicePixelRatio}`, ua: navigator.userAgent };
  try {
    const ad = await navigator.gpu.requestAdapter();
    const info = ad ? await ad.requestAdapterInfo() : null;
    if (info) {
      f.vendor = info.vendor; f.architecture = info.architecture;
      f.device = info.device; f.description = info.description;
    }
  } catch { /* stay unknown */ }
  f.trainWorker = S.session ? String(S.session.opts.trainWorker ?? 'auto') : '—';
  return (S.gpuFacts = f);
}

function renderGpuTab() {
  ui.detailBody.textContent = 'Probing the adapter …';
  ensureGpuFacts().then((f) => {
    if (S.detailTab !== 'gpu') return;
    const rows = [
      ['GPU', [f.vendor, f.architecture, f.device].filter(Boolean).join(' ') || 'unknown'],
      ['Description', f.description || '—'],
      ['Screen', f.screen],
      ['Training core', f.trainWorker],
      ['User agent', f.ua],
    ];
    ui.detailBody.replaceChildren();
    const dl = document.createElement('dl');
    dl.className = 'stats';
    for (const [k, v] of rows) {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v;
      dl.append(dt, dd);
    }
    ui.detailBody.append(dl);
  });
}

// ── library: runs + last capture survive the tab ─────────────────────────────
// Every run gets a record the moment training starts; finished runs keep
// their model (.ply) + recon + curve so closing the tab never discards an
// hour of GPU work. All of it stays in this origin's IndexedDB, like the
// full app's Yours shelf — the record shape matches store.js on purpose.

/** The tile name a staged intake trains under. */
function runNameFor(staged) {
  if (!staged) return 'run';
  if (staged.kind === 'photos') return `${staged.images.length} photos`;
  if (staged.file) return staged.file.name;
  if (staged.kind === 'model-url') {
    return String(staged.url || 'model').split('/').pop().split('?')[0] || 'model';
  }
  return staged.kind || 'run';
}

/** The tile's one-line read-out: status, horizon, size, score, cost. */
function describeRun(rec, live = false) {
  const status = rec.status === 'finished' ? 'finished'
    : live ? 'training' : (rec.status === 'training' ? 'interrupted' : (rec.status || 'run'));
  const bits = [status];
  if (rec.iter != null || rec.maxIters != null) bits.push(`${fmt(rec.iter || 0)} / ${fmt(rec.maxIters || 0)}`);
  if (rec.splats != null) bits.push(`${fmt(rec.splats)} splats`);
  if (rec.psnr != null && !Number.isNaN(+rec.psnr)) bits.push(`${(+rec.psnr).toFixed(2)} dB`);
  if (rec.minutes != null) bits.push(`${rec.minutes} min`);
  return bits.join(' · ');
}

/** Thin a history to ~max rows, keeping the endpoints. */
function compactRows(rows, max = 600) {
  if (!Array.isArray(rows) || !rows.length) return [];
  const stride = Math.max(1, Math.ceil(rows.length / max));
  const out = [];
  for (let i = 0; i < rows.length; i += stride) out.push(Array.from(rows[i] || []));
  const last = rows[rows.length - 1];
  if (out.length && rows.length > 1 && (out.length - 1) * stride < rows.length - 1) {
    out.push(Array.from(last || []));
  }
  return out;
}

/** The Details sheet's memory: curve, events, timings, settings, scores. */
function captureRunProgressDetails() {
  const ses = S.session;
  const perf = (ses && ses.perf) || {};
  const restoredStats = (S.restored && S.restored.reconJson && S.restored.reconJson.stats) || {};
  const history = ses && Array.isArray(ses.lossHistory) && ses.lossHistory.length
    ? ses.lossHistory : (restoredStats.chart || []);
  const holds = S.holdHist.length ? S.holdHist : (restoredStats.holds || []);
  return {
    chart: compactRows(history, 800),
    holds: compactRows(holds, 800),
    events: S.chartEvents.slice(-300).map((e) => ({ ...e })),
    perf: {
      frames: (perf.frames || []).slice(-6000).map((r) => Array.from(r || [])),
      marks: (perf.marks || []).slice(-1200).map((m) => ({ ...m })),
    },
    metrics: (S.perfMetrics || []).slice(-2000).map((m) => Array.from(m || [])),
    settings: { ...(S.runSettings || {}) },
    tileGrad: (ses && ses.trainer && ses.trainer.tileGrad) ?? null,
    maxIters: S.maxIters || 0,
    iter: (ses && ses.trainer && ses.trainer.iter) || 0,
    minutes: minutesLive(),
    psnrTrain: S.psnrTrain ?? null,
    psnrHold: S.psnrHold ?? null,
  };
}

/** Hand a saved run's memory back to the live sheet (perf times re-anchored
 *  to now, so a reopened log reads as one session, like the full app). */
function restoreRunDetails(run) {
  if (!run) return;
  const progress = run.detailProgress || {};
  S.chartEvents = Array.isArray(progress.events) ? progress.events.map((e) => ({ ...e })) : [];
  S.holdHist = Array.isArray(progress.holds) ? progress.holds.map((r) => Array.from(r || [])) : [];
  S.perfMetrics = Array.isArray(progress.metrics) ? progress.metrics.map((m) => Array.from(m || [])) : [];
  const lastMetric = S.perfMetrics[S.perfMetrics.length - 1];
  if (lastMetric && Number.isFinite(+lastMetric[2])) S.itersPerSec = +lastMetric[2];
  const ses = S.session;
  if (ses) {
    const history = Array.isArray(progress.chart) && progress.chart.length ? progress.chart : [];
    ses.lossHistory = history.map((r) => Array.from(r || []));
    ses.perf = ses.perf || { frames: [], marks: [] };
    const frameRows = ((progress.perf && progress.perf.frames) || []).map((r) => Array.from(r || []));
    const markRows = ((progress.perf && progress.perf.marks) || []).map((m) => ({ ...m }));
    const lastT = frameRows.length ? frameRows[frameRows.length - 1][0] : 0;
    const timeShift = frameRows.length ? performance.now() + 1 - lastT : 0;
    ses.perf.frames = frameRows.map((r) => { if (r.length) r[0] += timeShift; return r; });
    ses.perf.marks = markRows.map((m) => ({ ...m, ...(Number.isFinite(m.t) ? { t: m.t + timeShift } : {}) }));
    S.perfMetrics = S.perfMetrics.map((m) => { if (m.length) m[0] += timeShift; return m; });
  }
  if (progress.minutes != null) S.minutes = Math.max(S.minutes || 0, progress.minutes || 0);
  if (progress.psnrTrain != null) S.psnrTrain = progress.psnrTrain;
  if (progress.psnrHold != null) S.psnrHold = progress.psnrHold;
  S.runSettings = progress.settings || null;
  S.gpuFacts = run.detailGpu || null;
}

/** A crash-safe progress patch: iteration, size, score, and the sheet's
 *  memory. Best-effort — storage must never disturb a run. */
async function persistRunDetails() {
  if (!S.runId || !S.session) return;
  try {
    await patchRun(S.runId, {
      iter: (S.session.trainer && S.session.trainer.iter) || 0,
      splats: (S.session.trainer && S.session.trainer.n) || 0,
      psnr: S.psnrHold ?? S.psnrTrain ?? null,
      minutes: minutesLive(),
      detailProgress: captureRunProgressDetails(),
      detailGpu: S.gpuFacts || await ensureGpuFacts().catch(() => null),
    });
  } catch { /* the run matters, the record doesn't */ }
}

async function thumbFromImage(blob) {
  const bmp = await createImageBitmap(blob, { resizeWidth: 320 });
  const cv = document.createElement('canvas');
  cv.width = bmp.width; cv.height = bmp.height;
  cv.getContext('2d').drawImage(bmp, 0, 0);
  bmp.close();
  return new Promise((res) => cv.toBlob(res, 'image/webp', 0.8));
}

async function thumbFromViewer() {
  return new Promise((res) => ui.view.toBlob(res, 'image/webp', 0.8));
}

/** The full-app state shape buildReconJson reads, trimmed to what the
 *  browser tracks (no scene cloud — the recon keeps its own points). */
function reconFacade() {
  return {
    session: S.session,
    scene: null,
    preset: { id: '__own', name: S.runName || 'splat', capture: S.capture },
    _localRun: null,
    loadedFiles: S.loadedFiles || [],
    minutes: S.minutes || 0,
    psnrTrain: S.psnrTrain ?? null, psnrHold: S.psnrHold ?? null, psnrTest: null,
    holdHist: S.holdHist || [],
  };
}

/** The record is saved before the optional thumbnail; Back/close must not
 *  race image decoding and lose the run's first snapshot. A continued
 *  library run keeps its record (status flips back to training). */
async function saveRunStart(firstFile) {
  const ses = S.session;
  const rec = {
    id: S.runId || `run-${Date.now()}`, name: S.runName || 'run',
    status: 'training', createdAt: Date.now(), updatedAt: Date.now(),
    iter: (ses && ses.trainer && ses.trainer.iter) || 0, maxIters: S.maxIters,
    splats: (ses && ses.trainer && ses.trainer.n) || 0,
    psnr: S.psnrHold ?? S.psnrTrain ?? null,
    frames: (ses && ses.frames && ses.frames.length) || 0,
    detailProgress: captureRunProgressDetails(),
    detailGpu: await ensureGpuFacts().catch(() => null),
  };
  try {
    if (S.runId) {
      const { createdAt: _drop, ...patch } = rec;
      if (!await patchRun(rec.id, patch)) await saveRun(rec);
    } else {
      await saveRun(rec);
    }
  } catch { /* best-effort */ }
  S.runId = rec.id;
  renderLibrary().catch(() => {});
  if (firstFile) {
    try {
      const thumb = await thumbFromImage(firstFile).catch(() => null);
      if (thumb) {
        await patchRun(rec.id, { thumb });
        renderLibrary().catch(() => {});
      }
    } catch { /* the tile just goes textless */ }
  }
}

/** The finished run, secured: model + recon + curve + viewer thumbnail, so
 *  the shelf's tile reopens exactly this result. Async — the done screen
 *  is already live and must never wait on storage. */
async function secureFinishedRun() {
  if (!S.runId || !S.session) return;
  const ses = S.session, id = S.runId;
  try {
    setStatus('Saving the finished run on this device …');
    const [ply, thumb] = await Promise.all([
      ses.exportPlyBlob().catch(() => null),
      thumbFromViewer().catch(() => null),
    ]);
    const recon = buildReconJson(reconFacade());
    const patch = {
      status: 'finished',
      iter: (ses.trainer && ses.trainer.iter) || 0,
      splats: (ses.trainer && ses.trainer.n) || 0,
      psnr: S.psnrHold ?? S.psnrTrain ?? null,
      minutes: minutesLive(),
      ...(ply ? { ply } : {}), recon,
      detailProgress: captureRunProgressDetails(),
      detailGpu: S.gpuFacts || null,
      ...(thumb ? { thumb } : {}),
    };
    if (!await patchRun(id, patch)) {
      await saveRun({ id, name: S.runName || 'run', createdAt: Date.now(), updatedAt: Date.now(),
        maxIters: S.maxIters, frames: (ses.frames || []).length, ...patch });
    }
    setStatus('Training finished — saved on this device under Yours. Download the .ply, or keep orbiting.');
  } catch (e) {
    console.warn('local save failed', e);
  }
  renderLibrary().catch(() => {});
}

async function renderLibrary() {
  const runs = await listRuns().catch(() => []);
  ui.library.hidden = !runs.length;
  ui.runs.replaceChildren();
  for (const rec of runs) {
    const live = rec.id === S.runId && (S.phase === 'train' || S.phase === 'paused');
    const tile = document.createElement('div');
    tile.className = 'run';
    if (rec.thumb) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(rec.thumb);
      img.alt = '';
      tile.append(img);
    }
    const meta = document.createElement('div');
    meta.className = 'runmeta';
    const b = document.createElement('b');
    b.textContent = rec.name || 'run';
    const span = document.createElement('span');
    span.textContent = describeRun(rec, live);
    meta.append(b, span);
    tile.append(meta);
    if (rec.ply) {
      const open = document.createElement('button');
      open.className = 'btn'; open.textContent = 'Open';
      open.addEventListener('click', () => openLibraryRun(rec.id));
      tile.append(open);
    }
    const del = document.createElement('button');
    del.className = 'btn danger'; del.textContent = 'Delete';
    del.addEventListener('click', async () => {
      await deleteRun(rec.id).catch(() => {});
      renderLibrary().catch(() => {});
    });
    tile.append(del);
    ui.runs.append(tile);
  }
}

/** A saved model reopens as a finished run: viewer, curve, refine markers,
 *  and the Details sheet's memory — ready to continue with its photos. */
async function openLibraryRun(id) {
  const runs = await listRuns().catch(() => []);
  const rec = runs.find((r) => r.id === id);
  if (!rec || !rec.ply) return;
  resetRun();
  S.runId = rec.id;
  ui.run.hidden = false;
  ui.details.hidden = false;
  ensureChart().resize();
  setStatus('Opening the saved run …');
  try {
    const gaussians = parsePlyGaussians(new Uint8Array(await rec.ply.arrayBuffer()));
    await presentRestored({ gaussians, reconJson: rec.recon || null, state: null }, rec.name || 'saved run');
    restoreRunDetails(rec);
    const c = ensureChart();
    c.events = S.chartEvents.map((e) => ({ ...e, at: e.iter / S.maxIters }));
    c.draw();
    ui.start.disabled = true;   // nothing staged — a fresh input re-arms it
    ui.start.textContent = 'Start training →';
    if (!ui.details.hidden) renderDetails();
  } catch (e) {
    console.error(e);
    setStatus(`Could not open the saved run: ${e.message}`, 'error');
  }
}

// ── last capture: the walked set survives a refresh ─────────────────────────
// Staged photos/video/zips are written into IndexedDB (origin-local, nothing
// leaves the machine) and offered back as a tile — a refresh used to erase
// a capture that might have taken minutes to walk.

function rememberCapture(kind, files) {
  saveLastCapture({ kind, created: Date.now(), files: [...files].map((f) => ({ name: f.name, blob: f })) })
    .then(() => refreshLastCapture(), () => {});
}

async function refreshLastCapture() {
  const rec = await loadLastCapture().catch(() => null);
  if (!rec || !rec.files || !rec.files.length) { ui.lastcap.hidden = true; return; }
  const n = rec.files.length;
  const what = rec.kind === 'video' ? 'video'
    : rec.kind === 'zip' ? 'capture .zip' : `${n} photo${n > 1 ? 's' : ''}`;
  ui.lastcapText.textContent = `Last capture · ${what} · ${new Date(rec.created).toLocaleDateString()}`;
  ui.lastcap.hidden = false;
}

// ── restore: open a finished model (session .zip, .ply, .sog) ────────────────
// A phone session .zip (model.sog + recon.json + state.bin) lands in the
// viewer exactly as it trained; with its checkpoint it can also continue.
// Unlike the full app's reader this path inflates on the way in, so a
// user-recompressed (deflated) session zip restores too.

/** Zip entries -> { gaussians, reconJson|null, state|null }. The checkpoint
 *  wins when present (bit-exact); else the SOG decodes; else a packed PLY. */
async function modelFromEntries(entries) {
  const byBase = new Map();
  for (const [name, bytes] of entries) {
    const b = baseName(name).toLowerCase();
    if (!byBase.has(b)) byBase.set(b, bytes);
  }
  let reconJson = null;
  if (byBase.has('recon.json')) reconJson = JSON.parse(new TextDecoder().decode(byBase.get('recon.json')));
  let gaussians = null, state = null;
  if (byBase.has('state.bin')) {
    const st = parseState(byBase.get('state.bin'));
    gaussians = st.gaussians;
    state = { iter: st.iter, engine: st.engine, recipe: st.recipe };
  } else if (byBase.has('model.sog')) {
    gaussians = await sogToGaussians(byBase.get('model.sog'));
  } else if (byBase.has('model.ply')) {
    gaussians = parsePlyGaussians(byBase.get('model.ply'));
  }
  if (!gaussians) throw new Error('nothing loadable in the model file');
  return { gaussians, reconJson, state };
}

/** A shared recon travels as bare JSON or a one-entry STORED zip — both. */
async function fetchReconJson(reconUrl) {
  const r = await fetch(reconUrl);
  if (!r.ok) throw new Error(`recon fetch failed (${r.status})`);
  const rb = new Uint8Array(await r.arrayBuffer());
  if (rb[0] === 0x50 && rb[1] === 0x4b) {
    const entries = await readZip(rb);
    const hit = [...entries.entries()].find(([name]) => baseName(name).toLowerCase() === 'recon.json');
    const entry = hit ? hit[1] : [...entries.values()][0];
    if (!entry) throw new Error('empty recon archive');
    return JSON.parse(new TextDecoder().decode(entry));
  }
  return JSON.parse(new TextDecoder().decode(rb));
}

async function restoreModelFile(file, reconFile) {
  setStage(`Loading ${file.name} …`, 0);
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (reconFile) {
    let reconBytes;
    try {
      reconBytes = new TextEncoder().encode(await reconFile.text());
      JSON.parse(new TextDecoder().decode(reconBytes));
    } catch {
      throw new Error(`${reconFile.name} is not valid JSON`);
    }
    const ext = /\.sog$/i.test(file.name) ? 'model.sog' : 'model.ply';
    return presentRestored(
      await modelFromEntries(new Map([[ext, bytes], ['recon.json', reconBytes]])), file.name);
  }
  if (bytes[0] === 0x50 && bytes[1] === 0x4b) return restoreFromEntries(await readZip(bytes), file.name);
  return presentRestored(await decodeModel(bytes, null), file.name);
}

async function restoreModelUrl(url, reconUrl) {
  setStage('Fetching the model …', 0);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model fetch failed (${r.status})`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const got = (bytes[0] === 0x50 && bytes[1] === 0x4b)
    ? await modelFromEntries(await readZip(bytes))
    : await decodeModel(bytes, null);
  if (!got.reconJson && reconUrl) got.reconJson = await fetchReconJson(reconUrl);
  const label = url.split('/').pop().split('?')[0] || 'model';
  return presentRestored(got, label);
}

async function restoreFromEntries(entries, label) {
  setStage(`Loading ${label} …`, 0);
  return presentRestored(await modelFromEntries(entries), label);
}

/** The restored model lands in the done-state viewer: orbit, tour,
 *  exports — and Continue, when cameras + frames came along. */
async function presentRestored({ gaussians, reconJson, state }, label) {
  const iter = (state && state.iter) || (reconJson && reconJson.iter) || 0;
  log(`${label}: ${fmt(gaussians.n)} splats` +
    `${reconJson ? ` · ${(reconJson.cams || []).length} cameras` : ''}` +
    `${state ? ` · checkpoint @${fmt(state.iter)}` : ' · exported model'}`);
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
  await ses.seedFrom(gaussians, {
    viewOnly: true,
    sceneRadius: reconJson ? reconJson.sceneRadius : undefined,
    iter,
    trainer: { maxSplats: gaussians.n, capMult: 1 },
  });
  S.session = ses;
  S.restored = { gaussians, reconJson, state, label };
  S.capture = reconJson && reconJson.knownCameras
    ? { transforms: true, metadata: {}, levelingUp: reconJson.levelingUp || null } : null;
  S.phase = 'done';
  S.maxIters = Math.max(1, iter);
  S.runName = label;
  // the settings panel opens on the run's own SH degree (reselectable
  // before Continue) — the default the full app's config card shows
  if (reconJson && gaussians.shK != null && DEG_OF_SHK[gaussians.shK] != null) {
    ui.sh.value = String(DEG_OF_SHK[gaussians.shK]);
    syncQuality();
  }
  setupViewer(ses);
  const stats = (reconJson && reconJson.stats) || {};
  // the saved curve replays on the chart — the finished run reads as trained
  const c = ensureChart();
  c.reset();
  c.train = (stats.chart || []).map(([i, v]) => [i, v]);
  c.hold = (stats.holds || []).map(([i, v]) => [i, v]);
  c.draw();
  const bits = [`${fmt(gaussians.n)} splats`];
  if (iter) bits.push(`${fmt(iter)} cycles`);
  if (stats.psnrTrain) bits.push(`train ${stats.psnrTrain.toFixed(2)} dB`);
  if (stats.psnrHold) bits.push(`holdout ${stats.psnrHold.toFixed(2)} dB`);
  setStage(`Restored · ${label}`, 1);
  ui.metrics.textContent = bits.join(' · ');
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.ply.disabled = false;
  ui.prune.disabled = true;   // same — no trainer, nothing to prune
  ui.simplify.disabled = true;   // needs the live trainer — continue first
  ui.sess.disabled = true;   // no live trainer state to pack — continue first
  ui.imgs.disabled = true;   // restores carry no photo pixels
  ui.cont.hidden = !reconJson;
  setStatus(!reconJson
    ? 'Bare model, no cameras — orbit it, or download the .ply.'
    : state
      ? 'Bit-exact checkpoint — Continue training resumes it exactly.'
      : 'Exported model — Continue training replays it (opacities unbaked, needs the source photos).');
  ui.start.disabled = false;
  ui.start.textContent = 'Train new ↻';
}

// ── continue: train a restored model further ───────────────────────────────
// The photos come from the recon's URLs when it has them (preset runs);
// own-capture exports carry names only, so the visitor drops the matching
// source photos (the phone's capture .zip or photos .zip holds them).

async function continueClick() {
  const R = S.restored;
  if (!R || !R.reconJson || S.phase !== 'done') return;
  const recon = R.reconJson;
  const names = (recon.frames || []).map((f) => f.name);
  const urls = (recon.source && recon.source.urls) || [];
  ui.cont.disabled = true;
  try {
    if (urls.length === names.length && names.length && urls.every(Boolean)) {
      setStatus('Fetching the photos …');
      const files = [];
      for (let i = 0; i < urls.length; i++) {
        setStage(`Fetching photo ${i + 1} / ${urls.length}`, i / urls.length);
        const r = await fetch(urls[i]);
        if (!r.ok) throw new Error(`photo fetch failed (${r.status})`);
        files.push(new File([await r.blob()], names[i], { type: 'image/jpeg' }));
      }
      return continueRun(files, urls);
    }
    ui.needphotos.hidden = false;
    ui.photosHint.textContent = `This export carries no photo URLs — drop the ${names.length} source ` +
      `photos to continue (names must match${names.length ? `, e.g. ${names[0]}` : ''}).`;
    setStatus('Drop the source photos below to continue training.');
  } catch (e) {
    console.error(e);
    setStatus(e.message || String(e), 'error');
  } finally {
    ui.cont.disabled = false;
  }
}

/** Match dropped photos BY NAME to the run's own frame order — re-sorting
 *  could reorder against recon.cams imgIdx. Returns { files } or { missing }. */
function matchPhotosByName(files, names) {
  const byName = new Map([...files].map((f) => [f.name, f]));
  const missing = names.filter((n) => !byName.has(n));
  if (missing.length) return { missing };
  return { files: names.map((n) => byName.get(n)) };
}

async function continueRun(files, urls = null) {
  const R = S.restored;
  const recon = R.reconJson;
  const st = readSettings();
  S.runSettings = { ...st };
  S.runName = `${R.label || 'model'} (continued)`;
  ui.run.hidden = false;
  ui.details.hidden = false;
  ensureChart().resize();
  const g = R.gaussians;
  const hasState = !!R.state;
  // the trainer recipe the run was made with (state header, else the recon
  // JSON); null = a record from before recipes = the legacy defaults
  const recipe = (R.state && R.state.recipe) || recon.recipe || null;
  const baseIter = (R.state && R.state.iter) ?? recon.iter ?? 0;
  const engine = (R.state && R.state.engine) || 'v1';
  // a finished run continues PAST its old horizon — the automatic
  // half-horizon extra (10k minimum)
  const extra = Math.max(10000, Math.round(baseIter / 2));
  S.maxIters = baseIter + extra;
  S.lodPlan = null;
  S.phase = 'prep';
  ui.start.disabled = true;
  ui.cont.hidden = true;
  ui.needphotos.hidden = true;
  ui.ply.disabled = true;
  ui.sess.disabled = true;
  ui.imgs.disabled = true;
  setStatus('');
  try {
    if (S.session) { try { S.session.dispose(); } catch { /* view-only facade */ } }
    // the continuation replays the run's own input scale AND buffer: the
    // stored size is the COMBINED size (input × buffer), so the buffer is
    // divided back out before the value passes through the multiplier again
    const fr0 = recon.frames[0] || {};
    const prevDim = Math.max(fr0.tw || 0, fr0.th || 0) || undefined;
    const prevBuf = continuedBuf(recon.trainScale, undefined, prevDim);
    const trainMaxDim = continuedInputDim(prevDim, prevBuf);
    const bufScale = prevBuf !== 1 ? prevBuf : undefined;
    // the run's own budget carries over (the recon records it now); the
    // model size floors it, an explicit panel choice raises it
    const cap = Math.max(g.n, recon.maxSplats || 0, st.splats || 0);
    // rebuild the SAME trainer the run was made with: the recipe carries
    // the optimizer set and the refine cadence; sizing is per run
    const { refineEvery: recipeRefine, ...recipeTrainer } = recipe || {};
    if ((bufScale ?? 1) === 2 && phoneClass()) {
      log('Working buffer 2× roughly quadruples GPU working memory — on this phone the run may abort for lack of memory. 1× trains the same photos safely.');
    }
    // a reselected SH degree resamples the decoded harmonics (low bands kept,
    // growth zero-padded) so the continuation trains the chosen degree —
    // same as the full app; unsettled, the run's own degree replays
    let contShDeg = null;
    if (st.sh != null) {
      try {
        const curDeg = g.shK != null ? DEG_OF_SHK[g.shK] : recipeTrainer.shDeg;
        if (curDeg == null) throw new Error(`unfamiliar band count ${g.shK}`);
        if (curDeg !== st.sh) {
          convertSHDegree(g, st.sh);
          log(`SH degree ${curDeg} → ${st.sh} for the continuation (low bands kept, new bands start at zero)`);
        }
        contShDeg = st.sh;
      } catch (e) {
        log(`keeping the run's SH degree (${e.message})`);
      }
    }
    if ((contShDeg ?? recipeTrainer.shDeg ?? 3) > 0) recipeTrainer.shLr = 3e-4;
    log(`continuing from ${fmt(baseIter)} cycles — +${fmt(extra)} more · cap ${fmt(cap)} · ${engine}` +
      `${prevDim ? ` · ${prevDim}px input` : ''}${!hasState ? ' · unbaking export opacities' : ''}`);
    // the panel's soft-prune selection applies to continuations like a fresh
    // run; unset, the recipe replays the run's own setting (the spread above
    // already carries it — deleting here silently unpruned continuations of
    // pruned runs)
    if (st.softprune) recipeTrainer.softPrune = { ratio: parseFloat(st.softprune), every: 3000, until: 0.85 };
    const ses = createSession({
      maxIters: S.maxIters, holdout: -1,
      gpuUsage: st.gpu / 100,
      ...(recipeRefine != null ? { refineEvery: recipeRefine } : {}),
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
    wireSession(ses);
    S.capture = recon.knownCameras
      ? { transforms: true, metadata: {}, levelingUp: recon.levelingUp || null } : null;
    // a model trained WITHOUT Mip compensation exported unbaked — the
    // recipe says so; records without one are legacy (baked). The raw
    // checkpoint never carries the baking.
    const wasBaked = recipe ? recipe.mipComp !== false : true;
    setStage('Decoding photos …', 0);
    await ses.load(files);
    setPhotoSources(files);
    ses.useReconstruction({
      known: !!recon.knownCameras,
      cams: recon.cams.map((c) => ({ imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f, cx: c.cx, cy: c.cy })),
      points: [],
      k1: recon.k1 || 0, k2: recon.k2 || 0,
      ...(recon.fFeat ? { fFeat: recon.fFeat } : {}),
    });
    if (undistortFrames(ses.frames, ses.recon)) { /* targets match the original run */ }
    S.phase = 'seed';
    setStage('Seeding gaussians …', 0);
    // the recon JSON carries the run's scene radius — without it every
    // radius-scaled quantity is wrong on resume
    await ses.seedFrom(g, { iter: baseIter, sceneRadius: recon.sceneRadius || undefined,
      ...(!hasState && wasBaked ? { unbake: true } : {}) });
    setupViewer(ses);
    // the run-so-far curve stays under the continued one
    const c = ensureChart();
    c.reset();
    c.train = ((recon.stats && recon.stats.chart) || []).map(([i, v]) => [i, v]);
    c.hold = ((recon.stats && recon.stats.holds) || []).map(([i, v]) => [i, v]);
    S.phase = 'train';
    ui.pause.disabled = false;
    ui.finish.disabled = false;
    ui.prune.disabled = false;
    S.trainT0 = performance.now();
    ses.start();
    S.loadedFiles = files.map((f, i) => ({ name: f.name, url: (urls && urls[i]) || null }));
    saveRunStart(files[0]);
    setStatus('Continuing — you can orbit the preview while it runs.');
  } catch (e) {
    failRun(e);
  }
}

function failRun(e) {
  console.error(e);
  if (S.session) { try { S.session.pause(); } catch { /* already down */ } }
  S.phase = 'error';
  setStage('Failed', 0);
  setStatus(e.message || String(e), 'error');
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.start.disabled = false;
}

function completeRun(e) {
  if (S.phase === 'done') return;
  S.phase = 'done';
  // a continuation's starting snapshot is stale now — the trained model is
  // the session's, saved through the run library, not the old restore
  S.restored = null;
  ui.cont.hidden = true;
  ui.needphotos.hidden = true;
  // LOD snapshots become their own downloads next to the full model
  ui.lodlinks.replaceChildren();
  const snaps = (S.lodPlan && S.lodPlan.snaps) || [];
  if (snaps.length) {
    ui.lodlinks.append('LOD snapshots: ');
    for (const { n, blob } of snaps) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `splat-lod-${n}.ply`;
      a.textContent = `${fmt(n)} ↓`;
      a.className = 'linkish';
      ui.lodlinks.append(a, ' ');
    }
    ui.lodlinks.hidden = false;
  } else {
    ui.lodlinks.hidden = true;
  }
  setStage(`Done · ${fmt(e.iter)} iterations`, 1);
  ui.metrics.textContent = `${fmt(e.splats ?? S.session?.trainer?.n)} splats · finished`;
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.ply.disabled = false;
  ui.simplify.disabled = false;
  ui.sess.disabled = false;
  ui.prune.disabled = true;
  setStatus('Training finished — download the .ply, or keep orbiting the preview.');
  S.minutes = minutesLive();
  secureFinishedRun();
  ui.start.disabled = false;
  ui.start.textContent = 'Train again ↻';
  if (!ui.details.hidden) renderDetails();
}

function resetRun() {
  if (S.session) { try { S.session.pause(); } catch { /* gone */ } }
  S.session = null;
  S.capture = null;
  S.restored = null;
  S.lodPlan = null;
  S.chartEvents = []; S.holdHist = []; S.perfMetrics = [];
  S.psnrTrain = null; S.psnrHold = null; S.itersPerSec = 0; S.etaAt = null; S.minutes = 0;
  S.trainT0 = 0; S.runSettings = null; S.runName = ''; S.gpuFacts = null;
  S.runId = null; S.loadedFiles = []; S.lastPersist = 0;
  stopTour();
  for (const u of S.stripUrls || []) { try { URL.revokeObjectURL(u); } catch { /* gone */ } }
  S.stripUrls = []; S.photoBlobs = [];
  S.sogBlob = null; S._sogJob = null;
  ui.strip.replaceChildren();
  ui.strip.hidden = true;
  ui.sess.disabled = true;
  ui.imgs.disabled = true;
  chart?.reset();
  ui.details.hidden = true;
  ui.cont.hidden = true;
  ui.needphotos.hidden = true;
  if (ui.photosFile) ui.photosFile.value = '';
  for (const a of ui.lodlinks.querySelectorAll('a')) URL.revokeObjectURL(a.href);
  ui.lodlinks.replaceChildren();
  ui.lodlinks.hidden = true;
  S.orbit = null;
  S.phase = 'idle';
  ui.run.hidden = true;
  ui.viewer.hidden = true;
  ui.metrics.textContent = '';
  ui.bar.style.width = '0%';
  ui.ply.disabled = true;
  ui.prune.disabled = true;
  ui.simplify.disabled = true;
  ui.start.textContent = 'Start training →';
  setStatus('');
}

// ── preview: a small orbit around the solve ─────────────────────────────────

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const m = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / m, a[1] / m, a[2] / m]; };
/** The solve's world frame is arbitrary — "up" is not gravity. The dominant
 *  up over all cameras (each one's world up is minus its second R row;
 *  photographers hold the camera roughly level on average) becomes the
 *  orbit's axis, so horizons sit balanced. Solver runs only — known-camera
 *  runs level to the capture report instead (see setupViewer). */
function dominantUp(cams) {
  const acc = [0, 0, 0];
  let n = 0;
  for (const c of cams) {
    if (!c.R) continue;
    acc[0] -= c.R[3]; acc[1] -= c.R[4]; acc[2] -= c.R[5]; n++;
  }
  return n && Math.hypot(...acc) > 1e-6 ? norm(acc) : [0, 1, 0];
}

/** world position of a camera from its world-to-camera pose */
const camCentre = ({ R, t }) => [
  -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
  -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
  -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
];

function setupViewer(session) {
  stopTour();   // the orbit is rebuilt below — a playing tour would aim the old one
  const metas = session.trainer.camMeta || [];
  const meta = metas.length ? metas[Math.floor(metas.length / 2)] : null;
  const cv = ui.view;
  // a bare model has no training cameras — orbit a default frame instead
  cv.width = meta ? meta.w : 960; cv.height = meta ? meta.h : 600;
  session.view.attach(cv);
  ui.viewer.hidden = false;

  // known-camera runs keep the capture world's y-up frame — level to its
  // gravity (the capture report), not to the cameras' sensor-up
  // (portrait-held phones point it sideways, which rolls the view 90°)
  const cap = S.capture;
  const up = cap && cap.transforms
    ? gravityUpForKnownRun({ persisted: cap.levelingUp || null,
        leveling: cap.metadata && cap.metadata.leveling })
    : (session.recon && session.recon.cams ? dominantUp(session.recon.cams) : [0, 1, 0]);
  let A = cross(up, [0, 0, 1]);
  if (Math.hypot(...A) < 1e-3) A = cross(up, [1, 0, 0]);
  A = norm(A);
  const B = cross(up, A);
  const center = session.model?.center || [0, 0, 0];
  const radius = session.model?.radius || 1;

  // start where a real photo was taken: the middle camera's angles — or a
  // three-quarter view when the model brought no cameras of its own
  S.orbit = { up, A, B, center, radius, yaw: 0, pitch: 0,
    dist: radius * 2.5, f: meta ? meta.f : Math.max(cv.width, cv.height) * 1.1,
    cx: meta ? meta.cx : cv.width / 2, cy: meta ? meta.cy : cv.height / 2 };
  if (meta) frameOrbitFromCamera(Math.floor(metas.length / 2));
  else { S.orbit.yaw = 0.6; S.orbit.pitch = 0.35; applyOrbit(); }
  renderStrip();
  wireOrbit();
}

/** Point the orbit at training camera ci (used at setup and by reset view),
 *  keeping the drag state consistent with what's on screen. */
function frameOrbitFromCamera(ci) {
  const s = S.session, o = S.orbit;
  if (!s || !o || !s.trainer.camMeta.length) return;
  const cam = s.trainer.camMeta[clamp(ci, 0, s.trainer.camMeta.length - 1)];
  const fwd = norm(sub(o.center, camCentre(cam)));
  const d = dot(fwd, o.up);
  const h = sub(fwd, [o.up[0] * d, o.up[1] * d, o.up[2] * d]);
  o.yaw = Math.atan2(dot(h, o.A), dot(h, o.B));
  o.pitch = Math.asin(clamp(d, -1, 1));
  o.dist = o.radius * 2.5;
  o.f = cam.f; o.cx = cam.cx; o.cy = cam.cy;
  applyOrbit();
}

function orbitFwd(o) {
  const cp = Math.cos(o.pitch);
  return norm([
    o.up[0] * Math.sin(o.pitch) + cp * (o.A[0] * Math.cos(o.yaw) + o.B[0] * Math.sin(o.yaw)),
    o.up[1] * Math.sin(o.pitch) + cp * (o.A[1] * Math.cos(o.yaw) + o.B[1] * Math.sin(o.yaw)),
    o.up[2] * Math.sin(o.pitch) + cp * (o.A[2] * Math.cos(o.yaw) + o.B[2] * Math.sin(o.yaw)),
  ]);
}

function applyOrbit() {
  const o = S.orbit, s = S.session;
  if (!o || !s) return;
  const fwd = orbitFwd(o);
  const pos = sub(o.center, [fwd[0] * o.dist, fwd[1] * o.dist, fwd[2] * o.dist]);
  const r = norm(cross(fwd, o.up));
  const u = cross(r, fwd);
  const R = [r[0], r[1], r[2], u[0], u[1], u[2], -fwd[0], -fwd[1], -fwd[2]];
  const t = [-dot([R[0], R[3], R[6]], pos), -dot([R[1], R[4], R[7]], pos), -dot([R[2], R[5], R[8]], pos)];
  s.view.setCamera({ R, t, f: o.f, cx: o.cx, cy: o.cy, w: ui.view.width, h: ui.view.height });
}

let orbitWired = false;
function wireOrbit() {
  if (orbitWired) return;
  orbitWired = true;
  const cv = ui.view;
  let drag = null;
  cv.style.touchAction = 'none';
  cv.addEventListener('pointerdown', (e) => {
    stopTour();   // a grab takes over from the fly-through
    drag = { x: e.clientX, y: e.clientY };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', (e) => {
    if (!drag || !S.orbit) return;
    S.orbit.yaw -= (e.clientX - drag.x) * 0.005;
    S.orbit.pitch = clamp(S.orbit.pitch + (e.clientY - drag.y) * 0.005, -1.45, 1.45);
    drag = { x: e.clientX, y: e.clientY };
    applyOrbit();
  });
  cv.addEventListener('pointerup', () => { drag = null; });
  cv.addEventListener('pointercancel', () => { drag = null; });
  cv.addEventListener('wheel', (e) => {
    if (!S.orbit) return;
    e.preventDefault();
    S.orbit.dist = clamp(S.orbit.dist * Math.exp(e.deltaY * 0.001),
      S.orbit.radius * 0.4, S.orbit.radius * 8);
    applyOrbit();
  }, { passive: false });
}

// ── tour + strip: fly the capture, or jump to any photo ─────────────────────

const vdist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** The camera fly-through as a resampled path: co-located cameras collapse
 *  (a pano rig's six orientations share one position — flying "through"
 *  them whips the view), real breaks in the capture split the walk and the
 *  longest leg flies, positions pre-smooth over ±3 neighbours, and a
 *  Catmull-Rom through the smoothed nodes resamples into an arc-length
 *  table so playback walks at exactly constant velocity. Same path math as
 *  the full app's tour; the browser flies it orbit-style (leveled, aimed
 *  at the model) instead of through a roll-carrying pose. */
function tourPath(cams) {
  cams = (cams || []).filter((c) => c.R);
  if (cams.length < 2) return null;
  const centres = cams.map(camCentre);
  const gaps = [];
  for (let i = 1; i < centres.length; i++) {
    const d = vdist(centres[i], centres[i - 1]);
    if (d > 1e-9) gaps.push(d);
  }
  gaps.sort((a, b) => a - b);
  const medGap = gaps[gaps.length >> 1] || 1e-3;
  const dedup = [];
  for (let i = 0; i < cams.length; i++) {
    const last = dedup[dedup.length - 1];
    if (!last || vdist(centres[i], camCentre(last)) > 0.15 * medGap) dedup.push(cams[i]);
  }
  const segs = [[dedup[0]]];
  for (let i = 1; i < dedup.length; i++) {
    if (vdist(camCentre(dedup[i - 1]), camCentre(dedup[i])) > 4 * medGap) segs.push([]);
    segs[segs.length - 1].push(dedup[i]);
  }
  const leg = segs.reduce((best, s) => (s.length > best.length ? s : best), []);
  const n = leg.length;
  if (n < 2) return null;
  const raw = leg.map(camCentre);
  const pts = raw.map((_, i) => {
    const acc = [0, 0, 0];
    let w = 0;
    for (let k = -3; k <= 3; k++) {
      const j = clamp(i + k, 0, raw.length - 1), wt = 4 - Math.abs(k);
      for (let c = 0; c < 3; c++) acc[c] += raw[j][c] * wt;
      w += wt;
    }
    return [acc[0] / w, acc[1] / w, acc[2] / w];
  });
  const P = (k) => pts[clamp(k, 0, n - 1)];
  const cr = (i, f) => {
    const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
    const t2 = f * f, t3 = t2 * f;
    return [0, 1, 2].map((k) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * f +
      (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
      (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3));
  };
  const samples = [], cum = [0];
  const SUB = 8;
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < SUB; k++) samples.push(cr(i, k / SUB));
  }
  samples.push(cr(n - 2, 1));
  for (let k = 1; k < samples.length; k++) cum.push(cum[k - 1] + vdist(samples[k - 1], samples[k]));
  return { samples, cum, total: cum[cum.length - 1] || 1e-6, count: n };
}

function startTour() {
  const s = S.session, o = S.orbit;
  if (!s || !o || !s.recon || !(s.recon.cams || []).some((c) => c.R)) return;
  stopTour();
  const path = tourPath(s.recon.cams);
  if (!path) return;
  // pick up from wherever the view is — no jump-cut into the path
  const eye = orbitEye(o);
  let best = 0, bd = Infinity;
  path.samples.forEach((p, k) => {
    const dd = (p[0] - eye[0]) ** 2 + (p[1] - eye[1]) ** 2 + (p[2] - eye[2]) ** 2;
    if (dd < bd) { bd = dd; best = k; }
  });
  const duration = clamp(1.4 * path.count, 10, 30);
  S.tour = {
    ...path, speed: path.total / duration, s: path.cum[best],
    k: Math.min(best, path.cum.length - 2), dir: 1, last: performance.now(), raf: 0,
  };
  ui.tour.textContent = 'stop the tour';
  S.tour.raf = requestAnimationFrame(tourTick);
}

function stopTour() {
  const T = S.tour;
  if (!T) return;
  S.tour = null;
  if (T.raf) cancelAnimationFrame(T.raf);
  ui.tour.textContent = 'tour the cameras';
}

function tourTick() {
  const T = S.tour, o = S.orbit;
  if (!T || !o) return;
  const now = performance.now();
  const dt = Math.min(0.1, Math.max(0, (now - T.last) / 1000));
  T.last = now;
  T.s += dt * T.speed * T.dir;               // constant velocity, ping-pong
  if (T.s >= T.total) { T.s = T.total; T.dir = -1; }
  if (T.s <= 0) { T.s = 0; T.dir = 1; }
  while (T.k < T.cum.length - 2 && T.cum[T.k + 1] < T.s) T.k++;
  while (T.k > 0 && T.cum[T.k] > T.s) T.k--;
  const span = Math.max(1e-9, T.cum[T.k + 1] - T.cum[T.k]);
  const a = (T.s - T.cum[T.k]) / span;
  const A = T.samples[T.k], B = T.samples[T.k + 1];
  aimOrbitAt([A[0] + (B[0] - A[0]) * a, A[1] + (B[1] - A[1]) * a, A[2] + (B[2] - A[2]) * a]);
  T.raf = requestAnimationFrame(tourTick);
}

function orbitEye(o) {
  const fwd = orbitFwd(o);
  return sub(o.center, [fwd[0] * o.dist, fwd[1] * o.dist, fwd[2] * o.dist]);
}

/** Aim the orbit from a path eye: the tour flies the capture positions
 *  while the leveled orbit keeps aiming at the model. */
function aimOrbitAt(eye) {
  const o = S.orbit;
  if (!o) return;
  const fwd = norm(sub(o.center, eye));
  const d = dot(fwd, o.up);
  const h = sub(fwd, [o.up[0] * d, o.up[1] * d, o.up[2] * d]);
  o.yaw = Math.atan2(dot(h, o.A), dot(h, o.B));
  o.pitch = Math.asin(clamp(d, -1, 1));
  o.dist = clamp(vdist(o.center, eye), o.radius * 0.4, o.radius * 8);
  applyOrbit();
}

/** The filmstrip: every photo that trained this run, in frame order — a
 *  click frames the orbit from that photo's own camera. Restores carry no
 *  photo pixels, so the strip stays hidden for them. */
function renderStrip() {
  const s = S.session;
  ui.strip.replaceChildren();
  const n = s && s.frames ? s.frames.length : 0;
  if (!s || !n || !(S.stripUrls || []).some(Boolean)) { ui.strip.hidden = true; return; }
  s.frames.forEach((f, i) => {
    if (!S.stripUrls[i]) return;
    const b = document.createElement('button');
    b.className = 'thumb';
    b.title = f.name || `photo ${i + 1}`;
    const img = document.createElement('img');
    img.src = S.stripUrls[i];
    img.alt = '';
    img.loading = 'lazy';
    b.append(img);
    b.addEventListener('click', () => jumpToFrame(i));
    ui.strip.append(b);
  });
  ui.strip.hidden = !ui.strip.children.length;
}

/** Frame i photographed through camMeta[k] — camMeta and recon.cams share
 *  their order, so the camera is found by imgIdx. */
function jumpToFrame(i) {
  const s = S.session;
  if (!s) return;
  stopTour();
  const cams = (s.recon && s.recon.cams) || [];
  const k = cams.findIndex((c) => c.imgIdx === i);
  frameOrbitFromCamera(k >= 0 ? k : i);
}

/** The run's photo pixels, kept for the strip + the photos export (a File
 *  is its own blob; video frames ride on .source — same as the full app). */
function setPhotoSources(files) {
  for (const u of S.stripUrls || []) { try { URL.revokeObjectURL(u); } catch { /* gone */ } }
  S.photoBlobs = [...files];
  S.stripUrls = files.map((f) => {
    try { return URL.createObjectURL(f.source || f); } catch { return null; }
  });
  ui.imgs.disabled = !S.photoBlobs.length;
}

// ── wiring ──────────────────────────────────────────────────────────────────

ui.files.addEventListener('change', () => stageFiles(ui.files.files));
for (const ev of ['dragover', 'dragenter']) {
  ui.drop.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add('over'); });
}
for (const ev of ['dragleave', 'drop']) {
  ui.drop.addEventListener(ev, (e) => {
    e.preventDefault(); ui.drop.classList.remove('over');
    if (ev === 'drop' && e.dataTransfer?.files?.length) stageFiles(e.dataTransfer.files);
  });
}
ui.start.addEventListener('click', () => { if (S.staged) run(); });
ui.q.addEventListener('change', () => pickQuality(ui.q.value));
for (const el of [ui.res, ui.feat, ui.tier, ui.buf, ui.sh, ui.splats, ui.lod, ui.lodCount, ui.mcmc, ui.softprune, ui.iters]) {
  el.addEventListener('change', syncQuality);
}
ui.softprune.addEventListener('change', () => {
  ui.softpruneCustom.hidden = ui.softprune.value !== 'custom';
});
ui.gpu.addEventListener('input', () => {
  ui.gpuV.textContent = `${ui.gpu.value}%`;
  // live like the full app's dock slider: the loop reads opts.gpuUsage
  // every batch, so a running session picks this up without a restart
  if (S.session) S.session.opts.gpuUsage = parseInt(ui.gpu.value, 10) / 100;
});
syncQuality();
ui.recenter.addEventListener('click', () => {
  const s = S.session;
  if (!s || !S.orbit) return;
  stopTour();
  frameOrbitFromCamera(Math.floor(s.trainer.camMeta.length / 2));
});
ui.tour.addEventListener('click', () => {
  if (S.tour) stopTour();
  else startTour();
});
ui.pause.addEventListener('click', () => {
  const s = S.session;
  if (!s || (S.phase !== 'train' && S.phase !== 'paused')) return;
  if (S.phase === 'train') {
    s.pause();
    S.phase = 'paused';
    ui.pause.textContent = 'Resume';
    setStatus('Paused — resume anytime; nothing is lost.');
    persistRunDetails();
  } else {
    S.phase = 'train';
    s.start();
    ui.pause.textContent = 'Pause';
    setStatus('Training — you can orbit the preview while it runs.');
  }
});
ui.finish.addEventListener('click', async () => {
  const s = S.session;
  if (!s || S.phase !== 'train') return;
  ui.finish.disabled = true;
  setStatus('Finishing with the model as it stands …');
  await s.finish();   // emits train-complete → completeRun
});
ui.cont.addEventListener('click', continueClick);
for (const [k, el] of Object.entries(ui.tabs)) {
  el.addEventListener('click', () => { S.detailTab = k; renderDetails(); });
}
ui.details.addEventListener('toggle', () => { if (ui.details.open) renderDetails(); });
ui.scale.addEventListener('click', () => {
  S.scoreScale = S.scoreScale === 'similarity' ? 'db' : 'similarity';
  ui.scale.textContent = S.scoreScale === 'similarity' ? 'show as dB' : 'show as %';
  if (chart) { chart.scale = S.scoreScale; chart.draw(); }
  if (!ui.details.hidden && ui.details.open) renderDetails();
});
globalThis.addEventListener?.('resize', () => chart?.resize());
// a hidden tab mid-training may never come back — secure the progress line
if (typeof document !== 'undefined' && document.addEventListener) {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && S.phase === 'train') persistRunDetails();
  });
}
ui.lastcapUse.addEventListener('click', async () => {
  const rec = await loadLastCapture().catch(() => null);
  if (!rec || !rec.files || !rec.files.length) { ui.lastcap.hidden = true; return; }
  stageFiles(rec.files.map(({ name, blob }) => new File([blob], name, { type: blob.type || '' })));
});
ui.lastcapForget.addEventListener('click', async () => {
  await deleteLastCapture().catch(() => {});
  ui.lastcap.hidden = true;
});
ui.photosFile.addEventListener('change', () => {
  const R = S.restored;
  if (!R || !R.reconJson) return;
  const names = (R.reconJson.frames || []).map((f) => f.name);
  const { files, missing } = matchPhotosByName(ui.photosFile.files, names);
  if (missing) {
    setStatus(`Still missing ${missing.length} photo${missing.length > 1 ? 's' : ''} (e.g. ${missing[0]}).`, 'error');
    return;
  }
  continueRun(files);
});
ui.ply.addEventListener('click', async () => {
  const s = S.session;
  if (!s) return;
  ui.ply.disabled = true;
  try {
    setStatus('Packing the .ply …');
    const blob = await s.exportPlyBlob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `splat-${Date.now()}.ply`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
    setStatus(`Saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB, standard 3DGS .ply).`);
  } catch (e) {
    console.error(e);
    setStatus(`PLY export failed: ${e.message}`, 'error');
  } finally {
    ui.ply.disabled = false;
  }
});
async function runPruneContinue(ratio = 0.5) {
  const s = S.session;
  if (!s || !s.trainer || S._pruning) return;
  S._pruning = true;
  ui.prune.disabled = true;
  try {
    setStatus('Pruning the model …');
    const out = await s.pruneAndContinue(ratio, {
      onStatus: (m) => setStatus(String(m).replace(/^prune: /, '')),
    });
    const d = out.delta == null ? 'no holdout to score against'
      : `${out.delta >= 0 ? '+' : ''}${out.delta.toFixed(2)} dB holdout`;
    setStatus(`Pruned ${fmt(out.from)} → ${fmt(out.to)} splats (${d}) — training on.`);
  } catch (e) {
    console.error(e);
    setStatus(`Prune failed: ${e.message} — resume to carry on.`, 'error');
  } finally {
    S._pruning = false;
    ui.prune.disabled = S.phase !== 'train';
  }
}
/** Manual growth pass, any time: one immediate refine outside the schedule
 *  (the Prune/Grow button's other half). */
async function runGrowNow() {
  const s = S.session;
  if (!s || !s.trainer || S._pruning || S._growing) return;
  S._growing = true;
  ui.prune.disabled = true;
  try {
    setStatus('Growing into the error …');
    const r = await s.growNow();
    setStatus(`Grew +${fmt(r.grown || 0)} splats → ${fmt(r.n)} total.`);
  } catch (e) {
    console.error(e);
    setStatus(`Grow failed: ${e.message}`, 'error');
  } finally {
    S._growing = false;
    ui.prune.disabled = S.phase !== 'train';
  }
}
ui.prune.addEventListener('click', showPrunePicker);
/** The ceiling's question, asked inline in the status line: the session
 *  paused itself at a full cap — prune to half and grow on, or keep
 *  training at this size. */
function showCapPrompt(e) {
  const st = ui.status;
  st.replaceChildren();
  const span = document.createElement('span');
  span.textContent = `Splat cap reached (${fmt(e.splats)}): `;
  const mk = (label, cls, fn) => {
    const b = document.createElement('button');
    b.className = cls;
    b.textContent = label;
    b.addEventListener('click', fn);
    return b;
  };
  const go = (ratio) => { setStatus('Pruning the model …'); runPruneContinue(ratio); };
  const custom = document.createElement('input');
  custom.type = 'number'; custom.min = '5'; custom.max = '95'; custom.value = '50';
  custom.style.width = '4.5em'; custom.title = 'Custom keep percentage (5–95)';
  const goCustom = () => {
    const pct = Math.round(parseFloat(custom.value) || 0);
    go((pct >= 5 ? Math.min(95, pct) : 50) / 100);
  };
  st.append(span,
    mk('Prune to ¾', 'btn', () => go(0.75)), ' ',
    mk('Prune to ½', 'btn accent', () => go(0.5)), ' ',
    mk('Prune to ¼', 'btn', () => go(0.25)), ' ',
    custom, ' ',
    mk('Prune to %', 'btn', goCustom), ' ',
    mk('Keep training at cap', 'btn', () => {
      if (S.session) S.session.start();
      setStatus('Training on at the cap — growth stays off.');
    }));
}
/** Manual prune/grow, any time: the status line becomes a ratio picker
 *  with a growth pass on the side. */
function showPrunePicker() {
  const s = S.session;
  if (!s || !s.trainer || S._pruning) return;
  const st = ui.status;
  st.replaceChildren();
  const span = document.createElement('span');
  span.textContent = 'Prune to: ';
  const mk = (label, fn) => {
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = label;
    b.addEventListener('click', fn);
    return b;
  };
  const go = (ratio) => { setStatus('Pruning the model …'); runPruneContinue(ratio); };
  const custom = document.createElement('input');
  custom.type = 'number'; custom.min = '5'; custom.max = '95'; custom.value = '50';
  custom.style.width = '4.5em'; custom.title = 'Custom keep percentage (5–95)';
  const goCustom = () => {
    const pct = Math.round(parseFloat(custom.value) || 0);
    go((pct >= 5 ? Math.min(95, pct) : 50) / 100);
  };
  st.append(span,
    mk('Keep ¾', () => go(0.75)), ' ',
    mk('Keep ½', () => go(0.5)), ' ',
    mk('Keep ¼', () => go(0.25)), ' ',
    mk('Grow now', runGrowNow), ' ',
    custom, ' ',
    mk('Keep %', goCustom), ' ',
    mk('Cancel', () => setStatus('')));
}
ui.simplify.addEventListener('click', async () => {
  const s = S.session;
  if (!s || S.restored) return;   // the raw state needs its live trainer
  ui.simplify.disabled = true;
  try {
    setStatus('Simplifying the model …');
    const blob = await s.exportSimplifiedPlyBlob(0.5, {
      onStatus: (m) => setStatus(String(m).replace(/^simplify: /, '')),
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `splat-simplified-${Date.now()}.ply`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
    setStatus(`Saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB, half the splats).`);
  } catch (e) {
    console.error(e);
    setStatus(`Simplify failed: ${e.message}`, 'error');
  } finally {
    ui.simplify.disabled = false;
  }
});
/** The compressed model, built once per run: GPU k-means over the .ply.
 *  On demand only — finishing never waits on it, like the full app. */
async function getSogBlob() {
  if (S.sogBlob) return S.sogBlob;
  if (S._sogJob) return S._sogJob;
  return (S._sogJob = (async () => {
    try {
      setStatus('Compressing the model …');
      const ply = await S.session.exportPlyBlob();
      const blob = await plyToSog(new Uint8Array(await ply.arrayBuffer()), {
        onProgress: ({ label, frac }) => {
          if (label) setStage(`Compressing · ${label}`, frac);
        },
      });
      S.sogBlob = blob;
      return blob;
    } finally {
      S._sogJob = null;   // a failure clears the job so a retry works
    }
  })());
}
ui.sess.addEventListener('click', async () => {
  const s = S.session;
  if (!s || S.restored) return;   // the raw state needs its live trainer
  ui.sess.disabled = true;
  try {
    const sog = await getSogBlob();
    setStatus('Packing the session …');
    const zip = await buildSessionZip(reconFacade(), sog);
    await saveBlob(zip, `splat-session-${Date.now()}.zip`, 'application/zip');
    setStatus(`Session saved (${(zip.size / 1e6).toFixed(1)} MB) — drop it back to resume exactly.`);
  } catch (e) {
    console.error(e);
    setStatus(`Session export failed: ${e.message}`, 'error');
  } finally {
    ui.sess.disabled = false;
  }
});
ui.imgs.addEventListener('click', async () => {
  if (!S.photoBlobs.length) return;
  ui.imgs.disabled = true;
  try {
    setStatus('Packing the photos …');
    const entries = [];
    for (const f of S.photoBlobs) {
      const blob = f.source || f;
      entries.push({ name: f.name, data: new Uint8Array(await blob.arrayBuffer()) });
    }
    const zip = zipStore(entries);
    await saveBlob(zip, `photos-${Date.now()}.zip`, 'application/zip');
    setStatus(`${entries.length} photos saved in a ZIP.`);
  } catch (e) {
    console.error(e);
    setStatus(`Photos export failed: ${e.message}`, 'error');
  } finally {
    ui.imgs.disabled = false;
  }
});

if (!navigator.gpu) {
  ui.nogl.hidden = false;
  ui.summary.textContent = 'WebGPU is unavailable, so training is disabled in this browser.';
} else {
  log('ready — drop photos, a video, or an exported .zip');
}
// the shelf + the last walked set are storage, not GPU — they load regardless
refreshLastCapture();
renderLibrary().catch(() => {});

// ?model=<url> (a session .zip, .ply or .sog; &recon=<url> adds the camera
// path to bare model files) opens straight in the viewer, like the full app
if (typeof location !== 'undefined') {
  const q = new URLSearchParams(location.search);
  if (q.get('model')) {
    S.staged = { kind: 'model-url', url: q.get('model'), reconUrl: q.get('recon') };
    ui.summary.textContent = 'Opening the linked model …';
    ui.start.disabled = true;
    run();
  }
}

// pure intake helpers, exported for the Node tests (the module is otherwise
// a UI entry point, not a library)
export { assembleCapture, sortByCapture, modelFromEntries, matchPhotosByName, runNameFor, describeRun, tourPath };
