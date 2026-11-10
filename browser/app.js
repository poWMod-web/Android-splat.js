// browser/app.js — the browser-only trainer: photos, video, or an exported
// .zip in, a standard 3DGS .ply out. No capture, no device features — just
// the shared engine (../src) driven start to finish in one tab.
//
// Serve the REPO ROOT over http (ES modules + WebGPU forbid file://):
//   python3 -m http.server 8000   →   http://localhost:8000/browser/

import {
  createSession, parseCameras, extractSharpFrames, isVideoFile, solveTierOpts,
} from '../src/index.js';
import { readZip } from './zip.js';

const $ = (id) => document.getElementById(id);
const ui = {
  nogl: $('nogl'), files: $('files'), drop: $('drop'),
  tier: $('opt-tier'), iters: $('opt-iters'), gap: $('opt-gap'),
  summary: $('input-summary'), start: $('btn-start'),
  run: $('run'), stage: $('stage-label'), bar: $('stage-bar'), metrics: $('metrics'),
  pause: $('btn-pause'), finish: $('btn-finish'), ply: $('btn-ply'), status: $('status'),
  viewer: $('viewer'), view: $('view'), recenter: $('btn-recenter'),
  logwrap: $('logwrap'), log: $('log'),
};

const S = {
  staged: null,     // { kind: 'photos'|'video'|'zip', ... } — validated at Start
  session: null,
  maxIters: 30000,
  phase: 'idle',    // idle → prep → solve → train → done | error
  orbit: null,
  lines: 0,
};

const fmt = (n) => n == null || Number.isNaN(n) ? '—' : Math.round(n).toLocaleString('en-US');
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

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
    return;
  }
  const images = files.filter((f) => (f.type || '').startsWith('image/') || IMG_EXT.test(f.name || ''));
  const videos = files.filter(isVideoFile);
  const jsons = files.filter((f) => /\.json$/i.test(f.name || ''));
  if (videos.length && images.length < 2) {
    S.staged = { kind: 'video', file: videos[0] };
    ui.summary.textContent = `${videos[0].name} — sharp frames will be picked from it automatically.`;
    ui.start.disabled = false;
    return;
  }
  if (!images.length) {
    ui.summary.textContent = videos.length
      ? 'That video came with photos too — but fewer than 2 are usable. Pick at least a couple of overlapping photos.'
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
  ui.start.disabled = images.length < 2;
  if (images.length < 2) ui.summary.textContent = 'Pick at least a couple of overlapping photos of the same place.';
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
  if (!images.length && !videos.length) {
    const hasSession = ['recon.json', 'state.bin', 'model.sog'].some((n) =>
      names.some((e) => baseName(e).toLowerCase() === n));
    if (hasSession) {
      throw new Error('This looks like a trained-session export (recon.json / state.bin / model.sog): ' +
        'it carries the finished model but no source images, so there is nothing to train from here. ' +
        'To train, use the photos, a video, or a capture .zip with the frames in it.');
    }
    throw new Error(`No images in this zip (saw: ${names.slice(0, 6).map(baseName).join(', ') || 'nothing readable'}). ` +
      'A capture .zip holds the JPEGs next to transforms.json — unzip-and-drop also works.');
  }
  return { images, jsons, videos };
}

// ── the run ─────────────────────────────────────────────────────────────────

function findCamerasSidecar(jsons) {
  const byName = (n) => jsons.find((j) => j.name.toLowerCase() === n)?.json;
  return byName('transforms.json') || byName('cameras.json') || null;
}
function findPoints(jsons) {
  const doc = jsons.find((j) => j.name.toLowerCase() === 'seed_points.json')?.json;
  const pts = doc && Array.isArray(doc.points) ? doc.points : [];
  return pts.length >= 8 ? pts : [];
}

async function gatherInputs(staged) {
  if (staged.kind === 'zip') {
    setStage(`Unpacking ${staged.file.name} …`, 0);
    const { images, jsons, videos } = await unpackZip(staged.file, (f) =>
      setStage(`Unpacking ${staged.file.name} …`, f));
    log(`zip: ${images.length} images, ${jsons.length} JSON, ${videos.length} video`);
    if (images.length >= 2) return { files: images, jsons, video: false };
    if (videos.length) return gatherVideo(videos[0]);
    throw new Error(`Only ${images.length} usable image${images.length === 1 ? '' : 's'} in this zip — need at least 2.`);
  }
  if (staged.kind === 'video') return gatherVideo(staged.file);
  const jsons = [];
  for (const f of staged.jsons) {
    try { jsons.push({ name: f.name, json: JSON.parse(await f.text()) }); }
    catch { throw new Error(`${f.name} is not valid JSON`); }
  }
  return { files: staged.images, jsons, video: false };
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
    const { readVideoMeta } = await import('../src/io/qtmeta.js');
    const vm = await readVideoMeta(file);
    if (vm.focal35) {
      for (const f of frames) f.exif = { f35: vm.focal35, lens: vm.lensModel, video: true };
      log(`[video] lens ${vm.lensModel || '?'} · ${vm.focal35}mm equiv — focal prior for the solve`);
    }
  } catch (e) { log(`[video] lens metadata: ${e.message || e}`); }
  log(`video: ${frames.length} sharp frames from ${duration.toFixed(0)}s`);
  return { files: frames, jsons: [], video: true };
}

async function run() {
  if (S.phase === 'prep' || S.phase === 'solve' || S.phase === 'train') return;
  S.phase = 'prep';
  ui.start.disabled = true;
  ui.run.hidden = false;
  ui.ply.disabled = true;
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.pause.textContent = 'Pause';
  setStatus('');
  S.maxIters = parseInt(ui.iters.value, 10) || 30000;

  try {
    const { files, jsons, video } = await gatherInputs(S.staged);
    if (files.length < 2) throw new Error('Need at least 2 usable images.');
    const knownJson = findCamerasSidecar(jsons);
    const points = findPoints(jsons);
    log(`${files.length} images${knownJson ? ' + known cameras' : ''}${points.length ? ` + ${fmt(points.length)} seed points` : ''}${video ? ' (from video)' : ''}`);

    const session = createSession({
      maxIters: S.maxIters,
      holdout: -1,
      evalHoldEvery: 2500,
      sfm: {
        ...solveTierOpts(ui.tier.value),
        // a video is a chain: consecutive frames overlap by construction, so
        // pairs gate on absolute inlier counts, not the ratio photo sets use
        ...(video ? { pairMinInliers: 100, pairMinInliersAdj: 15 } : {}),
      },
    });
    S.session = session;
    session.on('log', (m) => log(m));
    session.on('stage', (e) => {
      if (e.stage === 'train') return;   // the metrics line owns the bar then
      const label = { decode: 'Decoding photos', solve: 'Solving cameras', seed: 'Seeding gaussians' }[e.stage] || e.stage;
      setStage(e.total > 1 ? `${label} · ${e.done} / ${e.total}` : label,
        e.total ? e.done / e.total : null);
    });
    session.on('metrics', (m) => {
      setStage(`Training · iteration ${fmt(m.iter)} / ${fmt(S.maxIters)}`, m.iter / S.maxIters);
      const bits = [`${fmt(m.splats)} splats`, `${fmt(m.itersPerSec)} it/s`];
      if (m.psnrTrain > 0) bits.push(`train PSNR ${m.psnrTrain.toFixed(2)}`);
      if (m.psnrHold > 0) bits.push(`holdout ${m.psnrHold.toFixed(2)}`);
      ui.metrics.textContent = bits.join(' · ');
    });
    session.on('event', (e) => {
      if (e.kind === 'train-complete') completeRun(e);
      else if (e.kind === 'device-lost') failRun(new Error('The GPU was reclaimed mid-run (backgrounded tab or driver reset). Keep the tab visible and try again.'));
      else if (e.kind === 'refine') log(`refine @${fmt(e.iter)}: ${fmt(e.kept)} kept`);
    });

    setStage('Decoding photos …', 0);
    await session.load(files);
    if (knownJson) {
      S.phase = 'solve';
      setStage('Using the known cameras (solve skipped) …', 0);
      const recon = session.useKnownCameras(parseCameras(knownJson),
        points.length ? { points } : {});
      log(`known cameras: ${recon.cams.length} of ${files.length} frames matched by file name`);
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
    setupViewer(session);
    S.phase = 'train';
    ui.pause.disabled = false;
    ui.finish.disabled = false;
    session.start();
    setStatus('Training — you can orbit the preview while it runs.');
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
  setStage(`Done · ${fmt(e.iter)} iterations`, 1);
  ui.metrics.textContent = `${fmt(e.splats ?? S.session?.trainer?.n)} splats · finished`;
  ui.pause.disabled = true;
  ui.finish.disabled = true;
  ui.ply.disabled = false;
  setStatus('Training finished — download the .ply, or keep orbiting the preview.');
  ui.start.disabled = false;
  ui.start.textContent = 'Train again ↻';
}

function resetRun() {
  if (S.session) { try { S.session.pause(); } catch { /* gone */ } }
  S.session = null;
  S.orbit = null;
  S.phase = 'idle';
  ui.run.hidden = true;
  ui.viewer.hidden = true;
  ui.metrics.textContent = '';
  ui.bar.style.width = '0%';
  ui.ply.disabled = true;
  ui.start.textContent = 'Start training →';
  setStatus('');
}

// ── preview: a small orbit around the solve ─────────────────────────────────

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const m = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / m, a[1] / m, a[2] / m]; };
/** world position of a camera from its world-to-camera pose */
const camCentre = ({ R, t }) => [
  -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
  -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
  -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
];

function setupViewer(session) {
  const meta = session.trainer.camMeta[Math.floor(session.trainer.camMeta.length / 2)];
  const cv = ui.view;
  cv.width = meta.w; cv.height = meta.h;
  session.view.attach(cv);
  ui.viewer.hidden = false;

  // the solve's world frame is arbitrary — "up" is not gravity. The dominant
  // up over all cameras (each one's world up is minus its second R row;
  // photographers hold the camera roughly level on average) becomes the
  // orbit's axis, so horizons sit balanced.
  const acc = [0, 0, 0];
  let n = 0;
  for (const c of session.recon.cams) {
    if (!c.R) continue;
    acc[0] -= c.R[3]; acc[1] -= c.R[4]; acc[2] -= c.R[5]; n++;
  }
  const up = n && Math.hypot(...acc) > 1e-6 ? norm(acc) : [0, 1, 0];
  let A = cross(up, [0, 0, 1]);
  if (Math.hypot(...A) < 1e-3) A = cross(up, [1, 0, 0]);
  A = norm(A);
  const B = cross(up, A);
  const center = session.model?.center || [0, 0, 0];
  const radius = session.model?.radius || 1;

  // start where a real photo was taken: the middle camera's angles
  S.orbit = { up, A, B, center, radius, yaw: 0, pitch: 0,
    dist: radius * 2.5, f: meta.f, cx: meta.cx, cy: meta.cy };
  frameOrbitFromCamera(Math.floor(session.trainer.camMeta.length / 2));
  wireOrbit();
}

/** Point the orbit at training camera ci (used at setup and by reset view),
 *  keeping the drag state consistent with what's on screen. */
function frameOrbitFromCamera(ci) {
  const s = S.session, o = S.orbit;
  if (!s || !o) return;
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

function applyOrbit() {
  const o = S.orbit, s = S.session;
  if (!o || !s) return;
  const cp = Math.cos(o.pitch);
  const fwd = norm([
    o.up[0] * Math.sin(o.pitch) + cp * (o.A[0] * Math.cos(o.yaw) + o.B[0] * Math.sin(o.yaw)),
    o.up[1] * Math.sin(o.pitch) + cp * (o.A[1] * Math.cos(o.yaw) + o.B[1] * Math.sin(o.yaw)),
    o.up[2] * Math.sin(o.pitch) + cp * (o.A[2] * Math.cos(o.yaw) + o.B[2] * Math.sin(o.yaw)),
  ]);
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
ui.recenter.addEventListener('click', () => {
  const s = S.session;
  if (!s || !S.orbit) return;
  frameOrbitFromCamera(Math.floor(s.trainer.camMeta.length / 2));
});
ui.pause.addEventListener('click', () => {
  const s = S.session;
  if (!s || (S.phase !== 'train' && S.phase !== 'paused')) return;
  if (S.phase === 'train') {
    s.pause();
    S.phase = 'paused';
    ui.pause.textContent = 'Resume';
    setStatus('Paused — resume anytime; nothing is lost.');
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

if (!navigator.gpu) {
  ui.nogl.hidden = false;
  ui.summary.textContent = 'WebGPU is unavailable, so training is disabled in this browser.';
} else {
  log('ready — drop photos, a video, or an exported .zip');
}
