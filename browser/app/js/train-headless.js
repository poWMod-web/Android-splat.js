// train-headless.js — the migrate-on-swipe driver. Runs ONLY in the :training
// service's headless WebView (booted from onTaskRemoved), never in the app.
//
// It resumes the swiped-away run from the last file checkpoint — photos,
// state.bin and recon.json from files/runs/<runId>/ — and keeps training with
// no UI: progress goes to the service notification, checkpoints back to files,
// and snapshots serve the reopened app's monitor view. The service drives it
// through the __headless* hooks (evaluateJavascript) and the AndroidHeadless
// bridge (this page has no DOM reads that matter — no canvas, no viewport).
//
// V1 limits (documented, not silent): resume-from-state only (a swipe before
// the first checkpoint cannot resume); LOD level schedules are not continued
// (the model grows to the cap instead); avatar runs never stage a handoff.

import { createSession, undistortFrames } from '../../src/index.js';
import {
  parseState, packState, buildReconJson, continuedInputDim, continuedBuf,
} from './session_io.js';
import * as H from './headless.js';

const qs = new URLSearchParams(location.search);
const runId = qs.get('run') || '';
const bridge = globalThis.AndroidHeadless || null;
const log = (...a) => console.log('[headless]', ...a);

let ses = null;
let manifest = null;
let recon = null;          // the checkpoint's recon (cloud + leveling carry over)
let snapCam = null;
let files = [];
let ending = false;        // finalize/abort in flight — the tick stands down
let ckptBusy = false;
let lastProgAt = 0, lastProgIter = -1;
let ips = 0, ipsAt = 0, ipsIter = 0;
let shotBusy = false;
let psnr = null;

const b64 = (u8, size = 262144) => {
  const out = [];
  for (let o = 0; o < u8.length; o += size) out.push(H.encodeB64(u8.subarray(o, o + size)));
  return out;
};

async function fetchJson(name) {
  const r = await fetch(H.runUrl(runId, name));
  if (!r.ok) throw new Error(`${name} unreadable (${r.status})`);
  return r.json();
}
async function fetchBytes(name) {
  const r = await fetch(H.runUrl(runId, name));
  if (!r.ok) throw new Error(`${name} unreadable (${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

function postStatus(patch) {
  if (!bridge) return;
  try {
    bridge.writeRunJson(runId, 'status.json', JSON.stringify({
      status: 'active', owner: 'service', iter: ses && ses.trainer ? ses.trainer.iter : 0,
      at: Date.now(), ...patch,
    }));
  } catch (e) { log('status post failed', e); }
}

/** Fatal before/during boot: say so in the status file (the monitor turns it
 *  into words) and release the service — a wedged headless run must never
 *  squat on the GPU with no UI to stop it. */
async function fail(msg) {
  log('FATAL', msg);
  try { postStatus({ status: 'interrupted', error: String(msg && msg.message || msg) }); } catch { /* gone */ }
  try { bridge && bridge.abortRun(runId, 'Training interrupted', String(msg && msg.message || msg)); } catch { /* gone */ }
}

async function boot() {
  try {
    if (!H.validRunId(runId)) throw new Error('no run id');
    if (!bridge || typeof bridge.writeRunJson !== 'function') throw new Error('no headless bridge');
    manifest = H.validateManifest(await fetchJson('manifest.json'));
    if (manifest.runId !== runId) throw new Error('manifest names another run');
    const st = parseState(await fetchBytes('state.bin'));
    recon = await fetchJson('recon.json');
    if (!recon || !Array.isArray(recon.cams) || !recon.cams.length) throw new Error('recon has no cameras');
    // photos matched BY NAME to the run's own frame order (same rule as the
    // app's continue path — re-sorting against imgIdx would mis-train)
    const staged = new Set(manifest.photos);
    files = [];
    for (const fr of recon.frames || []) {
      if (!staged.has(fr.name)) throw new Error(`photo ${fr.name} was not staged`);
      const r = await fetch(H.photoUrl(runId, fr.name));
      if (!r.ok) throw new Error(`photo ${fr.name} unreadable (${r.status})`);
      files.push(new File([await r.blob()], fr.name, { type: 'image/jpeg' }));
    }
    if (!files.length) throw new Error('no photos staged');
    log(`resuming ${files.length} photos at iter ${st.iter}`);

    // ── rebuild the SAME trainer the run was made with (the app's
    // continueLocalRun with no card overrides: the recipe replays exactly)
    const g = st.gaussians;
    const recipe = st.recipe || recon.recipe || {};
    const { refineEvery: recipeRefine, ...recipeTrainer } = recipe;
    const cap = Math.max(g.n, recon.maxSplats || 0, recon.splats || 0);
    const fr0 = recon.frames[0] || {};
    const prevDim = Math.max(fr0.tw || 0, fr0.th || 0) || undefined;
    const prevBuf = continuedBuf(recon.trainScale, manifest.settings, prevDim);
    const prevInputDim = continuedInputDim(prevDim, prevBuf);
    const bufScale = prevBuf !== 1 ? prevBuf : undefined;
    ses = createSession({
      maxIters: manifest.maxIters, holdout: -1,
      gpuUsage: (manifest.gpu ?? 100) / 100,
      ...(recipeRefine != null ? { refineEvery: recipeRefine } : {}),
      frames: { trainMaxDim: prevInputDim, ...(bufScale != null ? { trainScale: bufScale } : {}) },
      trainer: {
        ...recipeTrainer,
        maxSplats: cap, shDeg: recipeTrainer.shDeg ?? 3,
        capMult: Math.max(2, Math.ceil(cap / Math.max(1, g.n))),
        ...(st.engine === 'v2' ? { engine: 'v2', refineEvery: 200 } : {}),
      },
      trainWorker: false,   // headless trains in-page: no UI thread to spare
    });
    ses.on('event', onEvent);
    await ses.load(files);
    ses.useReconstruction({
      known: !!recon.knownCameras,
      cams: recon.cams.map((c) => ({ imgIdx: c.imgIdx, R: c.R, t: c.t, f: c.f, cx: c.cx, cy: c.cy })),
      points: [],
      k1: recon.k1 || 0, k2: recon.k2 || 0,
      ...(recon.fFeat ? { fFeat: recon.fFeat } : {}),
    });
    undistortFrames(ses.frames, ses.recon);
    await ses.seedFrom(g, { iter: st.iter, sceneRadius: recon.sceneRadius || undefined });
    // the monitor camera lives on the view but renders to nothing until a
    // snapshot asks: capture() rasterizes canvas-free from this camera
    snapCam = H.snapshotCam(recon);
    if (snapCam) ses.view.setCamera(snapCam);
    else log('no usable monitor camera — snapshots off');
    ipsIter = st.iter; ipsAt = Date.now();
    const wasPaused = (await fetchJson('status.json').catch(() => null) || {}).status === 'paused';
    postStatus({ status: wasPaused ? 'paused' : 'active' });
    postProgress(true);
    if (!wasPaused) ses.start();
    else log('resumed paused (the run was paused when swiped away)');
    setInterval(tick, 2500);
  } catch (e) {
    await fail(e);
  }
}

function onEvent(e) {
  if (!e) return;
  if (e.kind === 'train-complete') finalize();
  else if (e.kind === 'device-lost' || e.kind === 'worker-error' || e.kind === 'train-error') {
    fail(`headless run faulted: ${e.message || e.kind} (last checkpoint kept)`);
  }
}

function postProgress(force = false) {
  if (!bridge || !ses || !ses.trainer || ending) return;
  const now = Date.now();
  const iter = ses.trainer.iter | 0;
  if (!force && now - lastProgAt < 2500 && iter === lastProgIter) return;
  lastProgAt = now; lastProgIter = iter;
  if (now - ipsAt >= 5000) {
    ips = (iter - ipsIter) * 1000 / Math.max(1, now - ipsAt);
    ipsAt = now; ipsIter = iter;
  }
  const tail = ses.lossHistory && ses.lossHistory.length
    ? ses.lossHistory[ses.lossHistory.length - 1][1] : null;
  if (Number.isFinite(tail)) psnr = tail;
  try {
    bridge.postProgress(runId, JSON.stringify({
      title: `Training ${manifest ? manifest.name : 'model'}`,
      text: `${psnr != null ? `${psnr.toFixed(1)} dB · ` : ''}${iter.toLocaleString('en-US')} / ${(manifest ? manifest.maxIters : 0).toLocaleString('en-US')}`,
      done: iter, total: manifest ? manifest.maxIters : 0,
      iter, maxIters: manifest ? manifest.maxIters : 0,
      splats: ses.trainer.n | 0, psnr, ips: +ips.toFixed(1),
      training: !!ses.training, at: now,
    }));
  } catch (e) { log('progress post failed', e); }
}

async function tick() {
  if (ending || !ses || !ses.trainer) return;
  postProgress();   // progress only — file checkpoints are transition-driven (finish/stop)
}

/** One-shot file checkpoint: the raw state plus a recon that carries the
 *  display cloud and leveling forward (the rebuilt session has neither — its
 *  recon holds no points and no capture metadata). Transitions only. */
async function secureFileCheckpoint() {
  if (ckptBusy || !ses || !ses.trainer) return false;
  const iter = ses.trainer.iter | 0;
  ckptBusy = true;
  try {
    const state = await packState(ses);
    const chunks = b64(state);
    for (let i = 0; i < chunks.length; i++) {
      if (!bridge.writeRunChunk(runId, 'state.bin', chunks[i], i === chunks.length - 1)) return false;
    }
    const S = {
      session: ses, preset: { name: manifest.name }, loadedFiles: filesWithUrls(),
      minutes: 0,   // headless wall-clock is unmeasured in V1 (adopt carries iter/score, not minutes)
      psnrTrain: psnr, psnrHold: null, holdHist: [], _localRun: { recon },
    };
    const next = buildReconJson(S);
    next.cloud = recon.cloud || next.cloud;   // the checkpoint's display cloud survives
    bridge.writeRunJson(runId, 'recon.json', JSON.stringify(next));
    recon = next;
    log(`checkpoint at iter ${iter} (${(state.length / 1048576).toFixed(1)} MB)`);
    return true;
  } catch (e) {
    log('checkpoint failed', e);   // best-effort: never disturb the run
    return false;
  } finally { ckptBusy = false; }
}

const filesWithUrls = () => files.map((f) => Object.assign(f, { url: H.photoUrl(runId, f.name) }));

/** The finished run is secured exactly like the app's finish(): a raw state
 *  checkpoint plus recon, a snapshot for the tile, and the finished status —
 *  then the service posts the summary and stands down. */
async function finalize() {
  if (ending) return;
  ending = true;
  try {
    try { ses.pause(); } catch { /* already down */ }
    const iter = ses && ses.trainer ? ses.trainer.iter | 0 : 0;
    const splats = ses && ses.trainer ? ses.trainer.n | 0 : 0;
    await secureFileCheckpoint();
    await snapshot();
    postStatus({ status: 'finished', iter, splats, psnr });
    postProgress(true);
    bridge.finishRun(runId, 'Training complete',
      `${psnr != null ? `${psnr.toFixed(1)} dB · ` : ''}${splats.toLocaleString('en-US')} splats · ${iter.toLocaleString('en-US')} cycles`);
    log(`finished at iter ${iter}`);
  } catch (e) {
    await fail(e);
  }
}

async function snapshot() {
  if (shotBusy || !bridge || !ses || !ses.trainer || !snapCam) return 'noshot';
  shotBusy = true;
  try {
    const shot = await ses.view.capture();
    if (!shot || !shot.rgba || !shot.w || !shot.h) return 'empty';
    const rgba = shot.rgba instanceof Uint8Array ? shot.rgba : new Uint8Array(shot.rgba);
    const chunks = b64(rgba);
    for (let i = 0; i < chunks.length; i++) {
      if (!bridge.writeSnapshotChunk(runId, shot.w | 0, shot.h | 0, chunks[i], i === chunks.length - 1)) {
        return 'refused';
      }
    }
    return 'ok';
  } catch (e) {
    log('snapshot failed', e);
    return 'failed';
  } finally { shotBusy = false; }
}

// ── the service's controls (evaluateJavascript; every call is guarded at the
// call site, so a command racing boot is a no-op, never a crash) ──
globalThis.__headlessToggle = () => {
  if (!ses || ending) return;
  if (ses.training) ses.pause(); else ses.start();
  postStatus({ status: ses.training ? 'active' : 'paused' });
  postProgress(true);
};
globalThis.__headlessPause = () => {
  if (!ses || ending || !ses.training) return;
  ses.pause();
  postStatus({ status: 'paused' });
  postProgress(true);
};
globalThis.__headlessResume = () => {
  if (!ses || ending || ses.training) return;
  ses.start();
  postStatus({ status: 'active' });
  postProgress(true);
};
/** Stop here and hand back: the state is secured as interrupted, and the
 *  reopened app offers Continue-here from it. */
globalThis.__headlessStop = async () => {
  if (!ses || ending) return;
  ending = true;
  try {
    try { ses.pause(); } catch { /* already down */ }
    await secureFileCheckpoint();
    await snapshot();
    const iter = ses.trainer ? ses.trainer.iter | 0 : 0;
    postStatus({ status: 'interrupted', iter, stopped: true });
    bridge.stopRun(runId, 'Training stopped',
      `State kept at cycle ${iter.toLocaleString('en-US')} — continue any time.`);
  } catch (e) {
    await fail(e);
  }
};
globalThis.__headlessSnapshot = () => snapshot();

boot();
