// train-headless.js — the foreground-service driver. Runs ONLY in the :training
// service's headless WebView (booted when the Activity backgrounds), never in the app.
//
// It resumes the backgrounded run from the exact handoff checkpoint — photos,
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
import { psnrToApproxSsimPercent } from './chart.js';

const qs = new URLSearchParams(location.search);
const runId = qs.get('run') || '';
const bridge = globalThis.AndroidHeadless || null;
const log = (...a) => console.log('[headless]', ...a);

let ses = null;
let manifest = null;
let recon = null;          // the checkpoint's recon (cloud + leveling carry over)
let snapCam = null;
let files = [];
let ending = false;        // finalize/abort in flight — the metrics beat stands down
let ckptBusy = false;
let lastProgAt = 0, lastProgIter = -1;
let ips = 0, ipsAt = 0, ipsIter = 0;
let shotBusy = false;
let psnr = null;
let splatHist = [];
let chartEvents = [];
let earlyPrune = null;       // approved in the Activity before training starts

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
    const approvedPrune = manifest.settings?.earlyPrune;
    if (approvedPrune?.approved) {
      earlyPrune = {
        iter: Number.isFinite(approvedPrune.iter) ? approvedPrune.iter : 500,
        ratio: 0.25,   // approval is specifically for removing 75%
        // The manifest marker is written before either process prunes. Iteration
        // alone is not proof: the Activity can hand off just past cycle 500
        // before its deferred prune task runs.
        done: !!approvedPrune.done,
        busy: false,
      };
    }
    recon = await fetchJson('recon.json');
    if (!recon || !Array.isArray(recon.cams) || !recon.cams.length) throw new Error('recon has no cameras');
    if (Number.isFinite(recon.iter) && recon.iter !== st.iter) {
      throw new Error(`checkpoint transaction incomplete (${st.iter} / ${recon.iter})`);
    }
    chartEvents = Array.isArray(recon.stats?.events) ? recon.stats.events.map((e) => ({ ...e })) : [];
    // photos matched BY NAME to the run's own frame order (same rule as the
    // app's continue path — re-sorting against imgIdx would mis-train)
    const staged = new Set(manifest.photos);
    files = [];
    const urlByName = new Map(manifest.photos.map((name, i) =>
      [name, manifest.photoSource === 'urls' ? manifest.photoUrls[i] : H.photoUrl(runId, name)]));
    for (const fr of recon.frames || []) {
      if (!staged.has(fr.name)) throw new Error(`photo ${fr.name} was not staged`);
      const sourceUrl = urlByName.get(fr.name);
      const r = await fetch(sourceUrl);
      if (!r.ok) throw new Error(`photo ${fr.name} unreadable (${r.status})`);
      files.push(Object.assign(new File([await r.blob()], fr.name, { type: 'image/jpeg' }), { url: sourceUrl }));
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
    // The curve is part of the model transaction, not UI decoration. Continue
    // from the Activity samples so service metrics append instead of replacing
    // everything before the handoff.
    const priorChart = (recon.stats && recon.stats.chart) || [];
    splatHist = ((recon.stats && recon.stats.splats) || []).map((r) => Array.from(r || []));
    ses.setLossHistory(priorChart);
    if (priorChart.length && Number.isFinite(+priorChart[priorChart.length - 1][1]))
      psnr = +priorChart[priorChart.length - 1][1];
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
    // progress rides the metrics beat (every readout while frames run), not a
    // timer: this page is hidden from birth, so its intervals throttle to a
    // crawl and the notification would freeze behind a healthy run
    ses.on('metrics', (m) => {
      postProgress();
      // Metrics emit from the active frame promise; prune only after it exits,
      // or pauseAndDrain would wait on the frame that called this handler.
      const iter = m && Number.isFinite(m.iter) ? m.iter : ses.trainer.iter;
      setTimeout(() => { void maybeEarlyPrune(iter); }, 0);
    });
  } catch (e) {
    await fail(e);
  }
}

function onEvent(e) {
  if (!e) return;
  if (e.kind === 'refine' && e.grown > 0)
    chartEvents.push({ iter: e.iter, kind: 'grow', label: `Capacity +${e.grown.toLocaleString('en-US')}` });
  if (e.kind === 'soft-prune')
    chartEvents.push({ iter: e.iter, kind: 'prune', label: `Pruned −${(e.culled || 0).toLocaleString('en-US')}` });
  if (e.kind === 'dead-cull')
    chartEvents.push({ iter: e.iter, kind: 'prune', label: `Pruned −${(e.culled || 0).toLocaleString('en-US')} dead` });
  if (e.kind === 'grow-now' && e.grown > 0)
    chartEvents.push({ iter: e.iter, kind: 'grow', label: `Grew +${(e.grown || 0).toLocaleString('en-US')}` });
  if (e.kind === 'prune-continue')
    chartEvents.push({ iter: e.iter, kind: 'prune', label: `Pruned ${(e.from || 0).toLocaleString('en-US')}→${(e.to || 0).toLocaleString('en-US')}` });
  if (chartEvents.length > 300) chartEvents = chartEvents.slice(-300);
  if (['refine', 'soft-prune', 'dead-cull', 'grow-now', 'prune-continue'].includes(e.kind)) postProgress(true);
  if (e.kind === 'train-complete') finalize();
  else if (e.kind === 'device-lost' || e.kind === 'worker-error' || e.kind === 'train-error') {
    fail(`headless run faulted: ${e.message || e.kind} (last checkpoint kept)`);
  }
}

async function maybeEarlyPrune(iter) {
  if (!earlyPrune || earlyPrune.done || earlyPrune.busy || ending ||
      iter < earlyPrune.iter || !ses?.trainer) return;
  // The Activity obtained explicit approval. Publish the one-shot marker
  // before touching the model so a process restart never removes 75% twice.
  earlyPrune.busy = true;
  earlyPrune.done = true;
  manifest.settings = {
    ...(manifest.settings || {}),
    earlyPrune: { approved: true, iter: earlyPrune.iter, ratio: 0.25, done: true, busy: false },
  };
  try {
    if (!bridge.writeRunJson(runId, 'manifest.json', JSON.stringify(manifest)))
      throw new Error('native manifest write refused');
  } catch (e) {
    // Never perform a destructive one-shot operation without first making its
    // completion marker durable across a service restart.
    earlyPrune.busy = false;
    log('early-prune marker failed; training continues unpruned', e);
    return;
  }
  postProgress(true);
  try {
    log('approved speed prune at cycle 500: removing 75%');
    await ses.pruneAndContinue(0.25);
    await secureFileCheckpoint();
    postProgress(true);
  } catch (e) {
    // pruneAndContinue restores the intact unpruned trainer on failure.
    log('approved speed prune failed; training continues unpruned', e);
  } finally {
    earlyPrune.busy = false;
  }
}

function scoreText(value) {
  if (!Number.isFinite(value)) return '';
  return manifest?.scoreScale === 'similarity'
    ? `${psnrToApproxSsimPercent(value).toFixed(1)} %`
    : `${value.toFixed(1)} dB`;
}

function postProgress(force = false) {
  if (!bridge || !ses || !ses.trainer || ending) return;
  const now = Date.now();
  const iter = ses.trainer.iter | 0;
  if (!force && now - lastProgAt < 2500 && iter === lastProgIter) return;
  if (!splatHist.length || splatHist[splatHist.length - 1][0] !== iter) {
    splatHist.push([iter, ses.trainer.n | 0]);
    if (splatHist.length > 2000) splatHist = splatHist.slice(-2000);
  }
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
      text: `${scoreText(psnr) ? `${scoreText(psnr)} · ` : ''}${ses.trainer.n.toLocaleString('en-US')} splats · ${iter.toLocaleString('en-US')} / ${(manifest ? manifest.maxIters : 0).toLocaleString('en-US')}`,
      done: iter, total: manifest ? manifest.maxIters : 0,
      iter, maxIters: manifest ? manifest.maxIters : 0,
      splats: ses.trainer.n | 0, psnr, scoreScale: manifest?.scoreScale || 'db', ips: +ips.toFixed(1),
      training: !!ses.training, gpu: Math.round((ses.opts.gpuUsage ?? 1) * 100), at: now,
      // A bounded durable tail powers the foreground monitor without creating
      // a second trainer/WebGPU device. Values are measured, never inferred.
      scoreHistory: (ses.lossHistory || []).slice(-400),
      splatHistory: splatHist.slice(-400),
      events: chartEvents.slice(-300),
    }));
  } catch (e) { log('progress post failed', e); }
}

/** One-shot file checkpoint: the raw state plus a recon that carries the
 *  display cloud and leveling forward (the rebuilt session has neither — its
 *  recon holds no points and no capture metadata). Transitions only. */
async function secureFileCheckpoint() {
  if (ckptBusy || !ses || !ses.trainer) return false;
  let resume = false;
  ckptBusy = true;
  try {
    resume = await ses.pauseAndDrain();
    const iter = ses.trainer.iter | 0;
    const state = await packState(ses);
    const chunks = b64(state);
    for (let i = 0; i < chunks.length; i++) {
      if (!bridge.writeRunChunk(runId, 'state.bin', chunks[i], i === chunks.length - 1)) return false;
    }
    const S = {
      session: ses, preset: { name: manifest.name }, loadedFiles: filesWithUrls(),
      minutes: 0,   // headless wall-clock is unmeasured in V1 (adopt carries iter/score, not minutes)
      psnrTrain: psnr, psnrHold: recon.stats?.psnrHold ?? null,
      holdHist: (recon.stats && recon.stats.holds) || [], splatHist,
      chartEvents, _localRun: { recon },
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
  } finally {
    ckptBusy = false;
    if (resume && !ending && ses && !ses.training) ses.start();
  }
}

const filesWithUrls = () => files.map((f, i) => Object.assign(f, {
  url: manifest?.photoSource === 'urls' ? manifest.photoUrls[i] : H.photoUrl(runId, f.name),
}));

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
      `${scoreText(psnr) ? `${scoreText(psnr)} · ` : ''}${splats.toLocaleString('en-US')} splats · ${iter.toLocaleString('en-US')} cycles`);
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
  const wasTraining = !!ses.training;
  ending = true;
  try {
    try { ses.pause(); } catch { /* already down */ }
    const secured = await secureFileCheckpoint();
    if (!secured) {
      // Never hand stale state back as if it were current. Keep service
      // ownership and retry on the next foreground-return progress request.
      ending = false;
      if (wasTraining) ses.start();
      postStatus({ status: ses.training ? 'active' : 'paused', handbackError: 'checkpoint failed' });
      postProgress(true);
      return;
    }
    await snapshot();
    const iter = ses.trainer ? ses.trainer.iter | 0 : 0;
    postStatus({ status: 'interrupted', iter, stopped: true });
    bridge.stopRun(runId, 'Training returned to app',
      `State kept at cycle ${iter.toLocaleString('en-US')} — resuming in the app.`);
  } catch (e) {
    await fail(e);
  }
};
let saveBusy = false;
globalThis.__headlessSave = async () => {
  if (!ses || ending || saveBusy) return;
  saveBusy = true;
  let wasTraining = false;
  try {
    wasTraining = await ses.pauseAndDrain();
    await ses.finish(); // train-complete event owns checkpoint, snapshot and final status
  } catch (e) {
    if (wasTraining && ses && !ses.training) ses.start();
    postStatus({ status: ses?.training ? 'active' : 'paused', commandError: `stop & save failed: ${e.message || e}` });
    postProgress(true);
  } finally { saveBusy = false; }
};
globalThis.__headlessSnapshot = async () => {
  if (!ses || ending) return;
  let resume = false;
  try {
    resume = await ses.pauseAndDrain();
    await snapshot();
  } finally { if (resume && !ending && ses && !ses.training) ses.start(); }
};
let manualEdit = false;
async function manualModelEdit(kind, keep = 0.75) {
  if (!ses || !ses.trainer || ending || manualEdit) return;
  manualEdit = true;
  try {
    if (kind === 'grow') await ses.growNow();
    else await ses.pruneAndContinue(Math.max(0.01, Math.min(0.99, Number(keep) || 0.75)));
    await secureFileCheckpoint();
    postProgress(true);
  } catch (e) {
    log(`manual ${kind} failed`, e);
    postStatus({ status: ses.training ? 'active' : 'paused', commandError: String(e.message || e) });
  } finally { manualEdit = false; }
}
globalThis.__headlessGpu = (percent) => {
  const v = Math.max(10, Math.min(100, Math.round(Number(percent) || 100)));
  if (ses) ses.opts.gpuUsage = v / 100;
  if (manifest) manifest.gpu = v;
  postProgress(true);
};
globalThis.__headlessGrow = () => manualModelEdit('grow');
globalThis.__headlessPrune = (keep) => manualModelEdit('prune', keep);

boot();
