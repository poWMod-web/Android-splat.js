// headless.js — the Android foreground-service handoff, shared by the app and headless driver.
//
// Design B: the Activity trains locally (live session, live viewer) and stages
// a run directory under the app's files/runs/<runId>/. When the Activity backgrounds, the main
// process dies and the :training foreground service boots the headless driver
// (train-headless.js), which resumes from the last file checkpoint and keeps
// training. The reopened app monitors (snapshot viewer + progress), adopts the
// finish, or recovers.
//
// The files dir is the shared bus between the two processes: each WebView has
// its OWN IndexedDB (the :training profile is separate), so nothing crosses in
// storage. Commands ride startService intents, bulk bytes ride files:
//
//   runs/active.json          { runId } — a run exists that outlived / may outlive the Activity
//   runs/<runId>/manifest.json  run parameters (photos, horizon, settings)
//   runs/<runId>/photos.json    { names, complete } — photo handoff state
//   runs/<runId>/photos/<name>  the training photos, byte-identical
//   runs/<runId>/state.bin      the raw trainer state (packState format)
//   runs/<runId>/recon.json     cameras/frames/recipe/cloud (buildReconJson format)
//   runs/<runId>/progress.json  { iter, maxIters, splats, psnr, ips, training, at }
//   runs/<runId>/status.json    { status, owner, iter, at, error? }
//   runs/<runId>/service.json   { alive, at } — :training heartbeat
//   runs/<runId>/snapshot.png + snapshot.json { w, h, iter, at } — the monitor view
//
// Status values: staging -> active <-> paused -> finished | interrupted,
// plus booting while the service takes over. owner is 'activity' | 'service'.

// The asset-loader host both WebViews serve app files from (same constant as
// exportDataset's URIs in MainActivity — the two must agree).
export const ASSET_HOST = 'https://appassets.androidplatform.net';
export const RUNS_BASE = `${ASSET_HOST}/runs/`;

export const runUrl = (runId, name) => `${RUNS_BASE}${encodeURIComponent(runId)}/${name}`;
export const photoUrl = (runId, name) =>
  `${RUNS_BASE}${encodeURIComponent(runId)}/photos/${encodeURIComponent(name)}`;
export const activeUrl = () => `${RUNS_BASE}active.json`;
// asset-loader reads bypass the HTTP cache so the monitor never renders a
// stale snapshot/progress (the path handler keys off the path, not the query)
export const fresh = (url) => `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;

// Same character classes the native side (RunFiles) enforces — a run id or
// file name outside these never touches the filesystem on either side.
export function validRunId(id) {
  return typeof id === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(id);
}
export function validName(name) {
  return typeof name === 'string' && /^[A-Za-z0-9_.-]{1,120}$/.test(name) &&
    name !== '.' && name !== '..' && !name.includes('..');
}

/** The manifest the Activity stages at Start (versioned for forward tolerance). */
export function buildManifest({ runId, name, maxIters, gpu, srcKey, photos, photoSource, settings }) {
  return {
    version: 1, runId, name: name || 'model', maxIters: maxIters | 0,
    gpu: Math.min(100, Math.max(10, Math.round(gpu ?? 100))),
    srcKey: srcKey || null, photos: [...photos], photoSource,
    settings: settings ? { ...settings } : null,
    stagedAt: Date.now(),
  };
}

export function validateManifest(m) {
  if (!m || typeof m !== 'object') throw new Error('manifest is not an object');
  if (m.version !== 1) throw new Error(`unsupported manifest version ${m && m.version}`);
  if (!validRunId(m.runId)) throw new Error('manifest has a bad runId');
  if (!Number.isSafeInteger(m.maxIters) || m.maxIters < 1000) throw new Error('manifest has a bad horizon');
  if (!Array.isArray(m.photos) || m.photos.length < 3 || !m.photos.every(validName)) {
    throw new Error('manifest has a bad photo list');
  }
  if (m.photoSource !== 'files') throw new Error('manifest photoSource is not files');
  return m;
}

// Statuses under which a run is (or is becoming) live somewhere.
const LIVE = new Set(['staging', 'booting', 'active', 'paused']);
export const isLiveStatus = (s) => LIVE.has(s && s.status);
export const isOwnedBy = (s, owner) => !!s && s.owner === owner;

/** A monitor treats a live status with a heartbeat older than this as dead
 *  (service pressure-killed) and offers recovery instead of a frozen view. */
export const HEARTBEAT_STALE_MS = 30000;
export function heartbeatFresh(service, now = Date.now()) {
  return !!service && service.alive === true &&
    Number.isFinite(service.at) && now - service.at < HEARTBEAT_STALE_MS;
}

/** Whole-Blob base64 in slice-safe pieces (btoa's apply/spread ceiling). */
export function encodeB64(u8) {
  const bytes = u8 instanceof Uint8Array ? u8 : new Uint8Array(u8);
  let out = '';
  for (let o = 0; o < bytes.length; o += 8192) {
    out += String.fromCharCode.apply(null, bytes.subarray(o, o + 8192));
  }
  return btoa(out);
}

/** Fit a snapshot inside maxW, keeping the frame aspect. */
export function snapshotSize(tw, th, maxW = 512) {
  const w = Math.max(2, Math.min(maxW, Math.round(tw) || maxW));
  const h = Math.max(2, Math.round(w * (th / Math.max(1, tw)) || w * 9 / 16));
  return { w, h };
}

/** The monitor camera: the first photographer's viewpoint, rescaled to the
 *  snapshot size. Null when the recon has no usable camera (snapshots off). */
export function snapshotCam(recon) {
  const cams = (recon && recon.cams) || [];
  const frames = (recon && recon.frames) || [];
  const c0 = cams[0];
  const f0 = frames[c0 ? c0.imgIdx : -1] || frames[0] || {};
  if (!c0 || !Array.isArray(c0.R) || !Array.isArray(c0.t) || !(c0.f > 0)) return null;
  const tw = Math.max(1, f0.tw || f0.fw || 1), th = Math.max(1, f0.th || f0.fh || 1);
  const { w, h } = snapshotSize(tw, th);
  const k = w / tw;
  return { R: [...c0.R], t: [...c0.t], f: c0.f * k, cx: (c0.cx || tw / 2) * k, cy: (c0.cy || th / 2) * k, w, h };
}
