import { parseCameras, frameKey } from '../../src/io/cameras.js';
import { readZip } from './zip.js';

const finite = (values, length) => Array.isArray(values) && values.length === length && values.every(Number.isFinite);
const validPoint = (p) => finite(p?.X, 3) && finite(p?.rgb, 3) && p.rgb.every((v) => v >= 0 && v <= 1);

export function validateDataset(dataset) {
  if (!dataset?.files || dataset.files.length < 3) throw new Error('Use at least 3 overlapping photos; 20–200 views are recommended.');
  const names = dataset.files.map((file) => frameKey(file.name));
  if (new Set(names).size !== names.length) throw new Error('Image basenames must be unique so every photo has exactly one camera.');
  if (!dataset.transforms) return dataset;
  let cameras;
  try { cameras = parseCameras(dataset.transforms); }
  catch { throw new Error('Invalid camera file. Use a NeRF transforms.json or splat.js cameras.json.'); }
  const byName = new Map(cameras.map((c) => [frameKey(c.name), c]));
  if (byName.size !== cameras.length) throw new Error('Duplicate camera basenames in the camera file.');
  const definitions = dataset.transforms.frames || dataset.transforms.cameras;
  for (const fr of definitions) {
    if (!names.includes(frameKey(fr.file_path || fr.image))) continue;
    const skew = fr.skew ?? dataset.transforms.skew ?? 0;
    const projectionSkew = fr.projection_matrix?.[4] ?? 0;
    if (!Number.isFinite(skew) || !Number.isFinite(projectionSkew) || Math.abs(skew) > 1e-4 || Math.abs(projectionSkew) > 1e-7) {
      throw new Error('This known-camera input has nonzero projection skew, which splat.js cannot render accurately. Use a zero-skew capture/camera calibration or standard photos instead.');
    }
  }
  for (const name of names) {
    const c = byName.get(name);
    if (!c) throw new Error(`No known camera for ${name}. Select its matching image and camera JSON together.`);
    if (!finite(c.R, 9) || !finite(c.t, 3)) throw new Error(`Invalid pose for ${name}; matrix values must be finite.`);
    const focal = c.fx ?? c.angleX;
    if (!Number.isFinite(focal) || focal <= 0) throw new Error(`Missing or invalid camera focal length for ${name}.`);
    if (c.fy != null && (!Number.isFinite(c.fy) || c.fy <= 0)) throw new Error(`Invalid vertical focal length for ${name}.`);
    for (const key of ['w', 'h']) if (c[key] != null && (!Number.isFinite(c[key]) || c[key] <= 0)) throw new Error(`Invalid camera ${key} for ${name}.`);
    for (const key of ['cx', 'cy']) if (c[key] != null && !Number.isFinite(c[key])) throw new Error(`Invalid principal point for ${name}.`);
  }
  if (dataset.points && (!Array.isArray(dataset.points) || dataset.points.some((p) => !validPoint(p)))) {
    throw new Error('Seed points must have finite X coordinates and RGB values between 0 and 1.');
  }
  if (dataset.pointsByFrame) {
    if (typeof dataset.pointsByFrame !== 'object' || Array.isArray(dataset.pointsByFrame)) throw new Error('Per-frame seed ownership must be an object keyed by filename.');
    for (const owned of Object.values(dataset.pointsByFrame)) {
      if (!Array.isArray(owned) || owned.some((p) => !validPoint(p))) throw new Error('Per-frame seed points must have finite X coordinates and RGB values between 0 and 1.');
    }
  }
  return dataset;
}

export function filterDataset(dataset, selectedNames) {
  const names = new Set(selectedNames);
  const keys = new Set([...names].map(frameKey));
  const files = dataset.files.filter((f) => names.has(f.name));
  let transforms = dataset.transforms;
  if (transforms?.frames) transforms = { ...transforms, frames: transforms.frames.filter((f) => keys.has(frameKey(f.file_path))) };
  else if (transforms?.cameras) transforms = { ...transforms, cameras: transforms.cameras.filter((c) => keys.has(frameKey(c.image))) };
  const pointsByFrame = dataset.pointsByFrame;
  const candidates = pointsByFrame
    ? files.flatMap((f) => (pointsByFrame[f.name] || []).map((p) => ({ ...p, frameName: f.name })))
    : (dataset.points || []).filter((p) => !p.frameName || names.has(p.frameName));
  const voxels = new Set();
  const points = candidates.filter((p) => {
    const key = p.X.map((v) => Math.floor(v / .02)).join(':');
    if (voxels.has(key) || voxels.size >= 30000) return false;
    voxels.add(key); return true;
  });
  const retainedPointsByFrame = pointsByFrame ? Object.fromEntries(files.map((f) => [f.name, points.filter((p) => p.frameName === f.name)])) : null;
  const frameSources = Object.fromEntries(Object.entries(dataset.metadata?.frameSources || {}).filter(([name]) => names.has(name)));
  const sources = new Set(Object.values(frameSources).filter((source) => source !== 'none'));
  const seedSource = sources.size > 1 ? 'mixed' : ([...sources][0] || dataset.metadata?.seedSource || 'none');
  return { ...dataset, files, transforms, points, pointsByFrame: retainedPointsByFrame,
    metadata: { ...dataset.metadata, retainedFrames: files.length, pointCount: points.length, frameSources,
      pointsByFrame: retainedPointsByFrame, seedSource, randomSeedFallback: points.length < 8 } };
}

/** One-line provenance for the native export's world leveling (capture.json "leveling"). */
export function levelingNote(leveling) {
  if (!leveling || typeof leveling !== 'object') return '';
  const tilt = Number.isFinite(leveling.tiltDegrees) ? `${leveling.tiltDegrees.toFixed(1)}\u00b0` : null;
  if (leveling.method === 'arcore-horizontal-plane') {
    const plane = Number.isFinite(leveling.planeArea) ? ` (${leveling.planeArea.toFixed(2)} m\u00b2 ground plane)` : '';
    if (leveling.applied) return `World levelled to the ARCore ground plane${plane}${tilt ? `; ${tilt} tilt removed` : ''}.`;
    return `ARCore ground plane found${plane}; the session world was already level.`;
  }
  if (leveling.method === 'camera-up-average' && leveling.applied) {
    return `No ground plane in view; levelled to the cameras' average up${tilt ? ` (${tilt})` : ''}. Hold the phone upright while capturing.`;
  }
  if (leveling.method === 'none' && /horizontal/i.test(leveling.reason || '')) {
    return 'Portrait capture: the cameras\' average up lies sideways, so the gravity-aligned AR world is kept as captured.';
  }
  return 'No ground plane in view and the camera orientations disagree; the session world is kept as captured.';
}

export function measuredSeeds(dataset) {
  const points = dataset.points || [];
  return points.length >= 8 ? points : [];
}

export function cameraForFile(dataset, name) {
  if (!dataset.transforms) return null;
  const camera = parseCameras(dataset.transforms).find((c) => frameKey(c.name) === frameKey(name));
  if (!camera) return null;
  const { R, t } = camera;
  const position = [0, 1, 2].map((k) => -(R[k] * t[0] + R[k + 3] * t[1] + R[k + 6] * t[2]));
  const raw = dataset.transforms.frames?.find((f) => frameKey(f.file_path) === frameKey(name));
  return { position, R, t, fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy,
    w: camera.w, h: camera.h, transform_matrix: raw?.transform_matrix,
    timestamp: raw?.timestamp, orientation: raw?.orientation };
}

export function storedCapture(dataset) {
  return {
    kind: dataset.transforms ? 'ar' : 'photos', created: Date.now(),
    files: dataset.files.map((f) => ({ name: f.name, blob: f })),
    transforms: dataset.transforms || null, points: dataset.points || [],
    pointsByFrame: dataset.pointsByFrame || null, metadata: dataset.metadata || {},
  };
}

export async function datasetFromSelection(files, opts = {}) {
  // fromZip: a folder zip carries strays a hand pick never would (OS
  // metadata, thumbnails, second copies) — drop them with names attached
  // instead of failing the import. Hand picks stay strict.
  const fromZip = !!opts.fromZip;
  const dropped = [];
  const images = files.filter((f) => f.type.startsWith('image/'));
  const jsons = files.filter((f) => /\.json$/i.test(f.name));
  if (!jsons.length) return { kind: 'photos', files: images, points: [] };
  let transforms = null, keptCam = null, points = [], pointsByFrame = null, metadata = {};
  for (const file of jsons) {
    // a saved run zipped beside a capture: it restores through the session
    // door, never as cameras (its frames[] are viewer records, not poses)
    if (fromZip && SESSION_MARKERS.has(file.name.toLowerCase())) {
      dropped.push(`${file.name} (a saved run, not a capture — left out)`);
      continue;
    }
    const json = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
    if (Array.isArray(json.frames) || Array.isArray(json.cameras)) {
      if (transforms) {
        if (!fromZip) throw new Error('Choose only one camera JSON file with the matching photos.');
        if (file.name.toLowerCase() === 'transforms.json' && keptCam.toLowerCase() !== 'transforms.json') {
          dropped.push(`${keptCam} (a second camera file — kept transforms.json)`);
          transforms = json; keptCam = file.name;
        } else {
          dropped.push(`${file.name} (a second camera file — kept ${keptCam})`);
        }
        continue;
      }
      transforms = json; keptCam = file.name;
    } else if (Array.isArray(json.points)) { points = json.points; pointsByFrame = json.pointsByFrame || pointsByFrame; }
    else if (/^capture\.json$/i.test(file.name)) metadata = json;
  }
  if (!transforms) throw new Error('No camera definitions found. Select transforms.json or cameras.json with the matching photos.');
  let kept = images;
  if (fromZip) {
    const defs = transforms.frames || transforms.cameras;
    const camKeys = new Set(), camFull = new Set();
    for (const fr of defs) {
      const p = fr.file_path || fr.image;
      if (p) { camKeys.add(frameKey(p)); camFull.add(baseOf(p).toLowerCase()); }
    }
    const groups = new Map();
    for (const img of images) {
      if (!camKeys.has(frameKey(img.name))) { dropped.push(`${img.name} (no camera names it — left out)`); continue; }
      const k = frameKey(img.name);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(img);
    }
    kept = [];
    for (const g of groups.values()) {
      if (g.length === 1) { kept.push(g[0]); continue; }
      const win = g.find((img) => camFull.has(img.name.toLowerCase())) || g[0];
      for (const img of g) if (img !== win) dropped.push(`${img.name} (same photo as ${win.name} — kept one)`);
      kept.push(win);
    }
  }
  pointsByFrame = pointsByFrame || metadata.pointsByFrame || null;
  const dataset = validateDataset({ kind: 'ar', files: kept, transforms, points, pointsByFrame, metadata: { ...metadata, imported: true } });
  const shaped = validateDataset(filterDataset(dataset, kept.map((f) => f.name)));
  if (dropped.length) shaped.dropped = dropped;
  return shaped;
}

const IMAGE_MIME = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  webp: 'image/webp', bmp: 'image/bmp', gif: 'image/gif', avif: 'image/avif' };
const baseOf = (p) => String(p).split(/[\\/]/).pop();

/** Directory of a zip entry path (zips always use forward slashes). */
const dirOf = (p) => {
  const s = String(p);
  const i = s.lastIndexOf('/');
  return i < 0 ? '' : s.slice(0, i);
};
/** Basenames that mark a saved run (session restore), never a capture. */
const SESSION_MARKERS = new Set(['recon.json', 'state.bin', 'model.sog']);

/** OS metadata inside folder zips: macOS resource forks (`._photo.jpg`)
 *  and `__MACOSX` folders, Windows thumbnails, desktop settings. Never
 *  photos, never poses — skipped wherever they appear. */
export function isZipJunk(path) {
  const lower = String(path).toLowerCase();
  if (lower === '__macosx' || lower.startsWith('__macosx/') || lower.includes('/__macosx/')) return true;
  const base = baseOf(path);
  if (base.startsWith('._')) return true;
  return ['.ds_store', 'thumbs.db', 'desktop.ini'].includes(base.toLowerCase());
}

/** What a zip holds, by entry basenames (folders ignored — OS zips nest):
 *  photos train ('capture') even beside a stray saved run — our own
 *  session zips never hold photos, so photos-first can't misroute them;
 *  'session' restores, anything else is skipped. */
export function zipEntryKind(names) {
  const base = names.filter((n) => !isZipJunk(n)).map((n) => baseOf(n).toLowerCase());
  if (base.some((n) => IMAGE_MIME[n.split('.').pop()])) return 'capture';
  if (base.some((n) => SESSION_MARKERS.has(n))) return 'session';
  return null;
}

/** Unpack an exported capture zip (photos + camera JSON, flat or nested)
 *  into Files the photo flow already understands: images typed by
 *  extension, camera/seed/capture JSONs by name. OS metadata (`._` forks,
 *  `__MACOSX`, thumbnails) is skipped, second copies of a basename keep
 *  the one next to the camera file, and READMEs plus strays are dropped —
 *  basenames still keep pose↔photo matching exact. `skipped` names every
 *  entry left out so the call site can say so. Returns the raw bytes too,
 *  so session zips restore without a second read. */
export async function unpackZipFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const entries = await readZip(bytes);
  const kind = zipEntryKind([...entries.keys()]);
  if (kind !== 'capture') return { kind, files: [], bytes, skipped: [] };
  const skipped = [];
  // the camera JSON's folder is home: a second copy of a basename (another
  // capture zipped alongside) keeps the photo next to the poses
  let home = null;
  for (const name of entries.keys()) {
    const b = baseOf(name).toLowerCase();
    if (b === 'transforms.json' || b === 'cameras.json') { home = dirOf(name).toLowerCase(); break; }
  }
  const seen = new Map();
  const files = [];
  const pushFile = (name, data) => {
    const base = baseOf(name);
    const ext = base.split('.').pop().toLowerCase();
    if (IMAGE_MIME[ext]) files.push(new File([data], base, { type: IMAGE_MIME[ext] }));
    else if (ext === 'json') files.push(new File([data], base, { type: 'application/json' }));
  };
  for (const [name, data] of entries) {
    if (isZipJunk(name)) { skipped.push(`${baseOf(name)} (OS metadata — left out)`); continue; }
    const base = baseOf(name);
    const ext = base.split('.').pop().toLowerCase();
    if (!IMAGE_MIME[ext] && ext !== 'json') continue;
    const key = base.toLowerCase();
    if (!seen.has(key)) { seen.set(key, name); pushFile(name, data); continue; }
    const prev = seen.get(key);
    if (home != null && dirOf(prev).toLowerCase() !== home && dirOf(name).toLowerCase() === home) {
      seen.set(key, name);
      skipped.push(`${prev} (same file as ${name} — kept the one next to the camera file)`);
      files.splice(files.findIndex((f) => f.name.toLowerCase() === key), 1);
      pushFile(name, data);
    } else {
      skipped.push(`${name} (same file as ${prev} — kept one)`);
    }
  }
  return { kind, files, bytes, skipped };
}
