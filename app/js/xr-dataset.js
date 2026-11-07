import { parseCameras, frameKey } from '../../src/io/cameras.js';

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

export async function datasetFromSelection(files) {
  const images = files.filter((f) => f.type.startsWith('image/'));
  const jsons = files.filter((f) => /\.json$/i.test(f.name));
  if (!jsons.length) return { kind: 'photos', files: images, points: [] };
  let transforms = null, points = [], pointsByFrame = null, metadata = {};
  for (const file of jsons) {
    const json = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
    if (Array.isArray(json.frames) || Array.isArray(json.cameras)) {
      if (transforms) throw new Error('Choose only one camera JSON file with the matching photos.');
      transforms = json;
    } else if (Array.isArray(json.points)) { points = json.points; pointsByFrame = json.pointsByFrame || pointsByFrame; }
    else if (file.name === 'capture.json') metadata = json;
  }
  if (!transforms) throw new Error('No camera definitions found. Select transforms.json or cameras.json with the matching photos.');
  pointsByFrame = pointsByFrame || metadata.pointsByFrame || null;
  const dataset = validateDataset({ kind: 'ar', files: images, transforms, points, pointsByFrame, metadata: { ...metadata, imported: true } });
  return validateDataset(filterDataset(dataset, images.map((f) => f.name)));
}
