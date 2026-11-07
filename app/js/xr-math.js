// xr-math.js — small, browser-independent helpers for WebXR photo capture.
//
// WebXR matrices are column-major Float32Array values.  NeRF transforms in
// this project are nested row-major arrays, while points are column vectors.
// The helpers deliberately keep those conventions explicit so a capture never
// silently changes camera handedness or mixes image origins.

const EPSILON = 1e-10;

function requireMatrix16(matrix, name = 'matrix') {
  if (!matrix || typeof matrix.length !== 'number' || matrix.length < 16) {
    throw new TypeError(`${name} must contain 16 values`);
  }
  return matrix;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

/** Convert a WebXR/WebGL column-major 4x4 matrix to nested row-major rows. */
export function matrixRows(matrix) {
  const m = requireMatrix16(matrix);
  return [
    [m[0], m[4], m[8], m[12]],
    [m[1], m[5], m[9], m[13]],
    [m[2], m[6], m[10], m[14]],
    [m[3], m[7], m[11], m[15]],
  ].map((row) => row.map(Number));
}

/** Alias whose name describes the source convention. */
export const columnMajorToRows = matrixRows;
export const matrixToRows = matrixRows;

/** Convert nested row-major rows back to a column-major 16-value matrix. */
export function rowsToColumnMajor(rows) {
  if (!Array.isArray(rows) || rows.length < 4 || rows.some((row) => !row || row.length < 4)) {
    throw new TypeError('rows must be a 4x4 nested array');
  }
  return new Float32Array([
    rows[0][0], rows[1][0], rows[2][0], rows[3][0],
    rows[0][1], rows[1][1], rows[2][1], rows[3][1],
    rows[0][2], rows[1][2], rows[2][2], rows[3][2],
    rows[0][3], rows[1][3], rows[2][3], rows[3][3],
  ]);
}

function matrix16(matrix) {
  if (matrix && matrix.matrix && typeof matrix.length !== 'number') {
    matrix = matrix.matrix;
  }
  if (Array.isArray(matrix) && matrix.length >= 4 && Array.isArray(matrix[0])) {
    return rowsToColumnMajor(matrix);
  }
  return requireMatrix16(matrix);
}

/** Transform a 3D point by a column-major matrix or nested row-major rows. */
export function transformPoint(matrix, point) {
  if (!point || point.length < 3) throw new TypeError('point must contain x, y, z');
  const m = matrix16(matrix);
  const x = Number(point[0]);
  const y = Number(point[1]);
  const z = Number(point[2]);
  const qx = m[0] * x + m[4] * y + m[8] * z + m[12];
  const qy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const qz = m[2] * x + m[6] * y + m[10] * z + m[14];
  const qw = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (Math.abs(qw) > EPSILON && Math.abs(qw - 1) > EPSILON) {
    return [qx / qw, qy / qw, qz / qw];
  }
  return [qx, qy, qz];
}

/**
 * Calculate pixel intrinsics from a WebXR projection matrix.
 *
 * WebXR's projection matrix uses OpenGL's camera convention (camera looks
 * along -Z) and its sample defines the intrinsic calculation in the camera
 * texture's own dimensions.  `origin: 'top-left'` is the image convention
 * used by this capture module after flipping readPixels rows.  The default
 * therefore flips only cy; it does not crop, mirror, or scale the image.
 *
 * The first arguments may be (projectionMatrix, width, height, options), or
 * (projectionMatrix, { width, height, x, y }, options).  x/y are normally 0
 * for a full camera texture and are retained for tests and diagnostics.
 */
export function projectionIntrinsics(projectionMatrix, widthOrViewport, heightOrOptions, maybeOptions) {
  const p = requireMatrix16(projectionMatrix, 'projectionMatrix');
  let width;
  let height;
  let viewportX = 0;
  let viewportY = 0;
  let options;

  if (widthOrViewport && typeof widthOrViewport === 'object') {
    width = Number(widthOrViewport.width);
    height = Number(widthOrViewport.height);
    viewportX = Number(widthOrViewport.x || 0);
    viewportY = Number(widthOrViewport.y || 0);
    options = heightOrOptions || {};
  } else {
    width = Number(widthOrViewport);
    height = Number(heightOrOptions);
    options = maybeOptions || {};
  }
  if (!(width > 0) || !(height > 0)) throw new RangeError('projection dimensions must be positive');

  const origin = options.origin || 'top-left';
  const fx = width * 0.5 * p[0];
  const fy = height * 0.5 * p[5];
  const skew = width * 0.5 * p[4];
  const cx = viewportX + width * 0.5 * (1 - p[8]);
  const cyBottom = viewportY + height * 0.5 * (1 - p[9]);
  const cy = origin === 'bottom-left'
    ? cyBottom
    : viewportY + height - (cyBottom - viewportY);

  return {
    fl_x: fx,
    fl_y: fy,
    fx,
    fy,
    cx,
    cy,
    skew,
    w: width,
    h: height,
    origin,
  };
}

/** Alias used by callers that describe the same operation as a conversion. */
export const projectionToIntrinsics = projectionIntrinsics;
export const getProjectionIntrinsics = projectionIntrinsics;

function depthTransformMatrix(depthInfo) {
  const transform = depthInfo && depthInfo.normDepthBufferFromNormView;
  if (!transform) return null;
  return transform.matrix || transform;
}

/**
 * Convert normalized-view coordinates to a depth-buffer sample.
 *
 * The WebXR depth spec defines normalized view coordinates with origin at the
 * top-left.  It applies normDepthBufferFromNormView, scales by depth width and
 * height, truncates, then clamps.  getDepthInMeters() performs the same
 * operation internally; this helper is useful when inspecting raw CPU data.
 */
export function normalizedViewToDepthPixel(depthInfo, x, y) {
  const width = Number(depthInfo?.width);
  const height = Number(depthInfo?.height);
  if (!(width > 0) || !(height > 0)) throw new RangeError('depth dimensions must be positive');
  if (!(x >= 0 && x <= 1) || !(y >= 0 && y <= 1)) {
    throw new RangeError('normalized view coordinates must be in [0, 1]');
  }

  const matrix = depthTransformMatrix(depthInfo);
  const q = matrix ? transformHomogeneous(matrix, [x, y, 0, 1]) : [x, y, 0, 1];
  const w = Math.abs(q[3]) > EPSILON ? q[3] : 1;
  const nx = q[0] / w;
  const ny = q[1] / w;
  const column = clamp(Math.trunc(nx * width), 0, width - 1);
  const row = clamp(Math.trunc(ny * height), 0, height - 1);
  return {
    x: column,
    y: row,
    column,
    row,
    index: row * width + column,
    normalizedX: nx,
    normalizedY: ny,
  };
}

export const depthBufferCoordinates = normalizedViewToDepthPixel;

function transformHomogeneous(matrix, point) {
  const m = matrix16(matrix);
  return [
    m[0] * point[0] + m[4] * point[1] + m[8] * point[2] + m[12] * point[3],
    m[1] * point[0] + m[5] * point[1] + m[9] * point[2] + m[13] * point[3],
    m[2] * point[0] + m[6] * point[1] + m[10] * point[2] + m[14] * point[3],
    m[3] * point[0] + m[7] * point[1] + m[11] * point[2] + m[15] * point[3],
  ];
}

/** Read depth using the spec-defined normalized-view coordinates. */
export function sampleDepthMeters(depthInfo, x, y) {
  if (!depthInfo || typeof depthInfo.getDepthInMeters !== 'function') return 0;
  try {
    const depth = Number(depthInfo.getDepthInMeters(x, y));
    return Number.isFinite(depth) && depth > 0 ? depth : 0;
  } catch {
    return 0;
  }
}

/**
 * Unproject a CPU depth sample into camera coordinates and optionally into
 * local/reference-space coordinates.
 *
 * x and y are normalized-view coordinates (top-left origin), and depthMeters
 * is the WebXR depth value: distance along the camera plane's -Z direction,
 * not the Euclidean length of the camera-to-point vector.  WebXR projection
 * matrices are perspective matrices with p[11] = -1, so the equations retain
 * asymmetric principal points and p[4] skew exactly.
 *
 * Width and height are explicit arguments to keep callers honest about using
 * the full camera image dimensions.  They are validated even though the
 * normalized projection ray itself does not need them.
 */
export function unprojectDepth(projectionMatrix, width, height, x, y, depthMeters, cameraToWorld = null) {
  const p = requireMatrix16(projectionMatrix, 'projectionMatrix');
  if (!(Number(width) > 0) || !(Number(height) > 0)) throw new RangeError('projection dimensions must be positive');
  if (!(x >= 0 && x <= 1) || !(y >= 0 && y <= 1)) {
    throw new RangeError('normalized view coordinates must be in [0, 1]');
  }
  const depth = Number(depthMeters);
  if (!(depth > 0) || !Number.isFinite(depth)) return null;

  const ndcX = x * 2 - 1;
  const ndcY = 1 - y * 2;
  const yOverZ = -(ndcY + p[9]) / p[5];
  const xOverZ = -(ndcX + p[8] + p[4] * yOverZ) / p[0];
  const cameraPoint = [xOverZ * -depth, yOverZ * -depth, -depth];
  if (!cameraToWorld) return cameraPoint;
  return transformPoint(cameraToWorld, cameraPoint);
}

/** Object-shaped version convenient for depth-grid callers. */
export function unprojectDepthSample({
  projectionMatrix,
  width,
  height,
  x,
  y,
  depthMeters,
  depth,
  cameraToWorld = null,
}) {
  return unprojectDepth(
    projectionMatrix,
    width,
    height,
    x,
    y,
    depthMeters ?? depth,
    cameraToWorld,
  );
}

/** Alias describing the output coordinate space. */
export const depthPointToWorld = unprojectDepth;
export const unprojectDepthToWorld = unprojectDepth;

/** Project a camera-space point to normalized-view coordinates (test helper). */
export function projectCameraPoint(projectionMatrix, point) {
  const q = transformHomogeneous(projectionMatrix, [point[0], point[1], point[2], 1]);
  if (Math.abs(q[3]) <= EPSILON) throw new RangeError('projected point has zero homogeneous w');
  return {
    x: (q[0] / q[3] + 1) * 0.5,
    y: (1 - q[1] / q[3]) * 0.5,
    ndcX: q[0] / q[3],
    ndcY: q[1] / q[3],
  };
}
