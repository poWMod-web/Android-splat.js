import assert from 'node:assert/strict';
import {
  matrixRows,
  normalizedViewToDepthPixel,
  projectionIntrinsics,
  projectCameraPoint,
  rowsToColumnMajor,
  transformPoint,
  unprojectDepth,
} from '../../app/js/xr-math.js';

function approx(actual, expected, epsilon = 1e-5) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} ≉ ${expected}`);
}

function approxArray(actual, expected, epsilon = 1e-6) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => approx(value, expected[index], epsilon));
}

// WebXR column-major matrices become NeRF's nested row-major matrices.
{
  const columnMajor = new Float32Array([
     1,  2,  3,  0,
     4,  5,  6,  0,
     7,  8,  9,  0,
    10, 11, 12,  1,
  ]);
  assert.deepEqual(matrixRows(columnMajor), [
    [1, 4, 7, 10],
    [2, 5, 8, 11],
    [3, 6, 9, 12],
    [0, 0, 0, 1],
  ]);
  assert.deepEqual(Array.from(rowsToColumnMajor(matrixRows(columnMajor))), Array.from(columnMajor));
  approxArray(transformPoint(columnMajor, [1, 2, 3]), [40, 47, 54]);
}

// Portrait dimensions and an asymmetric projection must use camera texture
// dimensions, not the XRWebGLLayer viewport.  Values below encode the desired
// full-resolution intrinsics, including non-zero skew.
{
  const width = 400;
  const height = 800;
  const desired = { fx: 300, fy: 500, cx: 123, cy: 456, skew: 7 };
  const projection = new Float32Array([
    2 * desired.fx / width, 0, 0, 0,
    2 * desired.skew / width, 2 * desired.fy / height, 0, 0,
    1 - 2 * desired.cx / width, 2 * desired.cy / height - 1, -1, -1,
    0, 0, -0.2, 0,
  ]);
  const intrinsics = projectionIntrinsics(projection, width, height);
  approx(intrinsics.fl_x, desired.fx);
  approx(intrinsics.fl_y, desired.fy);
  approx(intrinsics.cx, desired.cx);
  approx(intrinsics.cy, desired.cy);
  approx(intrinsics.skew, desired.skew);
  assert.equal(intrinsics.w, width);
  assert.equal(intrinsics.h, height);
  const bottomLeft = projectionIntrinsics(projection, width, height, { origin: 'bottom-left' });
  approx(bottomLeft.cy, height - desired.cy);
}

// An asymmetric WebXR ray round-trips through the depth unprojection.  The
// projection's p[4], p[8], and p[9] terms are all active in this test.
{
  const width = 640;
  const height = 480;
  const projection = new Float32Array([
    1.7, 0, 0, 0,
    0.08, 2.1, 0, 0,
    0.16, -0.12, -1, -1,
    0, 0, -0.2, 0,
  ]);
  const cameraPoint = [0.32, -0.21, -2.4];
  const normalized = projectCameraPoint(projection, cameraPoint);
  const roundTrip = unprojectDepth(
    projection,
    width,
    height,
    normalized.x,
    normalized.y,
    -cameraPoint[2],
  );
  approxArray(roundTrip, cameraPoint);
  const cameraToWorld = [
    [1, 0, 0, 10],
    [0, 1, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1],
  ];
  approxArray(transformPoint(cameraToWorld, roundTrip), [10.32, 19.79, 27.6]);
  approxArray(unprojectDepth(
    projection,
    width,
    height,
    normalized.x,
    normalized.y,
    -cameraPoint[2],
    cameraToWorld,
  ), [10.32, 19.79, 27.6]);
}

// CPU depth sampling follows the spec: normalized view coordinates are
// transformed into normalized depth-buffer coordinates, then truncated and
// clamped after scaling by width/height.
{
  const depthInfo = {
    width: 4,
    height: 3,
    normDepthBufferFromNormView: {
      matrix: new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0.10, 0.20, 0, 1,
      ]),
    },
  };
  const sample = normalizedViewToDepthPixel(depthInfo, 0.25, 0.50);
  assert.equal(sample.column, 1);
  assert.equal(sample.row, 2);
  assert.equal(sample.index, 9);
  const clamped = normalizedViewToDepthPixel(depthInfo, 1, 1);
  assert.equal(clamped.column, 3);
  assert.equal(clamped.row, 2);
}

console.log('xr-math: matrix, intrinsics, depth coordinates, and unprojection passed');
