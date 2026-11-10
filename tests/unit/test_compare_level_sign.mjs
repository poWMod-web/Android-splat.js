import assert from 'node:assert/strict';
import {
  rollAngleToUp, snapRollAngle, rerollPose, exifIntrinsics,
} from '../../app/js/viewport.js';

// develop.js only touches the DOM to allocate its error-map canvas, which
// this test never invokes — a bare stub is enough to drive the real
// setBitmap roll decision.
globalThis.document = { createElement: () => ({}) };
const { Developer } = await import('../../app/js/develop.js');

// canvas rotate(), canvas-clockwise-positive, y-down — mirrors drawPhoto
const rot = (x, y, a) => {
  const t = (a * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  return [x * c - y * s, x * s + y * c];
};
const project = (R, t, f, cx, cy, X) => {
  const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [f * x / z + cx, f * y / z + cy];
};

const UP = [0, 1, 0];
const w = 640, h = 480, f = 500, cx = w / 2, cy = h / 2;   // centered, like native cams
const PTS = [[0, 1, 5], [2, -1, 4], [-3, 0.5, 6], [0.5, 0.5, 3], [-1, -2, 7]];

// End to end: a 3D point projects through the sensor pose, turns with the
// photograph's canvas rotation, and must land exactly where the leveled
// lock re-projects it. A sign flip anywhere in the chain misses by ~the
// frame size, so 1e-6 is a decisive tolerance on a 640px frame.
for (const [name, R] of [
  ['portrait (−90)', [0, -1, 0, 1, 0, 0, 0, 0, 1]],
  ['portrait (+90)', [0, 1, 0, -1, 0, 0, 0, 0, 1]],
  ['upside-down (180)', [1, 0, 0, 0, 1, 0, 0, 0, 1]],
]) {
  const snap = snapRollAngle(rollAngleToUp(R, UP));
  assert.ok(snap !== null && snap !== 0, `${name}: expected a nonzero snap`);

  // the photograph's real roll decision: sensor-framed pixels (bitmap and
  // sensor agree on landscape) turn against the snap — rerollPose turns the
  // camera's rows by +snap, so the image content turns by −snap
  const dev = new Developer();
  dev.setBitmap({ width: w, height: h }, `u:${name}`, { snap, camW: w, camH: h });
  assert.ok(dev.roll, `${name}: sensor-framed pixels should carry a roll`);
  assert.equal(dev.roll.angle, -snap, `${name}: photo canvas angle should oppose the snap`);

  // the lock's real pose: rerolled rows + EXIF intrinsics for the content turn
  const lv = rerollPose(R, [0, 0, 0], snap);
  const ex = exifIntrinsics(w, h, cx, cy, -snap);
  assert.deepEqual([ex.w, ex.h], [dev.roll.W, dev.roll.H],
    `${name}: lock frame and photo frame should agree`);

  for (const X of PTS) {
    const p = project(R, [0, 0, 0], f, cx, cy, X);
    const [vx, vy] = [p[0] - w / 2, p[1] - h / 2];
    const [rx, ry] = rot(vx, vy, dev.roll.angle);
    const photo = [rx + ex.w / 2, ry + ex.h / 2];
    const lock = project(lv.R, lv.t, f, ex.cx, ex.cy, X);
    assert.ok(Math.abs(photo[0] - lock[0]) < 1e-6 && Math.abs(photo[1] - lock[1]) < 1e-6,
      `${name}: photo [${photo}] should register with lock [${lock}] for [${X}]`);
  }
  // and the registered view is level: the point above the camera renders up
  const upPhoto = project(lv.R, lv.t, f, ex.cx, ex.cy, PTS[0]);
  assert.ok(upPhoto[1] < ex.h / 2, `${name}: gravity should render image-up`);
}

// a level frame keeps the raw lock and no roll
{
  const dev = new Developer();
  dev.setBitmap({ width: w, height: h }, 'u:level',
    { snap: snapRollAngle(rollAngleToUp([1, 0, 0, 0, -1, 0, 0, 0, -1], UP)), camW: w, camH: h });
  assert.equal(dev.roll, null, 'a level frame should leave the photograph unturned');
}

// EXIF-displayed pixels (portrait bitmap against landscape sensor) skip the
// turn, but still report the lock's frame so the error map can turn its
// sensor-framed snapshot underneath them
{
  const dev = new Developer();
  dev.setBitmap({ width: h, height: w }, 'u:swapped', { snap: -90, camW: w, camH: h });
  assert.ok(dev.roll, 'swapped pixels should still report the lock frame');
  assert.equal(dev.roll.angle, 0, 'swapped pixels take no turn');
  assert.deepEqual([dev.roll.W, dev.roll.H], [h, w], 'the frame follows the snap');
}

// exifIntrinsics itself: the branches must equal a canvas rotation of the
// principal point by the passed angle (off-centre, so every branch moves)
for (const a of [90, -90, 180]) {
  const W = 4000, H = 3000, CX = 2100, CY = 1400;
  const e = exifIntrinsics(W, H, CX, CY, a);
  const swap = Math.abs(a) % 180 !== 0;
  assert.deepEqual([e.w, e.h], swap ? [H, W] : [W, H]);
  const [rx, ry] = rot(CX - W / 2, CY - H / 2, a);
  assert.ok(Math.abs(e.cx - (rx + e.w / 2)) < 1e-9 && Math.abs(e.cy - (ry + e.h / 2)) < 1e-9,
    `exifIntrinsics(${a}) should rotate the principal point by ${a}`);
}

console.log('compare photo/lock registration sign: passed');
