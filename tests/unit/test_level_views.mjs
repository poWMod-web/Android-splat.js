import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  rotateAboutAxis, rollAngleToUp, snapRollAngle, rerollPose, exifIntrinsics,
} from '../../app/js/viewport.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const developSrc = readFileSync(new URL('../../app/js/develop.js', import.meta.url), 'utf8');

const near = (a, b, tol = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < tol);

// rollAngleToUp: the signed sensor-to-gravity roll about the look direction
{
  // identity camera (looking +Z, sensor-up -Y): gravity +Y is a half-turn away
  assert.ok(Math.abs(rollAngleToUp([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0]) - 180) < 1e-9);
  // portrait roll (the camera turned +90 about its axis): levels back by -90
  assert.ok(Math.abs(rollAngleToUp([0, -1, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0]) + 90) < 1e-9);
  // already level (image-up +Y): no roll
  assert.ok(Math.abs(rollAngleToUp([1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 1, 0])) < 1e-9);
  // staring straight down: no level roll exists
  assert.equal(rollAngleToUp([1, 0, 0, 0, 0, 1, 0, -1, 0], [0, 1, 0]), null);
}

// snapRollAngle: cardinal snaps within tolerance, raw otherwise
{
  assert.equal(snapRollAngle(null), null);
  assert.equal(snapRollAngle(0), 0);
  assert.equal(snapRollAngle(7), 0);
  assert.equal(snapRollAngle(-14.9), 0);
  assert.equal(snapRollAngle(95), 90);
  assert.equal(snapRollAngle(-95), -90);
  assert.equal(snapRollAngle(170), 180);
  assert.equal(snapRollAngle(-170), 180);
  assert.equal(snapRollAngle(180), 180);
  assert.equal(snapRollAngle(50), null);
  assert.equal(snapRollAngle(-50), null);
  assert.equal(snapRollAngle(30), null);
  assert.equal(snapRollAngle(106), null);
}

// rerollPose: position and aim pinned, image-up leveled, still a rotation
{
  const R = [0, -1, 0, 1, 0, 0, 0, 0, 1];   // portrait-rolled, looking +Z
  const pos = [3, -2, 5];
  const lv = rerollPose(R, pos, -90);
  assert.ok(near([lv.R[6], lv.R[7], lv.R[8]], [0, 0, 1]),
    'the aim should survive the reroll');
  assert.ok(near([-lv.R[3], -lv.R[4], -lv.R[5]], [0, 1, 0]),
    'image-up should level to gravity');
  const C = [0, 1, 2].map((k) => -(lv.R[k] * lv.t[0] + lv.R[k + 3] * lv.t[1] + lv.R[k + 6] * lv.t[2]));
  assert.ok(near(C, pos), 'the camera centre should survive the reroll');
  const det = lv.R[0] * (lv.R[4] * lv.R[8] - lv.R[5] * lv.R[7])
    - lv.R[1] * (lv.R[3] * lv.R[8] - lv.R[5] * lv.R[6])
    + lv.R[2] * (lv.R[3] * lv.R[7] - lv.R[4] * lv.R[6]);
  assert.ok(Math.abs(det - 1) < 1e-9, 'the rerolled pose should stay a rotation');
  assert.ok(Math.abs(rollAngleToUp(lv.R, [0, 1, 0])) < 1e-9,
    'rerolling by the exact roll should level the view');
}

// rotateAboutAxis: spot-checks (the reroll above leans on it throughout)
{
  assert.ok(near(rotateAboutAxis([1, 0, 0], [0, 0, 1], 90), [0, 1, 0]));
  assert.ok(near(rotateAboutAxis([1, 0, 0], [0, 0, 1], -90), [0, -1, 0]));
  assert.ok(near(rotateAboutAxis([0, 1, 0], [0, 1, 0], 45), [0, 1, 0]));
}

// exifIntrinsics: the frame and principal point follow cardinal pixels
{
  assert.deepEqual(exifIntrinsics(4000, 3000, 2100, 1400, 90), { w: 3000, h: 4000, cx: 1600, cy: 2100 });
  assert.deepEqual(exifIntrinsics(4000, 3000, 2100, 1400, -90), { w: 3000, h: 4000, cx: 1400, cy: 1900 });
  assert.deepEqual(exifIntrinsics(4000, 3000, 2100, 1400, 180), { w: 4000, h: 3000, cx: 1900, cy: 1600 });
  assert.deepEqual(exifIntrinsics(4000, 3000, 2100, 1400, 0), { w: 4000, h: 3000, cx: 2100, cy: 1400 });
  const e90 = exifIntrinsics(4000, 3000, 2100, 1400, 90);
  assert.deepEqual(exifIntrinsics(e90.w, e90.h, e90.cx, e90.cy, 90),
    exifIntrinsics(4000, 3000, 2100, 1400, 180),
    'two quarter-turns should equal a half-turn');
}

// the wiring: the tour levels every pose, frames snap to cardinal locks,
// the photograph follows, compare switching stays immediate
{
  assert.match(app, /const snap = snapRollAngle\(rollAngleToUp\(cam\.R, vp\.up\)\);/,
    'frame views should snap the sensor roll against gravity');
  assert.match(app, /vp\.lock = \{ R: lv\.R, t: lv\.t, f: cam\.f, w: ex\.w, h: ex\.h, cx: ex\.cx, cy: ex\.cy \};/,
    'snapped frames should lock a leveled EXIF-frame pose');
  assert.match(app, /dev\.setBitmap\(b, cam\.url, roll\);/,
    'the photograph should learn the lock’s snap');
  assert.match(app, /const level = rollAngleToUp\(Rq0, vp\.up\);\s*const Rq = level == null \? Rq0 : rerollPose\(Rq0, pos, level\)\.R;/,
    'the tour should level every pose to the orbit’s up (nadir keeps raw)');
  assert.match(app, /S\.compare = v;\s*\$\('stage'\)\.dataset\.cursor = cursorFor\(v\);\s*renderControls\(\);/,
    'compare modes should keep applying the moment they are pressed');
  assert.match(developSrc, /bw !== bh && \(bw > bh\) !== \(frame\.camW > frame\.camH\)/,
    'display-framed pixels should skip the turn, sensor-framed take it');
  assert.match(developSrc, /ctx\.rotate\(\(roll\.angle \* Math\.PI\) \/ 180\)/,
    'the photograph should turn by the lock’s snap');
  assert.match(developSrc, /drawFit\(model, roll \? -roll\.snap : 0\)/,
    'the error map should counter-turn the sensor-framed snapshot');
  assert.match(developSrc, /drawFit\(this\.bmp, roll \? roll\.angle : 0\)/,
    'the error map should turn the photograph with the display');
}

console.log('gravity-leveled frame and tour views: passed');
