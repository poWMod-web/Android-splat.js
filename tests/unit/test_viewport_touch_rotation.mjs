import assert from 'node:assert/strict';
import { Viewport, pointerAngleDelta } from '../../app/js/viewport.js';

assert.ok(Math.abs(pointerAngleDelta(-Math.PI + 0.1, Math.PI - 0.1) - 0.2) < 1e-10,
  'twist deltas should take the shortest path across the angle wrap');

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function rotate3(v, axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const cross = [
    axis[1] * v[2] - axis[2] * v[1],
    axis[2] * v[0] - axis[0] * v[2],
    axis[0] * v[1] - axis[1] * v[0],
  ];
  const projection = dot3(axis, v) * (1 - c);
  return v.map((x, i) => x * c + cross[i] * s + axis[i] * projection);
}

function makeViewport() {
  const handlers = new Map();
  const canvas = {
    getContext: () => ({}),
    addEventListener: (type, fn) => handlers.set(type, fn),
    setPointerCapture: () => {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 300, height: 200 }),
  };
  const view = new Viewport(canvas);
  view.resize();
  const send = (type, pointerId, clientX, clientY, shiftKey = false) => handlers.get(type)({
    pointerId, clientX, clientY, button: 0, shiftKey,
  });
  return { view, send };
}

const { view, send } = makeViewport();
send('pointerdown', 1, 100, 100);
send('pointerdown', 2, 200, 100);
const { fwd: fwd0, right: right0, down: down0 } = view._basis();
const yaw0 = view.yaw;
const dist0 = view.dist;
const target0 = view.target.slice();
send('pointermove', 2, 170, 170);
const { fwd: fwd1, right: right1 } = view._basis();
const screenRotation = Math.atan2(dot3(right1, down0), dot3(right1, right0));
assert.ok(screenRotation < -0.7 && screenRotation > -0.9,
  'a clockwise two-finger twist should rotate the model clockwise on screen');
assert.ok(Math.abs(view.yaw - yaw0) < 1e-10,
  'viewport twist should not orbit the camera around world-up');
assert.ok(fwd1.every((v, i) => Math.abs(v - fwd0[i]) < 1e-10),
  'viewport twist should leave the viewing direction unchanged');
assert.ok(right1.some((v, i) => Math.abs(v - right0[i]) > 0.1),
  'viewport twist should rotate the camera image axes');
assert.notEqual(view.dist, dist0,
  'two-pointer twist should retain pinch zoom behavior');
assert.notDeepEqual(view.target, target0,
  'two-pointer twist should retain centroid pan behavior');

const orbit = makeViewport();
const { view: orbitView, send: orbitSend } = orbit;
const orbitStart = orbitView._basis();
const orbitTarget = orbitView.target.slice();
orbitSend('pointerdown', 4, 100, 100);
orbitSend('pointermove', 4, 100, -400); // 2.5 radians, past the old 1.45-radian pitch stop
const orbitAfter = orbitView._basis();
const expectedFwd = rotate3(orbitStart.fwd, orbitStart.right.map((v) => -v), -2.5);
assert.ok(orbitAfter.fwd.every((v, i) => Math.abs(v - expectedFwd[i]) < 1e-10),
  'one-finger orbit should continue through the vertical pole instead of clamping pitch');
assert.ok(dot3(orbitAfter.fwd, orbitStart.fwd) < 0,
  'vertical orbit should pass beyond 90 degrees from its starting view');
assert.deepEqual(orbitView.target, orbitTarget,
  'one-finger orbit should rotate the view instead of panning its target');
assert.ok(Math.abs(dot3(orbitAfter.fwd, orbitAfter.right)) < 1e-10 &&
          Math.abs(dot3(orbitAfter.fwd, orbitAfter.down)) < 1e-10 &&
          Math.abs(dot3(orbitAfter.right, orbitAfter.down)) < 1e-10,
  'viewpoint-relative orbit should preserve an orthogonal camera frame');
const pole = makeViewport().view;
pole.setOrbitAngles(0.4, Math.PI / 2);
const poleBasis = pole._basis();
assert.ok(Math.abs(Math.hypot(...poleBasis.right) - 1) < 1e-10 &&
          Math.abs(Math.hypot(...poleBasis.down) - 1) < 1e-10,
  'the viewpoint orientation should remain defined at the vertical pole');

const pan = makeViewport();
pan.view.scene = { center: [1000, 0, 0], radius: 0.001 };
pan.view.dist = 1;
pan.view.target = [0, 0, 0];
pan.view.fpv = true; // avoid scene-pivot re-anchoring on pointerdown
pan.view.roll = 0.35;
const panBasis = pan.view._basis();
const panStart = pan.view.target.slice();
pan.send('pointerdown', 3, 0, 0, true);
pan.send('pointermove', 3, 100000, -100000);
const panDelta = pan.view.target.map((v, i) => v - panStart[i]);
const vertical = panDelta.reduce((sum, v, i) => sum + v * panBasis.down[i], 0);
const horizontal = panDelta.reduce((sum, v, i) => sum + v * panBasis.right[i], 0);
assert.ok(Math.abs(vertical) > 100,
  'vertical pan should remain unbounded even for a tiny scene');
assert.ok(Math.abs(horizontal) > 100,
  'horizontal pan should remain unbounded');
assert.equal(pan.view._panY, undefined,
  'panning should not track or clamp to a vertical travel limit');

console.log('viewport orbit, clockwise twist, pinch, and unbounded viewport-relative pan: passed');
