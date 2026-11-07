import assert from 'node:assert/strict';
import { Viewport, pointerAngleDelta } from '../../app/js/viewport.js';

assert.ok(Math.abs(pointerAngleDelta(-Math.PI + 0.1, Math.PI - 0.1) - 0.2) < 1e-10,
  'twist deltas should take the shortest path across the angle wrap');

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
const { fwd: fwd0, right: right0 } = view._basis();
const roll0 = view.roll;
const yaw0 = view.yaw;
const dist0 = view.dist;
const target0 = view.target.slice();
send('pointermove', 2, 170, 170);
const { fwd: fwd1, right: right1 } = view._basis();
assert.ok(Math.abs(view.roll - roll0) > 0.1,
  'twisting two pointers should rotate the model in the viewport');
assert.ok(Math.abs(view.yaw - yaw0) < 1e-10,
  'viewport twist should not orbit the camera around world-up');
assert.ok(fwd1.every((v, i) => Math.abs(v - fwd0[i]) < 1e-10),
  'viewport twist should leave the viewing direction unchanged');
assert.ok(right1.some((v, i) => Math.abs(v - right0[i]) > 0.1),
  'viewport twist should roll the camera image axes');
assert.notEqual(view.dist, dist0,
  'two-pointer twist should retain pinch zoom behavior');
assert.notDeepEqual(view.target, target0,
  'two-pointer twist should retain centroid pan behavior');

const pan = makeViewport();
pan.view.scene = { center: [1000, 0, 0], radius: 0.001 };
pan.view.dist = 1;
pan.view.target = [0, 0, 0];
pan.view.fpv = true; // avoid scene-pivot re-anchoring on pointerdown
pan.send('pointerdown', 3, 0, 0, true);
pan.send('pointermove', 3, 100000, 100000);
const { right, down } = pan.view._basis();
const vertical = pan.view.target.reduce((sum, v, i) => sum + v * down[i], 0);
const horizontal = pan.view.target.reduce((sum, v, i) => sum + v * right[i], 0);
assert.equal(pan.view._panY, pan.view.h,
  'vertical pan should clamp to the viewport height, not the scene bounds');
assert.ok(Math.abs(vertical) > 2 * pan.view.scene.radius,
  'vertical pan should be able to travel beyond any scene-relative limit');
assert.ok(Math.abs(horizontal) > 10,
  'centroid pan should remain unbounded left-to-right');

console.log('viewport twist, pinch, and bounded vertical pan: passed');
