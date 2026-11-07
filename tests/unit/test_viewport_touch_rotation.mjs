import assert from 'node:assert/strict';
import { Viewport, pointerAngleDelta } from '../../app/js/viewport.js';

assert.ok(Math.abs(pointerAngleDelta(-Math.PI + 0.1, Math.PI - 0.1) - 0.2) < 1e-10,
  'twist deltas should take the shortest path across the angle wrap');

const handlers = new Map();
const canvas = {
  getContext: () => ({}),
  addEventListener: (type, fn) => handlers.set(type, fn),
  setPointerCapture: () => {},
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 300, height: 200 }),
};
const view = new Viewport(canvas);
const send = (type, pointerId, clientX, clientY) => handlers.get(type)({
  pointerId, clientX, clientY, button: 0, shiftKey: false,
});

send('pointerdown', 1, 100, 100);
send('pointerdown', 2, 200, 100);
const yaw0 = view.yaw;
const dist0 = view.dist;
const target0 = view.target.slice();
send('pointermove', 2, 170, 170);
assert.ok(Math.abs(view.yaw - yaw0) > 0.1,
  'twisting a two-pointer gesture should rotate the model view');
assert.notEqual(view.dist, dist0,
  'two-pointer twist should retain pinch zoom behavior');
assert.notDeepEqual(view.target, target0,
  'two-pointer twist should retain centroid pan behavior');

console.log('two-finger twist, pinch, and pan: passed');
