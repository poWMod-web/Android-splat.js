import assert from 'node:assert/strict';
import * as viewportModule from '../../app/js/viewport.js';

const { Viewport } = viewportModule;

// The model viewer navigates exactly as upstream arrival-space/splat.js:
// pitch is bounded near the vertical (±1.45 rad) and the two-finger gesture
// is pinch-to-dolly plus centroid pan — no twist rotation.
assert.equal(viewportModule.pointerAngleDelta, undefined,
  'the twist helper should not exist once two-finger rotation is removed');

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

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
assert.equal(view.roll, undefined,
  'the orbit should carry no roll state');
assert.equal(typeof view.setOrbitAngles, 'undefined',
  'orientation should be plain yaw/pitch fields, without a free-rotation setter');
assert.equal(typeof view._rotateView, 'undefined',
  'the viewpoint should not support incremental free rotations');
send('pointerdown', 1, 100, 100);
send('pointerdown', 2, 200, 100);
const { fwd: fwd0, right: right0, down: down0 } = view._basis();
const yaw0 = view.yaw, pitch0 = view.pitch;
const dist0 = view.dist;
const target0 = view.target.slice();
send('pointermove', 2, 170, 170);
const { fwd: fwd1, right: right1, down: down1 } = view._basis();
assert.equal(view.yaw, yaw0,
  'a two-finger twist should not orbit the camera');
assert.equal(view.pitch, pitch0,
  'a two-finger twist should not tilt the camera');
assert.ok(fwd1.every((v, i) => Math.abs(v - fwd0[i]) < 1e-12),
  'a two-finger twist should leave the viewing direction unchanged');
assert.ok(right1.every((v, i) => Math.abs(v - right0[i]) < 1e-12) &&
          down1.every((v, i) => Math.abs(v - down0[i]) < 1e-12),
  'a two-finger twist should leave the camera image axes unchanged (no roll)');
assert.notEqual(view.dist, dist0,
  'two-pointer gestures should retain pinch zoom behavior');
assert.notDeepEqual(view.target, target0,
  'two-pointer gestures should retain centroid pan behavior');

const orbit = makeViewport();
const { view: orbitView, send: orbitSend } = orbit;
const orbitTarget = orbitView.target.slice();
orbitSend('pointerdown', 4, 100, 100);
orbitSend('pointermove', 4, 100, -400); // 2.5 radians up — past the pitch stop
assert.equal(orbitView.pitch, 1.45,
  'upward orbit should stop at the +1.45-radian vertical bound');
orbitSend('pointermove', 4, 100, 900); // 6.5 radians down from there
assert.equal(orbitView.pitch, -1.45,
  'downward orbit should stop at the −1.45-radian vertical bound');
assert.ok(Math.abs(dot3(orbitView._basis().fwd, [0, 0, 0]) + 0) < 2,
  'sanity: the basis stays defined at the pitch stops');
assert.deepEqual(orbitView.target, orbitTarget,
  'one-finger orbit should rotate the view instead of panning its target');
const yawBefore = orbitView.yaw;
orbitSend('pointermove', 4, 300, 900);
assert.ok(Math.abs(orbitView.yaw - yawBefore - 200 * 0.006) < 1e-10,
  'horizontal orbit should stay unbounded while pitch is clamped');

const fpv = makeViewport();
fpv.view.fpv = true;
fpv.send('pointerdown', 5, 100, 100);
fpv.send('pointermove', 5, 100, -2000);
assert.equal(fpv.view.pitch, -1.45,
  'first-person look should respect the same vertical bound');
fpv.send('pointermove', 5, 100, 2000);
assert.equal(fpv.view.pitch, 1.45,
  'first-person look should respect the same vertical bound upward');

const sync = makeViewport().view;
sync.setScene({ xyz: new Float32Array([0, 0, 0]), rgb: new Uint8Array([0, 0, 0]), center: [0, 0, 0], radius: 1 });
sync.syncTo({ R: [1, 0, 0, 0, 0, -1, 0, 1, 0], t: [0, 0, -5], f: 500, w: 100, h: 100, cx: 50, cy: 50 });
assert.ok(sync.pitch <= 1.45 && sync.pitch >= -1.45,
  'snapping to a steep photograph should land within the vertical bounds');

const pan = makeViewport();
pan.view.scene = { center: [1000, 0, 0], radius: 0.001 };
pan.view.dist = 1;
pan.view.target = [0, 0, 0];
pan.view.fpv = true; // avoid scene-pivot re-anchoring on pointerdown
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

console.log('viewport bounded pitch, pinch/pan-only touch, and unbounded pan: passed');
