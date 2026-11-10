import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gravityUpForKnownRun, levelingUpFromReport } from '../../app/js/session_io.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const sessionIo = readFileSync(new URL('../../app/js/session_io.js', import.meta.url), 'utf8');

// Known-camera runs keep the capture world's y-up frame, but phone captures
// are portrait-held: sensor-up lies sideways to gravity, so averaging it
// (detectUp) levels the horizon 90 degrees off. The orbit levels to gravity
// in the data frame instead: the leveling rotation's middle column (R·Y).

// No rotation applied, no gravity correction.
assert.equal(levelingUpFromReport(null), null);
assert.equal(levelingUpFromReport(undefined), null);
assert.equal(levelingUpFromReport({}), null);
assert.equal(levelingUpFromReport({ applied: false, rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1] }), null);
assert.equal(levelingUpFromReport({ applied: true }), null);
assert.equal(levelingUpFromReport({ applied: true, rotation: [1, 0, 0] }), null);

// The portrait bake: R carries +X onto +Y (about +Z by +90 deg), so gravity
// (the capture world's +Y) sits at -X in the data frame.
{
  const g = levelingUpFromReport({ applied: true, rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1] });
  assert.deepEqual(g.map((v) => +v.toFixed(6)), [-1, 0, 0]);
}
// A small tilt stays near-vertical: 5 deg about Z puts gravity at R·Y.
{
  const t = 5 * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const g = levelingUpFromReport({ applied: true, rotation: [c, -s, 0, s, c, 0, 0, 0, 1] });
  assert.ok(Math.abs(g[0] + s) < 1e-9 && Math.abs(g[1] - c) < 1e-9 && Math.abs(g[2]) < 1e-9,
    'a small leveling tilt should keep gravity near-vertical');
}

// Precedence: the persisted vector wins, then the capture report, then +Y.
assert.deepEqual(gravityUpForKnownRun({ persisted: [0, 0, 1] }), [0, 0, 1]);
assert.deepEqual(gravityUpForKnownRun({ persisted: [0, 0, 2] }), [0, 0, 1]);
assert.deepEqual(gravityUpForKnownRun({}), [0, 1, 0]);
assert.deepEqual(gravityUpForKnownRun({ leveling: { applied: false } }), [0, 1, 0]);
assert.deepEqual(gravityUpForKnownRun({ persisted: [0, 0, 0] }), [0, 1, 0]);
assert.deepEqual(gravityUpForKnownRun({ persisted: [1, 2] }), [0, 1, 0]);
{
  const g = gravityUpForKnownRun({ leveling: { applied: true, rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1] } });
  assert.deepEqual(g.map((v) => +v.toFixed(6)), [-1, 0, 0]);
}
{
  const g = gravityUpForKnownRun({
    persisted: [0, 1, 0],
    leveling: { applied: true, rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1] },
  });
  assert.deepEqual(g, [0, 1, 0]);
}

// The wiring: the recon JSON persists the vector, both orbit setups use it
// for known-camera runs, and the known-camera flag survives restore and
// continuation (solver runs keep detectUp untouched).
assert.match(sessionIo, /\.\.\.\(recon\.known \? \{ levelingUp: r6\(gravityUpForKnownRun\(\{/,
  'the recon JSON should persist gravity in the data frame for known-camera runs only');
assert.match(app, /if \(recon && recon\.known\) \{[\s\S]*?vp\.setUp\(gravityUpForKnownRun\(/,
  'live known-camera views should level to gravity, not sensor-up');
assert.match(app, /if \(reconJson && reconJson\.knownCameras\) vp\.setUp\(gravityUpForKnownRun\(\{ persisted: reconJson\.levelingUp \}\)\);/,
  'restored known-camera views should level to the persisted gravity');
assert.match(app, /else if \(cams\.length\) vp\.detectUp\(cams\);/,
  'solver restores should keep measuring up from the cameras');
assert.match(app, /ses\.recon = \{\s*known: !!reconJson\.knownCameras,/,
  'a lite restore should keep the known-camera frame flag');
assert.match(app, /ses\.useReconstruction\(\{\s*known: !!reconJson\.knownCameras,/,
  'a trainer restore should keep the known-camera frame flag');
assert.match(app, /ses\.useReconstruction\(\{\s*known: !!recon\.knownCameras,/,
  'a continuation should keep the known-camera frame flag');

console.log('gravity up for known-camera views: passed');
