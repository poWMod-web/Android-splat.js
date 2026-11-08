import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildDisplayCloud } from '../../app/js/scene-cloud.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const savedCloud = { xyz: [1, 2, 3, 4, 5, 6], rgb: [255, 128, 0, 20, 40, 60] };

const restored = buildDisplayCloud([], savedCloud, []);
assert.deepEqual(Array.from(restored.xyz), savedCloud.xyz,
  'view-only and continued local runs should restore the serialized cloud when recon.points is empty');
assert.deepEqual(Array.from(restored.rgb), savedCloud.rgb,
  'restored point colors should match the immutable saved cloud');

const staleReconPoints = [{ X: [9, 8, 7], rgb: [1, 0, 0] }];
const savedWins = buildDisplayCloud([], savedCloud, staleReconPoints);
assert.deepEqual(Array.from(savedWins.xyz), savedCloud.xyz,
  'the capture-time cloud should win over reconstruction points mutated during training');

const capturedWins = buildDisplayCloud([{ X: [7, 8, 9], rgb: [0.25, 0.5, 1] }], savedCloud, []);
assert.deepEqual(Array.from(capturedWins.xyz), [7, 8, 9],
  'full-resolution captured seed points should remain preferred when available');
assert.deepEqual(Array.from(capturedWins.rgb), [63, 127, 255]);

const reconFallback = buildDisplayCloud([], null, staleReconPoints);
assert.deepEqual(Array.from(reconFallback.xyz), [9, 8, 7],
  'new sessions should keep using reconstruction points when no saved cloud exists');

assert.match(app, /const savedCloud = S\._localRun\?\.recon\?\.cloud \|\| S\.restored\?\.reconJson\?\.cloud/,
  'scene construction should source the serialized cloud from a restored local run');
assert.match(app, /buildDisplayCloud\(capturedPoints, savedCloud, \(recon && recon\.points\) \|\| \[\]\)/,
  'scene construction should use the shared point-cloud restoration helper');

console.log('Saved camera-view point cloud restoration: passed');
