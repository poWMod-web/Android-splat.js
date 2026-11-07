import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');
const drawFrame = source.match(/        void onDrawFrame\(\) \{([\s\S]*?)\n        }\n\n        private void drawBackground/);
const capture = source.match(/        private void capture\(Frame frame\) \{([\s\S]*?)\n        }\n\n        private void addColoredSeeds/);
assert.ok(drawFrame, 'CoverageRenderer frame loop should be present');
assert.ok(capture, 'Native keyframe capture should be present');
assert.doesNotMatch(drawFrame[1], /updateCoverage\s*\(/, 'preview frames must not add coverage observations');
assert.match(capture[1], /image\s*=\s*frame\.acquireCameraImage\(\)[\s\S]*updateCoverage\(frame\)/,
  'coverage should be sampled only as part of successfully acquired keyframe processing');
assert.equal((source.match(/updateCoverage\(frame\)/g) || []).length, 1,
  'the keyframe path should be the sole caller that updates the accumulated coverage cloud');
console.log('coverage cloud updates only on keyframe capture: passed');
