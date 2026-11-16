import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');
const drawFrame = source.match(/        void onDrawFrame\(\) \{([\s\S]*?)\n        }\n\n        private void drawBackground/);
const capture = source.match(/        private boolean capture\(Frame frame\) \{([\s\S]*?)\n        }\n\n        private void addColoredSeeds/);
assert.ok(drawFrame, 'CoverageRenderer frame loop should be present');
assert.ok(capture, 'Native keyframe capture should be present');
assert.doesNotMatch(drawFrame[1], /updateCoverage\s*\(/, 'preview frames must not add coverage observations');
assert.match(capture[1], /image\s*=\s*frame\.acquireCameraImage\(\)[\s\S]*frame\.getTimestamp\(\)[\s\S]*image\.getTimestamp\(\)[\s\S]*camera\.getPose\(\)[\s\S]*jpegFromYuv\(image\)[\s\S]*updateCoverage\(frame\)/,
  'the matching image timestamp, pose, and intrinsics must be frozen before slow keyframe processing');
assert.match(source, /CAPTURE_TIMESTAMP_TOLERANCE_NS = 5_000_000L/,
  'stale CPU images should be rejected instead of paired with a moving camera pose');
assert.match(capture[1], /createAnchor\(pose\)[\s\S]*jpegFromYuv\(image\)/,
  'the exposure-time anchor must be created before JPEG encoding and disk I/O');
assert.match(capture[1], /capture_timestamp_ns[\s\S]*pose_timestamp_ns/,
  'export metadata should retain both synchronization timestamps for diagnostics');
assert.equal((source.match(/updateCoverage\(frame\)/g) || []).length, 1,
  'the keyframe path should be the sole caller that updates the accumulated coverage cloud');
console.log('coverage cloud updates only on keyframe capture: passed');
