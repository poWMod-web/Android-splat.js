import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');
assert.match(source, /CheckBox autoToggle = new CheckBox\(this\);[\s\S]*autoToggle\.setText\("Auto-capture"\)/);
assert.match(source, /autoToggle\.setChecked\(false\)/, 'native capture must retain the WebXR manual-capture default');
assert.match(source, /getInt\("capture_spacing_cm", 10\)/, 'native default spacing must be 10 cm');
assert.match(source, /if \(value < 2 \|\| value > 200\)/, 'spacing must be editable from 2 to 200 cm');
assert.match(source, /autoFirstCapturePending = running/,
  'pressing Start should arm the first sample while tracking warms up');
assert.match(source, /boolean firstAutoDue = autoCaptureEnabled && autoCaptureRunning && autoFirstCapturePending;[\s\S]*\(firstAutoDue \|\| cloudRenderer\.sampleTravel\(position\)\)/,
  'the first sample is due immediately; distance spacing governs later samples');
assert.match(source, /if \(firstAutoDue && saved\) autoFirstCapturePending = false/,
  'a transient unavailable camera image must not consume the first sample');
assert.match(source, /if \(!autoCaptureEnabled \|\| !autoCaptureRunning\) cloudRenderer\.breakTravelContinuity\(\);/,
  'a stopped run must not accumulate travel');
assert.match(source, /if \(autoCaptureEnabled\) \{\s*\/\/ auto mode: the center button starts\/stops the automatic run\s*setAutoCaptureRunning\(!autoCaptureRunning\);/,
  'the center button should start/stop the run in auto mode');
assert.match(source, /captureRequested\.set\(true\);\s*showStatus\("Shutter armed;/,
  'manual mode should keep arming one shot per press');
assert.match(source, /autoCaptureEnabled = enabled;\s*autoCaptureRunning = false;/,
  'checking/unchecking should arm/stop the mode, never start the run by itself');
assert.match(source, /else if \(autoCaptureRunning\) shutterView\.setLabel\("Stop", "Stop auto-capture"\);/,
  'a running auto capture should offer Stop on the center button');
assert.match(source, /else shutterView\.setLabel\("Start", "Start auto-capture"\);/,
  'an armed auto capture should offer Start on the center button');
assert.match(source, /boolean earlyCaptureReady = openingCaptureTrackingReady\(frame\)[\s\S]*if \(manualPending \|\| autoDue\)[\s\S]*if \(earlyCaptureReady\)[\s\S]*capture\(frame\)/,
  'manual and automatic opening captures should wait for positional tracking readiness');
assert.match(source, /OPENING_CAPTURE_MIN_REFERENCES = 24/,
  'the first two images should require a live set of ARCore positional references');
assert.match(source, /frame\.acquirePointCloud\(\)[\s\S]*currentTrackingReferences\+\+[\s\S]*return currentTrackingReferences >= OPENING_CAPTURE_MIN_REFERENCES/,
  'opening readiness should be based on current map evidence');
assert.doesNotMatch(source, /WARMUP_MS|FALLBACK_MS|SETTLE_MS|elapsedRealtime\(\)/,
  'opening capture readiness must not contain a timer warm-up or elapsed-time fallback');
assert.match(source, /keep manual\/first-auto requests pending/,
  'a shutter press while tracking becomes ready must remain armed rather than being discarded');
assert.match(source, /capturedAnchor = arSession\.createAnchor\(pose\)[\s\S]*frameAnchors\.add\(capturedAnchor\)/,
  'captured poses should remain attached to ARCore map refinement');
assert.match(source, /refreshCapturedAnchorPoses\(\)[\s\S]*anchor\.getPose\(\)\.toMatrix[\s\S]*transform_matrix/,
  'export should replace early shutter snapshots with the latest tracked anchor poses');
assert.match(source, /distanceSinceCapture \+= distance[\s\S]*return distanceSinceCapture \+ 1e-6f >= spacingMeters/);
assert.match(source, /void captured\(float\[\] position\)[\s\S]*distanceSinceCapture = 0f/);
console.log('native ARCore distance auto-capture UI and 10 cm default: passed');
