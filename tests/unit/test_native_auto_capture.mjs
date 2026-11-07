import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');
assert.match(source, /CheckBox autoToggle = new CheckBox\(this\);[\s\S]*autoToggle\.setText\("Auto-capture"\)/);
assert.match(source, /autoToggle\.setChecked\(false\)/, 'native capture must retain the WebXR manual-capture default');
assert.match(source, /getInt\("capture_spacing_cm", 10\)/, 'native default spacing must be 10 cm');
assert.match(source, /if \(value < 2 \|\| value > 200\)/, 'spacing must be editable from 2 to 200 cm');
assert.match(source, /boolean autoDue = autoCaptureEnabled && cloudRenderer\.sampleTravel\(position\)/);
assert.match(source, /if \(manualDue \|\| autoDue\) capture\(frame\)/);
assert.match(source, /distanceSinceCapture \+= distance[\s\S]*return distanceSinceCapture \+ 1e-6f >= spacingMeters/);
assert.match(source, /void captured\(float\[\] position\)[\s\S]*distanceSinceCapture = 0f/);
console.log('native ARCore distance auto-capture UI and 10 cm default: passed');
