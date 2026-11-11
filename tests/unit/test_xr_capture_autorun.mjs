import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const xr = readFileSync(new URL('../../app/js/xr-capture.js', import.meta.url), 'utf8');

// Auto mode mirrors the native capture: checking arms it, the shutter
// starts/stops the run, unchecking returns to one-shot manual photos.
assert.match(xr, /auto: false,\s*autoRunning: false,/,
  'auto mode should start disarmed and stopped');
assert.match(xr, /state\.auto = !!autoInput\.checked;\s*state\.autoRunning = false;/,
  'checking/unchecking should arm/stop the mode, never start the run by itself');
assert.match(xr, /if \(state\.auto\) \{\s*\/\/ auto mode: the shutter starts\/stops the automatic run\s*state\.autoRunning = !state\.autoRunning;/,
  'the shutter should start/stop the run in auto mode');
assert.match(xr, /state\.manualRequests \+= 1;\s*setStatus\('Shutter armed;/,
  'manual mode should keep arming one shot per press');
assert.match(xr, /const auto = state\.auto && state\.autoRunning && state\.distanceTrigger\.sample\(position\);/,
  'automatic shots need the mode armed AND the run started');
assert.match(xr, /if \(state\.auto && state\.autoRunning\) state\.distanceTrigger\.captured\(position\);/,
  'the distance accumulator should only reset while the run is started');
assert.match(xr, /const label = !state\.auto \? 'Shutter' : state\.autoRunning \? 'Stop' : 'Start';/,
  'the shutter should read Shutter / Start / Stop for the three states');

console.log('WebXR auto-capture start/stop on the shutter: passed');
