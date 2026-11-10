import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// The hover readout rides the cursor inside the plot: pinned to the top edge
// it hung past the stage in the expanded graph and hid behind the dock.
assert.match(chartSource, /yPct: this\.hover\.y \* 100/,
  'the chart should report the pointer height with its hover readout');

const tipStart = appSource.indexOf('function chartTip(');
const tipEnd = appSource.indexOf('// ── flash ──', tipStart);
assert.ok(tipStart >= 0 && tipEnd > tipStart, 'the chart hover readout should exist');
const tip = appSource.slice(tipStart, tipEnd);
assert.match(tip, /h\.yPct/,
  'the readout should anchor to the cursor height, not the plot top edge');
assert.match(tip, /wrap\.clientWidth[\s\S]*tip\.offsetWidth[\s\S]*Math\.max\(tw \/ 2/,
  'the readout should clamp sideways inside the plot');
assert.match(tip, /translate\(-50%, calc\(-100% - 70px\)\)[\s\S]*translate\(-50%, 70px\)/,
  'the readout should float 70px clear of the cursor, flipping below it near the top');

console.log('chart hover readout placement inside the plot: passed');
