import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// The hover readout rides the cursor inside the plot: pinned to the top edge
// it hung past the stage in the expanded graph and hid behind the dock.
assert.match(chartSource, /yPct: this\.hover\.y \* 100/,
  'the chart should report the pointer height with its hover readout');

const tipStart = appSource.indexOf('function positionChartTip(');
const tipEnd = appSource.indexOf('// ── flash ──', tipStart);
assert.ok(tipStart >= 0 && tipEnd > tipStart, 'the shared chart hover positioning should exist');
const tip = appSource.slice(tipStart, tipEnd);
assert.match(tip, /h\.yPct/,
  'the readout should anchor to the cursor height, not the plot top edge');
assert.match(tip, /wrap\.clientWidth[\s\S]*tip\.offsetWidth[\s\S]*Math\.max\(tw \/ 2/,
  'the readout should clamp sideways inside the plot');
assert.match(tip, /translate\(-50%, calc\(-100% - 70px\)\)[\s\S]*translate\(-50%, 70px\)/,
  'the readout should float 70px clear of the cursor, flipping below it near the top');
assert.match(tip, /function chartTip\(h\)[\s\S]*positionChartTip\(tip, h\)/,
  'foreground training should use the shared above-cursor placement');
assert.match(appSource, /new Chart\(\$\('mon-chart'\)[\s\S]*tip\.textContent[\s\S]*positionChartTip\(tip, h\)/,
  'background monitoring should use the same above-cursor placement');

// Layout shifts (window resize, panel toggle, chart expand) move the plot
// under a stationary cursor: the stale hover must drop, never re-fire.
assert.match(chartSource, /clearHover\(\) \{\s*this\.hover = null;\s*this\.onHover\?\.\(null\);/,
  'the chart should offer a hover reset that also hides the readout');
assert.match(chartSource, /resize\(\) \{[\s\S]*?this\.clearHover\(\);/,
  'resizing the chart should drop a stale hover');
assert.match(appSource, /\$\('settings'\)\.hidden = !open;[\s\S]*?chart\?\.clearHover\(\);/,
  'toggling the settings panel should drop a stale chart hover');

console.log('chart hover readout placement inside the plot: passed');
