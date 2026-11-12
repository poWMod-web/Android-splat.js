import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Chart, psnrToApproxSsimPercent, FastRobustZLEMA } from '../../app/js/chart.js';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browserSource = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const appCss = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseCss = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');

const formula = (psnr) => 100 / (1 + (18 / psnr) ** 4);
for (const psnr of [10, 20, 30, 45]) {
  assert.ok(Math.abs(psnrToApproxSsimPercent(psnr) - formula(psnr)) < 1e-10,
    `PSNR ${psnr} should use the requested approximation`);
}
assert.ok(Math.abs(psnrToApproxSsimPercent(18) - 50) < 1e-10,
  'PSNR 18 dB is the midpoint of the approximation and should read 50%');
assert.ok(psnrToApproxSsimPercent(10) < psnrToApproxSsimPercent(20) &&
  psnrToApproxSsimPercent(20) < psnrToApproxSsimPercent(30),
  'the approximation should rise monotonically with PSNR');
assert.equal(psnrToApproxSsimPercent(0), 0);
assert.equal(psnrToApproxSsimPercent(-1), 0);
assert.equal(psnrToApproxSsimPercent(Number.NaN), 0);
assert.equal(psnrToApproxSsimPercent(Infinity), 100);

const chartCanvas = { getContext: () => ({}), addEventListener: () => {} };
const scoreChart = new Chart(chartCanvas);
// flat past the hold: the filter latches flats exactly, so the conversion
// reads through sample by sample
scoreChart.train = [[0, 25], [100, 25], [200, 25], [300, 25], [400, 25], [500, 25]];
scoreChart.hold = [[0, 15], [100, 25]];
scoreChart.scale = 'similarity';
assert.deepEqual(scoreChart._activeSeries(), { train: scoreChart.train, hold: scoreChart.hold },
  'the approximate scale should reuse the PSNR samples, without a separate model-derived history');
const simPlotted = scoreChart._displaySeries(scoreChart.train);
assert.deepEqual(simPlotted.map((p) => p[0]), [300, 400, 500],
  'the similarity graph should plot past the held samples');
for (const p of simPlotted) {
  assert.ok(Math.abs(p[1] - formula(25)) < 1e-9,
    'the graph should apply the requested PSNR-to-SSIM approximation to each plotted sample');
}

assert.match(chartSource, /const ratio = 18 \/ psnr/,
  'the graph should implement the requested approximation formula');
assert.match(chartSource, /100 \/ \(1 \+ ratio \*\* 4\)/,
  'the graph should use the fourth-power logistic form of the approximation');
assert.match(chartSource, /this\.scale === 'similarity' \? psnrToApproxSsimPercent\(value\) : value/,
  'the approximate scale should convert PSNR samples, not evaluate the model for SSIM');
assert.doesNotMatch(chartSource, /ssimTrain|ssimHold/,
  'the chart should not depend on separately calculated SSIM histories');
assert.match(appSource, /function formatTrainingScore\(psnr\)[\s\S]*psnrToApproxSsimPercent\(psnr\)/,
  'the metric above splat count should show the same approximate value as the graph');
assert.match(appSource, /const unit = similarity \? '%' : 'dB'/,
  'the headline should show only a percent sign for the estimated scale');
assert.doesNotMatch(appSource, /id="t-title"|class="t-state"/,
  'training and paused status text should not precede the score');
assert.doesNotMatch(appSource, /SSIM % approx\.|Use approx\. SSIM %/,
  'the UI should use only % and Use % wording, not an approximation label');
assert.match(appSource, /const next = S\.scoreScale === 'db' \? 'Use %' : 'Use dB'/,
  'the scale button should say Use % for the percentage graph');
assert.match(chartSource, /unit: this\.scale === 'similarity' \? '%' : 'dB'/,
  'the chart tooltip should show only the percent unit');
assert.doesNotMatch(appSource, /maybeEvaluateSsim|evalGraphSsim|evalGraphMetrics/,
  'displaying the approximate scale must not launch a model SSIM evaluation');
assert.match(appSource, /function toggleScoreScale\(\)[\s\S]*S\.scoreScale = S\.scoreScale === 'db' \? 'similarity' : 'db'/,
  'the controls should switch between PSNR dB and the approximate SSIM percentage');
assert.match(appSource, /<div class="score-sub"><span id="t-splats">[\s\S]*?<\/div>\s*<button class="chart-scale score-scale" id="chart-scale"/,
  'the training scale toggle should sit directly below the splat count');
assert.match(appSource, /const expandPlot = training && !S\.trainVisuals[\s\S]*stage\.insertBefore\(plot, \$\('hud'\)\)[\s\S]*plot\.classList\.add\('chartwrap-expanded'\)/,
  'hiding training visuals should move the score graph into the stage');
assert.match(appSource, /plot\.classList\.remove\('chartwrap-expanded'\)/,
  'showing training visuals should return the score graph to the dock');
const trainingDockPositions = [...appSource.matchAll(/dock\('train'\);/g)].map((match) => match.index);
assert.ok(trainingDockPositions.length >= 2 && trainingDockPositions.every((position) => {
  const syncAt = appSource.indexOf('syncTrainingVisualsControl();', position);
  const drawAt = appSource.indexOf('if (chart)', position);
  return syncAt > position && syncAt < drawAt;
}), 'initial training and continued training should expand the chart after creating the training dock');
assert.match(appSource, /if \(plotMoved\) requestAnimationFrame\(\(\) => chart\?\.resize\(\)\)/,
  'the chart backing canvas should be redrawn at its enlarged size instead of stretching the old bitmap');
assert.match(appCss, /\.stage > \.chartwrap-expanded\s*\{[\s\S]*inset: 0/,
  'the expanded score graph should fill the training visual stage');
assert.match(baseCss, /\.score \.score-scale\s*\{[\s\S]*display: block/,
  'the training scale toggle should render below the splat count');
assert.match(html, /id="d-chart-scale"/,
  'the Details score graph should expose the same scale toggle');
assert.match(appSource, /\$\('d-chart-wrap'\)\.hidden = vis !== 'chart'/,
  'the score chart wrapper should be shown when the Details score tab is active');
assert.match(appCss, /\.sheet-vis canvas\[hidden\]\s*\{\s*display:\s*none\s*;/,
  'inactive Details canvases should stay hidden despite the general canvas display rule');

// the cold start is HELD, not trimmed after the fact: short series draw
// nothing at all, so early points never appear and then vanish
const holdChart = new Chart(chartCanvas);
holdChart.train = [[0, 10], [100, 12], [200, 14]];
assert.deepEqual(holdChart._displaySeries(holdChart.train), [],
  'the plot should hold back the first few samples entirely');
const trimChart = new Chart(chartCanvas);
trimChart.train = [[0, 10], [100, 12], [200, 14], [300, 16], [400, 18], [500, 20]];
const drawn = trimChart._displaySeries(trimChart.train);
assert.deepEqual(drawn.map((p) => p[0]), [300, 400, 500],
  'the plot should start the line past the held samples');
assert.equal(drawn[0][1], 16,
  'the filter should latch the first plotted sample');
assert.match(chartSource, /SKIP_FIRST = 3/,
  'the cold-start hold should cover the first three samples');
// zero-lag EMA with a Z-score spike clamp: latch, track flats, tame spikes
const zlatch = new FastRobustZLEMA();
assert.equal(zlatch.update(20), 20, 'the first sample latches the level');
const zflat = new FastRobustZLEMA();
zflat.update(25);
assert.ok(Math.abs(zflat.update(25) - 25) < 1e-9 && Math.abs(zflat.update(25) - 25) < 1e-9,
  'flat samples pass through the filter');
const zspike = new FastRobustZLEMA();
for (let i = 0; i < 30; i++) zspike.update(20);
const zclamped = zspike.update(40);
assert.ok(zclamped < 21,
  `a +20 dB spike should clamp near the level instead of yanking the curve (got ${zclamped})`);
const zseries = new FastRobustZLEMA().series([[0, 10], [100, 12], [200, 14]]);
assert.deepEqual(zseries.map((p) => p[0]), [0, 100, 200],
  'a series pass should keep every sample iteration');
assert.equal(zseries[0][1], 10, 'a series pass should latch its first sample');
assert.match(chartSource, /class FastRobustZLEMA/,
  'the curve should run through the zero-lag filter');
assert.match(chartSource, /new FastRobustZLEMA\(\)\.series\(held\)/,
  'each draw should run a fresh filter pass over the held samples');
// raw dots: tiny — the curve carries the reading, the scatter just shows it
assert.match(chartSource, /ctx\.arc\(X\(p\[0\]\), Y\(p\[1\]\), 0\.75 \* dpr, 0, 7\)/,
  'raw-sample dots should be much smaller than the old 1.5px');
// dynamic x: the axis grows with the plotted data, like the y-axis
const xChart = new Chart(chartCanvas);
assert.equal(xChart._xMaxFor([100]), 106, 'early data should use the full width plus headroom');
assert.equal(xChart._xMaxFor([0]), 1, 'an empty plot should still have a span');
xChart.maxIter = 40000;
assert.equal(xChart._xMaxFor([40000]), 40000, 'the axis should never run past the horizon');
assert.equal(xChart._xMaxFor([100, 5000]), 5300, 'the furthest series should set the axis');
// the score readouts show the filtered curve end, not the raw sample
assert.match(appSource, /function shownScore\(which\)/,
  'the app should resolve the displayed score through the filter');
assert.match(appSource, /formatTrainingScore\(shownScore\('train'\)\)/,
  'the dock score should show the filtered train value');
assert.match(appSource, /const gap = \(shownScore\('train'\) != null && shownScore\('hold'\) != null\)/,
  'the details gap should compare filtered scores');
assert.match(browserSource, /const ft = shownScore\('train'\), fh = shownScore\('hold'\);/,
  'the browser metrics line should show filtered scores');

console.log('PSNR and approximate SSIM graph scales: passed');
