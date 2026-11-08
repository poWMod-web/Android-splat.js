import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Chart, psnrToApproxSsimPercent } from '../../app/js/chart.js';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const appCss = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseCss = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');

const formula = (psnr) => (psnr / (psnr + (320 / psnr) ** 1.5)) * 100;
for (const psnr of [10, 20, 30, 45]) {
  assert.ok(Math.abs(psnrToApproxSsimPercent(psnr) - formula(psnr)) < 1e-10,
    `PSNR ${psnr} should use the requested approximation`);
}
assert.ok(Math.abs(psnrToApproxSsimPercent(30) -
  (100 * 30 / (30 + 5700 * 30 ** -1.5))) < 0.2,
'320^1.5 and the rounded 5700 coefficient should give equivalent results');
assert.equal(psnrToApproxSsimPercent(0), 0);
assert.equal(psnrToApproxSsimPercent(-1), 0);
assert.equal(psnrToApproxSsimPercent(Number.NaN), 0);
assert.equal(psnrToApproxSsimPercent(Infinity), 100);

const chartCanvas = { getContext: () => ({}), addEventListener: () => {} };
const scoreChart = new Chart(chartCanvas);
scoreChart.train = [[0, 20], [100, 30]];
scoreChart.hold = [[0, 15], [100, 25]];
scoreChart.scale = 'similarity';
assert.deepEqual(scoreChart._activeSeries(), { train: scoreChart.train, hold: scoreChart.hold },
  'the approximate scale should reuse the PSNR samples, without a separate model-derived history');
assert.deepEqual(scoreChart._displaySeries(scoreChart.train), [
  [0, formula(20)], [100, formula(30)],
], 'the graph should apply the requested PSNR-to-SSIM approximation to each sample');

assert.match(chartSource, /const penalty = \(320 \/ psnr\) \*\* 1\.5/,
  'the graph should implement the requested approximation formula');
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

console.log('PSNR and approximate SSIM graph scales: passed');
