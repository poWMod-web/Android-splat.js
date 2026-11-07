import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ssimToPercent } from '../../app/js/chart.js';
import { ssimIndexRgb } from '../../src/gs/ssim.js';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const appCss = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseCss = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');
const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const shaderSource = readFileSync(new URL('../../src/gs/shaders.js', import.meta.url), 'utf8');

assert.equal(ssimToPercent(0), 0);
assert.equal(ssimToPercent(0.5), 50);
assert.equal(ssimToPercent(1), 100);
assert.equal(ssimToPercent(-0.1), 0, 'negative SSIM should not produce a negative percentage');
assert.equal(ssimToPercent(1.1), 100, 'SSIM percentage should be capped at 100');
assert.equal(ssimToPercent(Number.NaN), 0, 'invalid metrics should not poison the graph scale');

const w = 21, h = 21;
const a = new Float32Array(w * h * 3).fill(0.35);
const identical = ssimIndexRgb(a, a, w, h);
assert.ok(Math.abs(identical - 1) < 3e-6, `identical RGB images should score 1, got ${identical}`);
const changed = Float32Array.from(a, (v, i) => i % 3 === 0 ? Math.min(1, v + 0.25) : v);
assert.ok(ssimIndexRgb(a, changed, w, h) < identical, 'structurally different images must score lower');
assert.equal(ssimIndexRgb(a, a, w, h, new Uint8Array(w * h)), null,
  'an image with no valid target pixels should produce no SSIM score');

assert.match(chartSource, /ssimToPercent\(value\)/,
  'percentage coordinates must come from the SSIM samples');
assert.doesNotMatch(chartSource, /psnrToSimilarity|10\s*\*\*\s*\(-psnr/,
  'PSNR must not be converted into a fake similarity percentage');
assert.match(appSource, /function toggleScoreScale\(\)[\s\S]*S\.scoreScale = S\.scoreScale === 'db' \? 'similarity' : 'db'/,
  'the controls should switch between PSNR dB and actual SSIM percentage');
assert.match(appSource, /evalGraphMetrics\(trainCi, holdCi\)/,
  'SSIM samples should be evaluated from rendered train and holdout views');
assert.match(appSource, /S\.ssimTrain = train\.ssim[\s\S]*S\.ssimTrainHist\.push\(\[iter, train\.ssim\]\)/,
  'the SSIM headline and chart history must retain actual SSIM evaluation results');
assert.match(appSource, /function renderTrainingScore\(\)[\s\S]*S\.scoreScale === 'similarity'[\s\S]*ssimToPercent\(v\)[\s\S]*t-punit[\s\S]*SSIM %/,
  'the metric above splat count must display SSIM as a percentage when that scale is selected');
assert.match(trainerSource, /evalCamMetrics\(ci\)[\s\S]*_evalPass\(ci, undefined, true\)/,
  'the graph must use the trainer render evaluation path to obtain SSIM inputs');
assert.match(shaderSource, /export const SSIM_METRIC_SRC[\s\S]*cam\.proj\.w[\s\S]*metric\[oi \+ 4u\]/,
  'the GPU readback should use rendered RGB and the exposure-corrected target');
assert.match(appSource, /<div class="score-sub"><span id="t-splats">[\s\S]*?<\/div>\s*<button class="chart-scale score-scale" id="chart-scale"/,
  'the training scale toggle should sit directly below the splat count');
assert.match(appSource, /const expandPlot = training && !S\.trainVisuals[\s\S]*stage\.insertBefore\(plot, \$\('hud'\)\)[\s\S]*plot\.classList\.add\('chartwrap-expanded'\)/,
  'hiding training visuals should move the score graph into the stage');
assert.match(appSource, /plot\.classList\.remove\('chartwrap-expanded'\)/,
  'showing training visuals should return the score graph to the dock');
assert.match(appSource, /if \(plotMoved\) requestAnimationFrame\(\(\) => chart\?\.resize\(\)\)/,
  'the chart backing canvas should be redrawn at its enlarged size instead of stretching the old bitmap');
assert.match(appCss, /\.stage > \.chartwrap-expanded\s*\{[\s\S]*inset: 0/,
  'the expanded score graph should fill the training visual stage');
assert.match(baseCss, /\.score \.score-scale\s*\{[\s\S]*display: block/,
  'the training scale toggle should render below the splat count');
assert.match(html, /id="d-chart-scale"/,
  'the Details score graph should expose the same scale toggle');

console.log('PSNR and real SSIM graph scales: passed');
