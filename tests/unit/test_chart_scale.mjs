import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { psnrToSimilarity } from '../../app/js/chart.js';

const chartSource = readFileSync(new URL('../../app/js/chart.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const appCss = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');
const baseCss = readFileSync(new URL('../../app/css/base.css', import.meta.url), 'utf8');

assert.equal(psnrToSimilarity(0), 0);
assert.equal(psnrToSimilarity(10), 90);
assert.ok(Math.abs(psnrToSimilarity(20) - 99) < 1e-10);
assert.equal(psnrToSimilarity(-10), 0, 'negative PSNR should not produce a negative similarity');
assert.equal(psnrToSimilarity(1000), 100, 'similarity should remain capped at 100 percent');
assert.equal(psnrToSimilarity(Number.NaN), 0, 'invalid metrics should not poison the graph scale');
assert.match(chartSource, /this\.scale === 'similarity'[\s\S]*?%/,
  'the selected scale must drive percent axis labels and coordinates');
assert.match(appSource, /function toggleScoreScale\(\)[\s\S]*S\.scoreScale = S\.scoreScale === 'db' \? 'similarity' : 'db'/,
  'the graph controls should switch between dB and similarity on both score charts');
assert.match(appSource, /<div class="score-sub"><span id="t-splats">[\s\S]*?<\/div>\s*<button class="chart-scale score-scale" id="chart-scale"/,
  'the training scale toggle should sit directly below the splat count');
assert.match(appSource, /const expandPlot = training && !S\.trainVisuals[\s\S]*stage\.insertBefore\(plot, \$\('hud'\)\)[\s\S]*plot\.classList\.add\('chartwrap-expanded'\)/,
  'hiding training visuals should move the score graph into the stage');
assert.match(appSource, /plot\.classList\.remove\('chartwrap-expanded'\)/,
  'showing training visuals should return the score graph to the dock');
assert.match(appCss, /\.stage > \.chartwrap-expanded\s*\{[\s\S]*inset: 0/,
  'the expanded score graph should fill the training visual stage');
assert.match(baseCss, /\.score \.score-scale\s*\{[\s\S]*display: block/,
  'the training scale toggle should render below the splat count');
assert.match(html, /id="d-chart-scale"/,
  'the Details score graph should expose the same scale toggle');

console.log('PSNR to normalized pixel-similarity scale and graph toggle: passed');
