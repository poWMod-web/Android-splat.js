import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { convertSHDegree } from '../../app/js/session_io.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');

// Continuing a finished run reselects the training configuration first (like
// a new training); a paused run resumes its original configuration directly.
assert.match(app, /if \(r && r\.status === 'finished'\) \{ continueConfigCard\(r\); return; \}/,
  'keep-training on a finished run should open the configuration card');

const cardStart = app.indexOf('function continueConfigCard(');
const cardEnd = app.indexOf('/** Continue training a stored local run:', cardStart);
assert.ok(cardStart >= 0 && cardEnd > cardStart, 'the continue-config card should exist');
const card = app.slice(cardStart, cardEnd);
assert.match(card, /id="ltc-res"/, 'the card should offer a training resolution select');
assert.match(card, /id="ltc-splats"/, 'the card should offer a splat budget select');
assert.match(card, /id="ltc-iters"/, 'the card should offer an extra-cycles select');
assert.match(card, /id="ltc-sh"/, 'the card should offer a spherical-harmonics select');
assert.match(card, /id="ltc-buf"/, 'the card should offer a working-buffer select');
assert.match(card, /id="ltc-lod"/, 'the card should offer an LOD select');
assert.match(card, /id="ltc-mcmc"/, 'the card should offer an MCMC-optimizer select');
assert.match(card, /\.\.\.\[480, 640, 960, 1280, 1600, 1920\]\.map\(\(v\) =>/,
  'the card should list every start-panel resolution step, unfiltered');
assert.match(card, /Same as current/,
  'the splat budget should default to Same as current');
assert.match(card, /Same as run \(\$\{prevSh\}\)/,
  'spherical harmonics should default to the run’s own degree');
assert.match(card, /Same as run \(\$\{prevBuf\}×\)/,
  'the working buffer should default to the run’s own scale');
assert.match(card, /Same as run \(\$\{prevMcmc \? 'On' : 'Off'\}\)/,
  'the MCMC optimizer should default to the run’s own set');
assert.match(card, /continueLocalRun\(false, \{\s*res:[\s\S]*splats:[\s\S]*extraIters,[\s\S]*sh:[\s\S]*buf:[\s\S]*lod:[\s\S]*mcmc:/,
  'confirming the card should continue with the full chosen configuration');

const contStart = app.indexOf('async function continueLocalRun(viewOnly = false, opts = {})');
assert.ok(contStart >= 0, 'the continuation should accept configuration overrides');
const contEnd = app.indexOf('function continueTraining(', contStart);
const cont = app.slice(contStart, contEnd);
assert.match(cont, /opts\.extraIters[\s\S]*defaultExtra/,
  'the continuation horizon should honor configured extra cycles, defaulting to the automatic extension');
assert.match(cont, /opts\.res === 'auto' \? undefined[\s\S]*opts\.res === 'full' \? nativeDim[\s\S]*: prevDim/,
  'the continuation should honor a reselected resolution, defaulting to the run’s own');
assert.match(cont, /Math\.max\(g\.n, lr\.splats \|\| 0,[\s\S]*opts\.splats \|\| 0\)/,
  'the continuation budget should honor a reselected splat count, never below the trained size');
assert.match(cont, /frames: \{ trainMaxDim, \.\.\.\(bufScale != null \? \{ trainScale: bufScale \} : \{\}\) \}/,
  'the rebuilt session should train at the selected resolution and buffer scale');
assert.match(cont, /const bufScale = opts\.buf \?\? \(recon\.trainScale/,
  'an unsettled buffer should replay the run’s own recorded scale');
assert.match(cont, /convertSHDegree\(g, opts\.sh\)/,
  'a reselected SH degree should resample the decoded checkpoint');
assert.match(cont, /shDeg: contShDeg \?\? recipeTrainer\.shDeg \?\? 3/,
  'the rebuilt trainer should use the reselected SH degree');
assert.match(cont, /opts\.mcmc === true[\s\S]*growRate: 0\.05, mcmcNoise: true, scaleReg: 0\.01, moveCap: 0\.25/,
  'enabling MCMC should rewrite the start-panel optimizer set');
assert.match(cont, /opts\.mcmc === false[\s\S]*delete recipeTrainer\.mcmcNoise; delete recipeTrainer\.scaleReg;/,
  'disabling MCMC should restore the classic schedule');
assert.match(cont, /const contRefine = opts\.mcmc === true \? 500 : opts\.mcmc === false \? undefined : recipeRefine;/,
  'the refine cadence should follow the MCMC toggle');
assert.match(cont, /if \(opts\.lod === true && Math\.max\(lr\.splats \|\| 0, opts\.splats \|\| 0\) < 1000000\)/,
  'LOD should be gated on a 1M+ budget before any teardown');
assert.match(cont, /plan\.levels\.filter\(\(l\) => l > g\.n\)/,
  'a continued LOD schedule should start above the model’s current size');
assert.match(cont, /ses\.trainer\.growLimit = S\.lodPlan\.levels\[S\.lodPlan\.idx\]/,
  'the growth limit should honor the fresh or resumed LOD plan');

// SH resampling: lowering keeps the low bands per channel, raising zero-pads.
{
  const n = 2, K = 15;
  const sh = new Float32Array(n * K * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) for (let k = 0; k < K; k++) {
    sh[(i * 3 + c) * K + k] = (i * 3 + c) * 100 + k;
  }
  const g = { n, shK: K, sh };
  convertSHDegree(g, 1);
  assert.equal(g.shK, 3);
  assert.equal(g.sh.length, n * 3 * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) {
    assert.equal(g.sh[(i * 3 + c) * 3 + k], (i * 3 + c) * 100 + k);
  }
  convertSHDegree(g, 3);
  assert.equal(g.shK, 15);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
    for (let k = 0; k < 3; k++) assert.equal(g.sh[(i * 3 + c) * 15 + k], (i * 3 + c) * 100 + k);
    for (let k = 3; k < 15; k++) assert.equal(g.sh[(i * 3 + c) * 15 + k], 0);
  }
  convertSHDegree(g, 0);
  assert.equal(g.shK, 0);
  assert.equal(g.sh, null);
  const flat = { n: 2, shK: 0, sh: null };
  convertSHDegree(flat, 2);
  assert.equal(flat.shK, 8);
  assert.ok(flat.sh.every((v) => v === 0));
  assert.throws(() => convertSHDegree({ n: 1, shK: 15, sh: new Float32Array(45) }, 4), /SH degree must be 0-3/);
  assert.throws(() => convertSHDegree({ n: 1, shK: 7, sh: new Float32Array(21) }, 1), /unknown band count/);
}

console.log('continue-training configuration reselection: passed');
