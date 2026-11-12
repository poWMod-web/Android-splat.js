import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { convertSHDegree, continuedInputDim, continuedBuf } from '../../app/js/session_io.js';
import { adaptiveTrainCap } from '../../src/io/frames.js';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const sessionIo = readFileSync(new URL('../../app/js/session_io.js', import.meta.url), 'utf8');

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
assert.match(card, /id="ltc-softprune"/, 'the card should offer a soft-prune select');
assert.match(card, /\.\.\.\[480, 640, 960, 1280, 1600, 1920\]\.map\(\(v\) =>/,
  'the card should list every start-panel resolution step, unfiltered');
assert.match(card, /Same as current/,
  'the splat budget should default to Same as current');
assert.match(card, /const prevBuf = continuedBuf\(recon\.trainScale, r\.detailProgress && r\.detailProgress\.settings, prevDim \|\| undefined\);/,
  'the card should resolve the run’s buffer from the recon or its saved settings');
assert.match(card, /<option value="same" selected>\$\{prevInputDim \? `\$\{prevInputDim\} px` : 'Run size'\}<\/option>/,
  'the resolution should default to the run’s own input size, in start-panel language');
assert.match(card, /<option value="" selected>\$\{prevSh\}<\/option>/,
  'spherical harmonics should default to the run’s own degree, stated plainly');
assert.match(card, /<option value="" selected>\$\{prevBuf\}×<\/option>/,
  'the working buffer should default to the run’s own scale, stated plainly');
assert.match(card, /<option value="" selected>\$\{prevMcmc \? 'On' : 'Off'\}<\/option>/,
  'the MCMC optimizer should default to the run’s own set, stated plainly');
assert.match(card, /const prevSoftRatio = recipe0 && recipe0\.softPrune && Number\.isFinite\(recipe0\.softPrune\.ratio\)/,
  'the card should read the run’s own trim from its saved recipe');
assert.match(card, /<option value="" selected>\$\{prevSoft\}<\/option>/,
  'soft prune should default to the run’s own trim, stated plainly');
assert.match(card, /softprune: softRaw === '' \? undefined : parseFloat\(softRaw\)/,
  'confirming the card should pass the reselected soft-prune trim through');
assert.ok(!/Same as (before|run)/.test(card),
  'only the splat budget should keep a Same-as label');
assert.match(card, /continueLocalRun\(false, \{\s*res:[\s\S]*splats:[\s\S]*extraIters,[\s\S]*sh:[\s\S]*buf:[\s\S]*lod:[\s\S]*mcmc:/,
  'confirming the card should continue with the full chosen configuration');

const contStart = app.indexOf('async function continueLocalRun(viewOnly = false, opts = {})');
assert.ok(contStart >= 0, 'the continuation should accept configuration overrides');
const contEnd = app.indexOf('function continueTraining(', contStart);
const cont = app.slice(contStart, contEnd);
assert.match(cont, /opts\.extraIters[\s\S]*defaultExtra/,
  'the continuation horizon should honor configured extra cycles, defaulting to the automatic extension');
assert.match(cont, /opts\.res === 'auto' \? undefined[\s\S]*opts\.res === 'full' \? nativeDim[\s\S]*: prevInputDim/,
  'the continuation should honor a reselected resolution, defaulting to the run’s own');
assert.match(cont, /const prevBuf = continuedBuf\(recon\.trainScale, lr\.detailProgress && lr\.detailProgress\.settings, prevDim\);/,
  'the continuation should resolve the run’s buffer from the recon or its saved settings');
assert.match(cont, /const prevInputDim = continuedInputDim\(prevDim, prevBuf\);/,
  'the default should divide the buffer back out before the multiplier runs again');
assert.match(cont, /if \(opts\.softprune !== undefined\) \{/,
  'an untouched soft-prune knob should replay the recipe’s own setting');
assert.match(cont, /recipeTrainer\.softPrune = \{ ratio: opts\.softprune, every: 3000, until: 0\.85 \}/,
  'a reselected soft-prune trim should rebuild the periodic cull schedule');
assert.match(cont, /Math\.max\(fr0\.nw \|\| fr0\.fw \|\| 0, fr0\.nh \|\| fr0\.fh \|\| 0\)/,
  'Full size should use the file’s own pixel size, not the feature-capped dims');
assert.match(sessionIo, /\.\.\.\(Number\.isFinite\(f\.srcW\) \? \{ nw: f\.srcW, nh: f\.srcH \} : \{\}\)/,
  'the recon JSON should persist each frame’s native pixel size');
assert.match(cont, /Math\.max\(g\.n, lr\.splats \|\| 0,[\s\S]*opts\.splats \|\| 0\)/,
  'the continuation budget should honor a reselected splat count, never below the trained size');
assert.match(cont, /frames: \{ trainMaxDim, \.\.\.\(bufScale != null \? \{ trainScale: bufScale \} : \{\}\) \}/,
  'the rebuilt session should train at the selected resolution and buffer scale');
assert.match(cont, /const bufScale = opts\.buf \?\? \(prevBuf !== 1 \? prevBuf : undefined\);/,
  'an unsettled buffer should replay the run’s own resolved scale');
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

// The default-resolution round trip, through the REAL cap: the stored size
// is adaptiveTrainCap(input, buf); the continuation must recover an input
// that reproduces the stored size exactly (no double multiplication).
{
  assert.equal(continuedInputDim(undefined, 0.5), undefined);
  assert.equal(continuedInputDim(480, undefined), 480);
  assert.equal(continuedInputDim(480, 1), 480);
  assert.equal(continuedInputDim(0, 0.5), 0);
  const cap = (input, nat, buf) => adaptiveTrainCap(10, nat, Math.round(nat * 0.75),
    { trainMaxDim: input, ...(buf === 1 ? {} : { trainScale: buf }) });
  for (const buf of [0.5, 1, 2]) {
    for (const nat of [960, 1920, 4000]) {
      for (let input = 240; input <= 2400; input += 17) {
        const stored = cap(input, nat, buf);
        const recovered = continuedInputDim(stored, buf);
        assert.equal(cap(recovered, nat, buf), stored,
          `input ${input} @${nat} buf ${buf} should round-trip through the stored size`);
      }
    }
  }
}

// The replayed buffer: the recon's own trainScale (exact); else the run
// record's saved settings when their resolution × buffer triple reproduces
// the stored size (drifted settings fail the triple and fall back to 1).
{
  assert.equal(continuedBuf(0.5, null, 480), 0.5);
  assert.equal(continuedBuf(1, null, 960), 1);
  assert.equal(continuedBuf(undefined, null, 480), 1);
  assert.equal(continuedBuf(undefined, { res: 960, buf: 0.5 }, 480), 0.5);
  assert.equal(continuedBuf(undefined, { res: 960, buf: 2 }, 480), 1);
  assert.equal(continuedBuf(undefined, { res: 1920, buf: 0.5 }, 480), 1);
  assert.equal(continuedBuf(undefined, { res: 0, buf: 0.5 }, 480), 0.5);
  assert.equal(continuedBuf(undefined, { res: -1, buf: 2 }, 1920), 2);
  assert.equal(continuedBuf(undefined, {}, 480), 1);
  assert.equal(continuedBuf(undefined, { res: 960, buf: 0.5 }, 0), 0.5);
  assert.equal(continuedBuf(NaN, { res: 960, buf: 0.5 }, 480), 0.5);
}

// A fresh run's done screen must reach the same finished-run path: finish()
// adopts the secured record in memory (Train then offers the config card +
// history like Yours), and an immediate tap waits for the securing pass
// instead of taking the blind live resume.
assert.match(app, /S\._securing = \(async \(\) => \{/,
  'finish should expose its securing work for the Train gate');
assert.match(app, /status: 'finished', createdAt: Date\.now\(\), updatedAt: Date\.now\(\),/,
  'finish should adopt the finished record in memory');
assert.match(app, /if \(S\.state === 'done'\) renderControls\(\);   \/\/ swap the quick Train for the run's own/,
  'adopting the record should re-render the done controls');
assert.match(app, /more\.addEventListener\('click', trainAfterDone\);/,
  'the pre-secure Train button should go through the securing gate');
assert.match(app, /if \(S\._localRun\) \{ trainLocalChoice\(\); return; \}/,
  'an adopted record should take the finished-run choice dialog');
assert.match(app, /try \{ await S\._securing; \} catch/,
  'an immediate tap should wait for the securing pass first');
// The blind live resume stays for record-less sessions only — and it must
// await the horizon: assigning continueFor's promise used to NaN the
// chart's x-scale and blank the restored history (2026-11).
assert.match(app, /async function continueTraining\(\) \{/,
  'the live resume should be async');
assert.match(app, /S\.maxIters = await S\.session\.continueFor\(MORE_ITERS\);/,
  'the live resume should await the horizon (a number, never a promise)');

console.log('continue-training configuration reselection: passed');
