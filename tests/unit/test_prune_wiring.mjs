import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const trainer = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../../src/gs/shaders.js', import.meta.url), 'utf8');
const protocol = readFileSync(new URL('../../src/gs/trainer-protocol.js', import.meta.url), 'utf8');
const worker = readFileSync(new URL('../../src/gs/trainworker.js', import.meta.url), 'utf8');
const remote = readFileSync(new URL('../../src/gs/remote-trainer.js', import.meta.url), 'utf8');
const session = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const appHtml = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');
const browser = readFileSync(new URL('../../browser/app.js', import.meta.url), 'utf8');
const browserHtml = readFileSync(new URL('../../browser/index.html', import.meta.url), 'utf8');

// ---- the trainer culls by rendered mass, on demand -------------------------
assert.match(trainer, /async softPrune\(\) \{/,
  'the trainer serves a soft-prune call');
assert.match(trainer, /planSoftPrune\(masses, this\.n, ratio, \{/,
  'scoring plans through the pure planner');
assert.match(trainer, /this\.pipeCompact = d\.createComputePipeline\(\{/,
  'compaction runs GPU-side');
assert.match(trainer, /\|\| !!this\.opts\.softPrune\)\);/,
  'enabling soft-prune forces the stats accumulation its score needs');
assert.match(trainer, /const capped = growPhase && this\.n >= limit && limit >= this\.cap;/,
  'the ceiling reports itself (true cap only — LOD holds stay quiet)');

// ---- the gather scores without draining ------------------------------------
assert.match(shaders, /let drain = gu\.mode != 2u;/,
  'gather mode 2 is the load-only scoring pass');
assert.match(shaders, /atomicLoad\(&gradP\[b \+ 10u\]\)/,
  'mode 2 reads the rendered-mass window without zeroing it');
assert.match(shaders, /select\(em, wm, gu\.mode != 0u\);/,
  'scoring reads rendered mass like the v2 growth policy');

// ---- the worker path serves it ---------------------------------------------
assert.match(protocol, /'softPrune',    \/\/ importance-cull barrier/,
  'the RPC contract lists the call');
assert.match(worker, /case 'softPrune': \{/,
  'the worker serves it');
assert.match(remote, /async softPrune\(\) \{/,
  'the remote proxy mirrors n/iter like refine does');

// ---- the session drives cadence, prompt, and prune-and-continue ------------
assert.match(session, /const s = await trainer\.softPrune\(\);/,
  'the loop fires soft-prunes on cadence');
assert.match(session, /kind: 'soft-prune', iter: trainer\.iter/,
  'culls surface as events');
assert.match(session, /if \(refineResult && refineResult\.capped && !this\._capAsked\) \{/,
  'a capped refine pauses for the ceiling question, once per ceiling');
assert.match(session, /kind: 'cap-hit', iter: trainer\.iter, splats: trainer\.n/,
  'the question reaches the UI as an event');
assert.match(session, /async pruneAndContinue\(ratio = 0\.5,/,
  'prune-and-continue defaults to half');
assert.match(session, /await this\.continueFor\(ext\);/,
  'the reseed stretches the horizon and resumes');
assert.match(session, /kind: 'prune-continue', iter, \.\.\.out/,
  'the outcome (counts, probe delta, horizon) reaches the UI');

// ---- the full app: setting, button, card -----------------------------------
assert.match(appHtml, /<select id="set-softprune">/,
  'the gear panel offers soft prune');
assert.match(app, /st\.softprune = \$\('set-softprune'\)\.value;/,
  'the selection persists into settings');
assert.match(app, /trainerOpts\.softPrune = \{ ratio: parseFloat\(st\.softprune\), every: 3000, from: 6000 \};/,
  'fresh runs map it to trainer options');
assert.match(app, /<button class="tbtn-sm" id="t-prune"/,
  'the training dock carries a manual prune button');
assert.match(app, /\$\('t-prune'\)\.addEventListener\('click', runPruneAndContinue\);/,
  'the button runs the flow');
assert.match(app, /id="cap-prune">Prune to ½ &amp; continue<\/button>/,
  'the ceiling card offers prune-and-continue');
assert.match(app, /id="cap-keep">Keep training at cap<\/button>/,
  'the ceiling card offers staying put');

// ---- the browser mirrors it ------------------------------------------------
assert.match(browserHtml, /<select id="opt-softprune">/,
  'the browser settings offer soft prune');
assert.match(browser, /softprune: ui\.softprune\.value,/,
  'the browser reads the setting');
assert.match(browserHtml, /<button id="btn-prune" class="btn" disabled>Prune ½<\/button>/,
  'the run row carries a prune button');
assert.match(browser, /ui\.prune\.addEventListener\('click', runPruneContinue\);/,
  'the button runs the flow');
assert.match(browser, /prune\.textContent = 'Prune to ½ & continue';/,
  'the ceiling prompt offers prune-and-continue');
assert.match(browser, /keep\.textContent = 'Keep training at cap';/,
  'the ceiling prompt offers staying put');

console.log('prune-and-continue + soft-prune wiring: passed');
