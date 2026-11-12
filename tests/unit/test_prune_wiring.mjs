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
assert.match(trainer, /async softPrune\(ratioOverride = null\) \{/,
  'the trainer serves a soft-prune call, with a milestone override');
assert.match(trainer, /const ratio = ratioOverride \?\? cfg\.ratio \?\? 0\.1;/,
  'a milestone override culls its own fraction instead of the configured trim');
assert.match(trainer, /planSoftPrune\(masses, this\.n, ratio, \{/,
  'scoring plans through the pure planner');
assert.match(trainer, /this\.pipeCompact = d\.createComputePipeline\(\{/,
  'compaction runs GPU-side');
assert.match(trainer, /\|\| !!this\.opts\.softPrune\)\);/,
  'enabling soft-prune forces the stats accumulation its score needs');
assert.match(trainer, /if \(this\.growLimit && this\.growLimit < this\.cap && this\.n >= this\.growLimit\) \{/,
  'an LOD level hold pauses culling (polish, not churn)');
assert.match(trainer, /this\._softPruneFloor = plan\.kept;/,
  'each prune sets the growth-ratchet floor for the next one');
assert.match(session, /growth ratchet held the floor/,
  'the log names the ratchet when it binds');
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
assert.match(remote, /async softPrune\(ratio = null\) \{/,
  'the remote proxy mirrors n/iter like refine does');
assert.match(worker, /const r = await tr\.softPrune\(params\.ratio \?\? null\);/,
  'the worker forwards a milestone override to its trainer');

// ---- the session drives cadence, prompt, and prune-and-continue ------------
assert.match(session, /const pruned = await trainer\.softPrune\(\);/,
  'every tandem refine culls first');
assert.match(session, /tandem prune @/,
  'the cull rides inside the scheduled refine');
assert.match(session, /\[tandem: culled /,
  'the refine log names the cull it regrew');
assert.match(session, /pruned = await trainer\.softPrune\(1 - due\[1\]\);/,
  'early milestones cull their own keep-fraction');
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
assert.match(appHtml, /<option value="0\.3">Trim 30%<\/option>/,
  'the trims run 5% to 30%');
assert.match(app, /st\.softprune = readSoftprune\(\);/,
  'the selection persists into settings');
assert.match(appHtml, /<option value="custom">Custom…<\/option>/,
  'the gear panel offers a custom trim');
assert.match(appHtml, /<input id="set-softprune-custom" type="number" min="1" max="50"/,
  'a custom trim reads 1–50%');
assert.match(app, /showSoftprune\(st\.softprune\);/,
  'a stored custom fraction restores into the custom input');
assert.match(app, /trainerOpts\.softPrune = \{ ratio: parseFloat\(st\.softprune\), every: 3000, until: 0\.85 \};/,
  'fresh runs map it to trainer options');
assert.match(app, /<button class="tbtn-sm" id="t-prune"/,
  'the training dock carries a manual prune button');
assert.match(app, /\$\('t-prune'\)\.addEventListener\('click', showPrunePicker\);/,
  'the button opens the ratio picker');
assert.match(app, /async function runPruneAndContinue\(ratio = 0\.5\) \{/,
  'the flow prunes to the picked ratio, default half');
assert.match(app, /id="prunepick-pick"/,
  'the manual picker card offers the ratios');
assert.match(app, /<input id="prunepick-custom" type="number" min="5" max="95"/,
  'the manual picker offers a custom 5–95% keep');
assert.match(app, /data-custom="1">Keep %<\/button>/,
  'the custom keep runs the same prune flow');
assert.match(app, /<input id="cap-custom" type="number" min="5" max="95"/,
  'the ceiling card offers a custom 5–95% keep');
assert.match(app, /function readKeepCustom\(id\) \{/,
  'custom keeps clamp before the flow runs');
assert.match(app, /data-ratio="0\.75">Prune to ¾<\/button>/,
  'the ceiling card offers pruning to three quarters');
assert.match(app, /data-ratio="0\.5">Prune to ½<\/button>/,
  'the ceiling card offers pruning to half');
assert.match(app, /data-ratio="0\.25">Prune to ¼<\/button>/,
  'the ceiling card offers pruning to a quarter');
assert.match(app, /id="cap-keep">Keep training at cap<\/button>/,
  'the ceiling card offers staying put');

// ---- the browser mirrors it ------------------------------------------------
assert.match(browserHtml, /<select id="opt-softprune">/,
  'the browser settings offer soft prune');
assert.match(browserHtml, /<option value="0\.3">Trim 30%<\/option>/,
  'the trims run 5% to 30%');
assert.match(browser, /softprune: readSoftprune\(\),/,
  'the browser reads the setting');
assert.match(browserHtml, /<input id="opt-softprune-custom" type="number" min="1" max="50"/,
  'the browser offers a custom 1–50% trim');
assert.match(browserHtml, /<button id="btn-prune" class="btn" disabled>Prune\/Grow<\/button>/,
  'the run row carries a prune/grow button');
assert.match(browser, /ui\.prune\.addEventListener\('click', showPrunePicker\);/,
  'the button opens the ratio picker');
assert.match(browser, /async function runPruneContinue\(ratio = 0\.5\) \{/,
  'the flow prunes to the picked ratio, default half');
assert.match(browser, /mk\('Prune to ¾', 'btn', \(\) => go\(0\.75\)\)/,
  'the ceiling prompt offers pruning to three quarters');
assert.match(browser, /mk\('Prune to ½', 'btn accent', \(\) => go\(0\.5\)\)/,
  'the ceiling prompt offers pruning to half');
assert.match(browser, /mk\('Prune to ¼', 'btn', \(\) => go\(0\.25\)\)/,
  'the ceiling prompt offers pruning to a quarter');
assert.match(browser, /mk\('Keep training at cap', 'btn', \(\) => \{/,
  'the ceiling prompt offers staying put');

// ---- prune/grow tandem: milestones, horizon stop, boosted regrowth --------
assert.match(session, /\[\[1000, 0\.9\], \[2000, 0\.9\], \[3000, 0\.9\], \[4000, 0\.9\], \[5000, 0\.9\], \[6000, 0\.9\]\]/,
  'early milestones trim 10% every 1000 iterations through 6000');
assert.match(session, /trainer\.iter - at < \(spm\.every \?\? 3000\)/,
  'a continued run never replays old milestones back-to-back');
assert.match(session, /trainer\.iter <= \(sp\.until \?\? 0\.85\) \* this\._maxIters\(\)/,
  'tandem pruning stops at 85% of the run');
assert.match(session, /trainer\.iter <= \(spm\.until \?\? 0\.85\) \* this\._maxIters\(\)/,
  'milestone pruning stops at 85% of the run');
assert.match(session, /trainerOpts = this\._withTandemBoost\(trainerOpts\);/,
  'every boot (local, remote, worker) recomputes the tandem boost');
assert.match(session, /const g1 = \(1 \+ g0\) \/ \(1 - sp\.ratio\) - 1;/,
  'the per-refine boost holds the no-prune trajectory against the cull');
assert.match(session, /kind: 'tandem-regrow', iter: trainer\.iter/,
  'the bonus growth pass surfaces as an event');
assert.match(session, /async growNow\(\) \{/,
  'the session serves a manual growth pass');
assert.match(app, />Prune\/Grow<\/button>/,
  'the training dock button offers prune and grow');
assert.match(app, /data-grow="1">Grow now<\/button>/,
  'the manual picker offers a growth pass');
assert.match(app, /async function runGrowNow\(\) \{/,
  'the flow triggers one immediate refine');
assert.match(browser, /mk\('Grow now', runGrowNow\)/,
  'the browser picker offers a growth pass');
assert.match(browser, /mk\('Keep %', goCustom\)/,
  'the browser picker offers a custom keep');
assert.match(browser, /mk\('Prune to %', 'btn', goCustom\)/,
  'the browser ceiling prompt offers a custom keep');

console.log('prune-and-continue + soft-prune wiring: passed');
