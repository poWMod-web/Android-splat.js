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

// ---- the trainer culls by rendered mass, on demand -------------------------
assert.match(trainer, /async softPrune\(ratioOverride = null, \{ skipRatchet = false \} = \{\}\) \{/,
  'the trainer serves a soft-prune call, with a milestone override and a ratchet bypass');
assert.match(trainer, /this\._pruneVotes = new Uint16Array\(0\);/,
  'each trainer opens with a clean vote sheet');
assert.match(trainer, /votes: this\._pruneVotes, votesNeeded: cfg\.votes \?\? 2,/,
  'every cull votes its candidates, two consecutive windows to die');
assert.match(trainer, /nvotes\[r\] = this\._pruneVotes\[i\];/,
  'survivors keep their vote history through the compaction');
assert.match(trainer, /if \(!skipRatchet\) this\._softPruneFloor = plan\.kept;/,
  'a bypassed cull neither holds nor sets the ratchet');
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
assert.match(remote, /async softPrune\(ratio = null, opts = null\) \{/,
  'the remote proxy forwards the ratchet bypass with n/iter like refine does');
assert.match(worker, /const r = await tr\.softPrune\(params\.ratio \?\? null, \{ skipRatchet: !!params\.skipRatchet \}\);/,
  'the worker forwards a milestone override and bypass to its trainer');

// ---- the session drives cadence, prompt, and prune-and-continue ------------
assert.match(session, /const pruned = await trainer\.softPrune\(\);/,
  'every tandem refine culls first');
assert.match(session, /tandem prune @/,
  'the cull rides inside the scheduled refine');
assert.match(session, /\[tandem: culled /,
  'the refine log names the cull it regrew');
assert.match(session, /pruned = await trainer\.softPrune\(1 - due\[1\], \{ skipRatchet: true \}\);/,
  'early milestones cull their own keep-fraction, ratchet bypassed');
assert.match(session, /kind: 'soft-prune', iter: trainer\.iter/,
  'culls surface as events');
assert.match(session, /if \(refineResult && refineResult\.capped && !this\._capAsked\) \{/,
  'a capped refine pauses for the ceiling question, once per ceiling');
assert.match(session, /kind: 'cap-hit', iter: trainer\.iter, splats: trainer\.n/,
  'the question reaches the UI as an event');
assert.match(session, /async pruneAndContinue\(ratio = 0\.5,/,
  'prune-and-continue defaults to half');
assert.match(session, /async pruneAndContinue\(ratio = 0\.5, \{ onStatus = null, onProgress = null \}/,
  'a pure prune takes no horizon extension');
assert.match(session, /this\.training = wasTraining;/,
  'a pure prune restores the run state instead of resuming');
assert.match(session, /resumed: wasTraining/,
  'the outcome reports whether training resumed');
const pruneBody = session.slice(session.indexOf('async pruneAndContinue'), session.indexOf('async growNow()'));
assert.ok(!pruneBody.includes('continueFor') && !pruneBody.includes('trainer.refine('),
  'a pure prune neither extends the horizon nor grows');
assert.match(session, /kind: 'prune-continue', iter, \.\.\.out/,
  'the outcome (counts, probe delta, horizon) reaches the UI');

// ---- the full app: setting, button, card -----------------------------------
assert.match(appHtml, /<select id="set-softprune">/,
  'the gear panel offers soft prune');
assert.match(appHtml, /<option value="0\.01">Trim 1%<\/option>/,
  'the trims start at 1%');
assert.match(appHtml, /<option value="0\.1">Trim 10%<\/option>/,
  'the trims top out at 10%');
assert.doesNotMatch(appHtml, /Trim (20|30)%/,
  'the old 20–30% trims are gone from the presets');
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
assert.match(app, /data-ratio="0\.75">Prune ¼<\/button>/,
  'the manual picker offers pruning a quarter');
assert.match(app, /data-ratio="0\.5">Prune ½<\/button>/,
  'the manual picker offers pruning half');
assert.match(app, /<input id="prunepick-custom" type="number" min="5" max="95"/,
  'the manual picker offers a custom 5–95% prune');
assert.match(app, /function readPruneCustom\(id\) \{/,
  'a custom prune amount converts to the keep fraction the flow wants');
assert.match(app, /data-custom="1">Prune %<\/button>/,
  'the custom prune runs the same prune flow');
assert.match(app, /if \(!e\.target \|\| e\.target\.tagName !== 'BUTTON'\) return;/,
  'tapping the custom input must not dismiss the picker');
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
const mileBody = session.slice(session.indexOf('early milestones (opt-in'), session.indexOf("the ceiling's own prompt"));
assert.ok(!mileBody.includes('trainer.refine(') && !mileBody.includes('tandem-regrow'),
  'milestones cull pure: no regrow, no ratchet');
assert.match(session, /async growNow\(\) \{/,
  'the session serves a manual growth pass');
assert.match(app, />Prune\/Grow<\/button>/,
  'the training dock button offers prune and grow');
assert.match(app, /data-grow="1">Grow now<\/button>/,
  'the manual picker offers a growth pass');
assert.match(app, /async function runGrowNow\(\) \{/,
  'the flow triggers one immediate refine');
const growBody = session.slice(session.indexOf('async growNow()'), session.indexOf('async finish()'));
assert.ok(!growBody.includes('softPrune'),
  'a manual growth pass never culls, even with soft prune on');

console.log('prune-and-continue + soft-prune wiring: passed');
