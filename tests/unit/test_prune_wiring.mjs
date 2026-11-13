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
  'the trainer serves a soft-prune call, with a programmatic ratio override and a ratchet bypass');
assert.match(trainer, /this\._pruneVotes = new Uint16Array\(0\);/,
  'each trainer opens with a clean vote sheet');
assert.match(trainer, /votes: this\._pruneVotes, votesNeeded: cfg\.votes \?\? 2,/,
  'every cull votes its candidates, two consecutive windows to die');
assert.match(trainer, /nvotes\[r\] = this\._pruneVotes\[i\];/,
  'survivors keep their vote history through the compaction');
assert.match(trainer, /if \(!skipRatchet\) this\._softPruneFloor = plan\.kept;/,
  'a bypassed cull neither holds nor sets the ratchet');
assert.match(trainer, /const ratio = ratioOverride \?\? cfg\.ratio \?\? 0\.1;/,
  'a programmatic override culls its own fraction instead of the configured trim');
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
  'the worker forwards an override and bypass to its trainer');
assert.match(trainer, /import \{ planSoftPrune, planDeadCull, chunkPlan/,
  'the trainer plans dead culls through the pure dead planner');
assert.match(trainer, /async pruneDead\(\{ votesNeeded = 2, minCull = 1024 \} = \{\}\) \{/,
  'the trainer serves a dead-splat cull with vote defaults');
assert.match(trainer, /if \(this\.opts\.softPrune\) return \{ culled: 0, n: this\.n \};/,
  'the dead cull yields when the tandem owns culling');
assert.match(trainer, /const \{ masses, logitOp, meanLogS \} = await this\._gatherMasses\(\);/,
  'both culls share one scoring gather');
assert.match(trainer, /this\._compactToPlan\(plan\);/,
  'both culls share one compaction');
assert.match(protocol, /'pruneDead',/,
  'the RPC contract lists the dead cull');
assert.match(worker, /case 'pruneDead': \{/,
  'the worker serves the dead cull');
assert.match(remote, /async pruneDead\(\) \{/,
  'the remote proxy forwards the dead cull');
assert.match(app, /e\.kind === 'dead-cull'/,
  'the UI marks dead culls on the chart');
assert.match(appHtml, /<option value="">Off \(dead only\)<\/option>/,
  'Off names what it still does');

// ---- the session drives cadence, prompt, and prune-and-continue ------------
assert.match(session, /const pruned = await trainer\.softPrune\(\);/,
  'every tandem refine culls first');
assert.match(session, /tandem prune @/,
  'the cull rides inside the scheduled refine');
assert.match(session, /\[tandem: culled /,
  'the refine log names the cull it regrew');
assert.match(session, /\} else if \(!sp && trainer\.iter <= 0\.85 \* this\._maxIters\(\)\) \{/,
  'soft-prune-off refines cull dead splats exactly where the tandem would run');
assert.match(session, /const dead = await trainer\.pruneDead\(\);/,
  'the off-mode cull asks the trainer for vote-confirmed dead rows');
assert.match(session, /kind: 'dead-cull', iter: trainer\.iter/,
  'dead culls surface as events');
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
assert.match(session, /const resume = \(\) => \{ if \(wasTraining\) \{ this\.training = false; this\.start\(\); \} \};/,
  'a pure prune restores the run state through start() (a paused run stays paused, a training one reschedules)');
assert.match(session, /resumed: wasTraining/,
  'the outcome reports whether training resumed');
const pruneBody = session.slice(session.indexOf('async pruneAndContinue'), session.indexOf('async growNow()'));
assert.match(pruneBody, /resume\(\);\n        this\._capAsked = false;/,
  'the success path resumes the loop it paused');
assert.match(pruneBody, /try \{ resume\(\); \} catch/,
  'a failed prune resumes onto the intact previous trainer instead of wedging paused');
assert.match(pruneBody, /the unpruned model is intact/,
  'a moved-seed reseed failure says the previous model survived (not "reload")');
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
assert.match(app, /<input id="prunepick-custom" type="number" min="1" max="99"/,
  'the manual picker offers a custom 1–99% prune');
assert.match(app, /function readPruneCustom\(id\) \{/,
  'a custom prune amount converts to the keep fraction the flow wants');
assert.match(app, /if \(!Number\.isInteger\(raw\) \|\| raw < 1 \|\| raw > 99\) return null;/,
  'invalid custom values must be rejected instead of falling back to 50%');
assert.match(app, /return 1 - raw \/ 100;/,
  'a valid 1–99 prune percentage converts to its complementary keep fraction');
assert.match(app, /customKeep == null[\s\S]*Enter a whole prune percentage from 1 to 99[\s\S]*return;/,
  'invalid custom input should keep the picker open, explain the range, and not prune');
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

// ---- prune/grow tandem: no milestones, horizon stop, boosted regrowth -----
assert.doesNotMatch(session, /milestones/,
  'no scheduled milestone trims remain');
assert.doesNotMatch(session, /_pruneFired/,
  'no milestone replay bookkeeping remains');
assert.match(session, /trainer\.iter <= \(sp\.until \?\? 0\.85\) \* this\._maxIters\(\)/,
  'tandem pruning stops at 85% of the run');
assert.match(session, /trainerOpts = this\._withTandemBoost\(trainerOpts\);/,
  'every boot (local, remote, worker) recomputes the tandem boost');
assert.match(session, /const g1 = \(1 \+ g0\) \/ \(1 - sp\.ratio\) - 1;/,
  'the per-refine boost holds the no-prune trajectory against the cull');
const deadAt = session.indexOf('} else if (!sp');
assert.ok(deadAt > 0, 'the off-mode dead branch exists');
const deadBody = session.slice(deadAt, deadAt + 800);
assert.ok(deadBody.includes('trainer.pruneDead()') && !deadBody.includes('trainer.refine('),
  'the dead cull asks only for dead rows and never refines');
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


// ---- failed boots publish nothing (the prune-failure cascade, 2026-11) ----
assert.match(session, /const next = await GSTrainer\.create\(\{ \.\.\.trainerOpts, gpu: this\.gpu \}\);/,
  'local boots build into a local, not onto the live trainer');
assert.match(session, /this\._setupTrainer\(next, boot\.cams, boot\.frames, boot\.maxW, boot\.maxH, boot\.radius\);/,
  'setup runs pre-swap against the local');
assert.match(session, /if \(this\._worker\) this\._terminateWorker\(\);   \/\/ the replacement is in: retire the previous core/,
  'only a fully booted replacement retires the previous core');
assert.match(session, /this\.trainer = next;/,
  'the live trainer swaps once, after setup completes');
assert.match(session, /terminate ONLY the replacement/,
  'a failed remote boot terminates only its own worker (the previous trainer stays live)');
assert.match(worker, /const next = await createTrainerImpl\(\{ \.\.\.trainerOpts, gpu \}\);/,
  'worker init builds into a local, not onto the published trainer');
assert.match(worker, /trainer = next;/,
  'worker init publishes only after setup (a retry reboots instead of hitting already-initialized)');
assert.match(trainer, /this\.ready = false;/,
  'a fresh trainer opens unready');
assert.match(trainer, /this\.ready = true;/,
  'setup completion flips the ready flag');
assert.match(trainer, /if \(!this\.ready\) throw new Error\('GSTrainer\.readGaussians: setup\(\) did not complete'\);/,
  'readback on a half-booted trainer fails loud, not as a GPU TypeError');
assert.match(trainer, /if \(!this\.ready\) throw new Error\('GSTrainer\.stepOnce: setup\(\) did not complete'\);/,
  'steps on a half-booted trainer fail loud, not as .length on a missing deck');
assert.match(session, /if \(trainer\.ready === false\) \{/,
  'the frame loop refuses a half-booted trainer loudly');
assert.match(session, /seed data was released after the first boot \(lowMem\) — the current run is intact, but it cannot reseed/,
  'a refused reseed says the run trains on (prune keeps its never-bricks promise)');
assert.match(session, /seed data was released after the first boot \(lowMem\) — reload to train again/,
  'recover-after-device-loss still says reload (its trainer is dead, not intact)');
// pixel-less prune: the worker (no row-upload RPC) refuses before the
// expensive merge-down; the local trainer applies it in place instead.
// seedFrom's guard still runs before it touches this.model (a refused
// reseed leaves the session bit-identical for the resume in the prune catch)
const pruneFn = session.slice(session.indexOf('async pruneAndContinue('), session.indexOf('async recover()'));
assert.ok(pruneFn.includes('if (noPixels && this._remote)'),
  'pixel-less prune on a worker refuses (rows cannot cross the RPC)');
assert.ok(pruneFn.indexOf('const noPixels = this._seedPixelsReleased()') < pruneFn.indexOf('simplifyGaussians('),
  'the pixel check runs before the merge-down burns phone CPU');
assert.ok(pruneFn.includes('await trainer.applyPrunedModel('),
  'pixel-less prune on a local trainer applies in place, no reseed');
// the in-place apply: refine-patch for params+moments, direct SH upload,
// then the shared count-shrink tail (votes reset, ratchet floor held)
assert.match(trainer, /applyPrunedModel\(\{ data, n: kept, sh = null, shK = 0 \}\) \{/,
  'the trainer serves an in-place pruned-model apply');
assert.ok(trainer.includes('this._refinePatch(touched, data);'),
  "the apply patches params and zeroes moments through growth's own kernel");
assert.ok(trainer.includes('this.device.queue.writeBuffer(this.bufSH, 0, sh.subarray(0, kept * this.shK * 3));'),
  'merged SH uploads direct (recomputed values, not a permutation)');
assert.ok(trainer.includes('this._pruneVotes = new Uint16Array(kept);'),
  'merged rows start with no vote history');
assert.ok(trainer.includes('this._softPruneFloor = kept;'),
  'the soft-prune ratchet holds the hard-pruned floor');
const seedFromFn = session.slice(session.indexOf('async seedFrom('), session.indexOf('async pruneAndContinue('));
assert.ok(seedFromFn.indexOf('_seedPixelsReleased(opts.viewOnly)') < seedFromFn.indexOf('this.model = '),
  'seedFrom guards before it touches the model');
assert.match(session, /trainerOpts = \{ \.\.\.trainerOpts, useStats: true \};/,
  'soft-prune-off boots accumulate the mass window the dead cull scores');
assert.match(app, /training on with the unpruned model\./,
  'a failed prune tells a resumed run it kept training');

console.log('prune-and-continue + soft-prune wiring: passed');
