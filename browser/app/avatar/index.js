// index.js — avatar mode, the app's one import. Lazy-loaded when the box on
// the start-training card is ticked; the app keeps two touch points:
//
//   prepareCapture(frames, source, ui)  on Start training, before the session
//       exists: cuts the person out (matte stage) and asks "is this the
//       person?". Resolves the manifest to hang on the capture record, or
//       null when the user says it is not an avatar (the run continues as a
//       plain scene). It runs HERE, not when the video is read, so the choice
//       sits on the start-training card next to the other settings and no
//       minutes of matting are spent before the run is actually wanted
//       (the user, 2026-09-18).
//   trainingOptions(manifest)            what the session needs for a
//       masked, avatar-sized run.
//   afterTraining(...)                   the remaining stages (landmarks,
//       face pass, body fit, bind, publish) — see runner.js.
//
// Everything avatar-specific lives under app/avatar/; src/ stays generic.
import { newManifest, setStage } from './manifest.js';
import { solveTierOpts } from '../../src/sfm/sfm.js';
import { cutoutsView } from './ui/cutouts.js';

export { isAvatar, STAGES, STAGE_LABEL, nextStage, setStage } from './manifest.js';
export { afterTraining } from './runner.js';
export { addPersonCrops } from './crops.js';
export { faceSeedPoints, faceSeedGaussians } from './faceseed.js';
export { headSeedGaussians } from './headseed.js';   // the whole head from the fitted body model, protected from relocation   // a dense seed on the face mesh, next to the sparse cloud   // native windows of the person as extra cameras, between solve and seed

export async function prepareCapture(frames, source, { card, flash = () => {}, log = (m) => console.log('[avatar]', m), progress = () => {} }) {
  const meter = (title, sub) => {
    card.innerHTML = `
      <div class="vid-head"><b>${title}</b><span class="prep-sub" id="avp-sub">${sub}</span></div>
      <div class="prep-meter"><i id="avp-bar" style="width:0%"></i></div>
      <canvas class="prep-cut" id="avp-cut" width="640" height="640"></canvas>`;
  };
  meter('Cutting the person out', 'loading the matting model …');
  // the frame just cut, shown while the matte runs (photo x matte over the panel colour);
  // one draw at a time — a slow decode never queues behind the next frame
  let drawing = false;
  const showCut = async (f) => {
    const cv = card.querySelector('#avp-cut'); if (!cv || drawing || !f || !f.mask) return;
    drawing = true;
    try {
      const [bmp, mask] = await Promise.all([createImageBitmap(f.source), createImageBitmap(f.mask)]);
      const h = 640, w = Math.round((bmp.width / bmp.height) * h);   // drawn at 2x, shown 320 px tall (the user: at least double, 2026-09-15)
      if (cv.width !== w) { cv.width = w; cv.style.width = `${w / 2}px`; }
      const tmp = new OffscreenCanvas(w, h); const t = tmp.getContext('2d');
      t.drawImage(bmp, 0, 0, w, h); t.globalCompositeOperation = 'destination-in';
      const m = new OffscreenCanvas(w, h); const mg = m.getContext('2d'); mg.drawImage(mask, 0, 0, w, h);
      const md = mg.getImageData(0, 0, w, h); const ad = mg.createImageData(w, h);
      for (let p = 0; p < w * h; p++) ad.data[p * 4 + 3] = md.data[p * 4];
      mg.putImageData(ad, 0, 0); t.drawImage(m, 0, 0);
      const g = cv.getContext('2d'); g.clearRect(0, 0, w, h); g.drawImage(tmp, 0, 0);
      bmp.close(); mask.close(); cv.classList.add('on');
    } catch (e) { /* preview only */ } finally { drawing = false; }
  };
  const manifest = newManifest(source);
  setStage(manifest, 'matte', { status: 'running' });
  let res;
  try {
    const { run } = await import('./stages/matte.js');
    res = await run(frames, {
      log,
      onProgress: (d, t, f) => {
        const bar = card.querySelector('#avp-bar'), sub = card.querySelector('#avp-sub');
        if (bar) bar.style.width = `${(d / t) * 100}%`;
        if (sub) sub.textContent = `${d} / ${t} frames`;
        progress(d, t);   // the dock's own beat, at the top of the window
        showCut(f);
      },
    });
  } catch (e) {
    log(`matte failed: ${e.message || e}`);
    flash(`Could not cut the person out (${e.message || e}) — continuing as a scene.`, 8000);
    return null;
  }
  setStage(manifest, 'matte', { status: 'done', ...res });
  if (res.coverage < 0.02) {
    flash('Almost nothing was recognised as a person in these frames — continuing as a scene.', 8000);
    for (const f of frames) delete f.mask;
    return null;
  }
  // The cut-outs used to be a gate: "Looks right / Not a person". The person
  // was chosen one screen earlier, on the start-training card, and the user
  // asked for no confirmations once a run has started (2026-09-18) — so the
  // strip is SHOWN and the run goes straight on into the solve. A matte that
  // finds nobody still bails out on its coverage, above.
  cutoutsView(card, frames, res);
  return manifest;
}

// The avatar's splat ceiling. `cap` is an UP-FRONT allocation in the trainer
// (min(seed x capMult, maxSplats)), so "no cap" means a ceiling high enough never to
// bind, not the absence of one — an unbounded value is not available at any price.
// Two costs scale with it, both allocated whether or not the model grows into them:
// at SH degree 3 a capped splat is ~1.2 kB of per-splat buffers, and the tile-entry
// budget is maxSplats x 24 pairs x 8 B = ~192 B/splat. So 2M is ~2.8 GB.
//
// The ceiling is clamped at 4M because the allocation is paid TWICE: the cut stage
// builds a second session while the source is still resident (nothing disposes it),
// so 4M is ~11.4 GB across the two, plus targets — the practical limit of a 16 GB
// card. 8M would survive its first createBuffer and then lose the device at the cut,
// after the whole training run had finished. Worst possible place to fail.
//
// WHEN IT BINDS: the app's avatar run grows +5 % per refine (growRate 0.05, the MCMC
// set) every ~550 iters (mcmcActive is on by default -> refineEvery 500; the session's
// own 2500 / 0.15 defaults are NOT what runs), until 0.75 x the horizon. Measured on
// Tom's 65 frames (2026-09-17, scratch/diag_tom_console.log, 20k horizon): 100k seed
// -> 339,504 live at the last growth step (iter 14,867). Extrapolated to 30k (41
// steps): ~740k, so the old 600k held only the last few steps of a 30k run and bit
// hard on anything longer (100k: 5.8M nominal). Read the cadence off a log before
// doing this arithmetic again — two people got it wrong from the defaults.
//   ?maxsplats=N  the ceiling, clamped to [100k, 4M]
//   ?capmult=M    the growth multiplier off the seed
const AV_Q = (k) => (typeof location !== 'undefined' ? new URLSearchParams(location.search).get(k) : null);
const AV_MAX_SPLATS = Math.min(4000000, Math.max(100000, +AV_Q('maxsplats') || 2000000));
const AV_CAP_MULT = Math.max(2, +AV_Q('capmult') || 24);

/** Session/trainer options for the masked run. The session turns the masks
 *  into the random-background target, the seed filter and the hull on its
 *  own; here only the sizing differs from a scene: a person is one subject,
 *  not a room. */
export function trainingOptions(manifest, { iters = 30000, shHorizontal = false, av = null } = {}) {
  // av: the avatar specifics as toggles (app settings.av; every key on when absent). With all of
  // them off the session and trainer options are the app's scene defaults, so an avatar run solves
  // and trains exactly like a plain run — only the stages after training differ.
  const on = (k) => !av || (av[k] !== false && av[k] !== 'off');
  return {
    budget: on('budget'),   // read by the app: the 1.1 GB decode budget
    // the precise solve tier: an orbit around a person is a small, low-texture
    // scene, and the quick/standard tiers collapsed a 1080p orbit into a
    // rotation-only solution once (2026-09-13) — nothing downstream survives that
    // the person trains as part of the ROOM (maskTraining false: the mattes
    // stay for the hull and the cut after the face pass) — the masked recipe
    // gave needle ratio 344 vs 15 for the same clip trained whole (2026-09-14b)
    // horizontal-only SH is a TOGGLE, off by default (the user, 2026-09-15: settings 'Horizontal SH (avatar)' or ?shup=1):
    // an orbit never looks down at a person, so the vertical colour variation was
    // unconstrained — blotches from above; the highlight still turns with the walk-around.
    // viewers and the client evaluate SH on the full direction, so 'on' needs the export refit to ship faithfully.
    // ?sfmall=1: the focal search on every image even above 120 frames (experiment, 2026-09-15)
    session: { evalSplit: 0, ...(on('seed') ? { initTarget: 100000 } : {}), sfm: { ...(on('tier') ? solveTierOpts('precise') : {}), ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('sfmall') === '1' ? { searchSubset: false } : {}), ...(+AV_Q('pairiters') > 0 ? { pairIters: +AV_Q('pairiters') } : {}), ...(AV_Q('pairdebug') === '1' ? { pairDebug: true } : {}) }, maskTraining: false, shHorizontal: shHorizontal || (typeof location !== 'undefined' && new URLSearchParams(location.search).get('shup') === '1') },
    // NO FLAT CEILING (the user, 2026-09-17): see AV_MAX_SPLATS above for the cost, the
    // 4M clamp and the growth arithmetic that says which runs this actually frees.
    // The 09-14 note said lifting 600k -> 1M only grew low-opacity splats the export
    // pruned (package 108.9k either way) — that predates opaRegUntil 0.5 (09-15p),
    // which stops the very pressure that was dimming them. Retest, don't assume.
    // ?camopt=1 (experiment, 2026-09-14): photometric pose refinement of every camera
    // during training — Filip's front views ghost (the head moved between the far
    // start and the close-up end of the orbit); the target is per-window pose
    // freedom for the crops, this is the first check that the mechanism helps
    // ?avsh=N (experiment): the avatar run's SH degree (the app's ?sh= is not a URL switch)
    // opacity pressure 0.004 for a person (trainer default 0.01): the pressure dims what
    // the loss does not defend, and on an avatar that is the face — at 0.003 the visible
    // face splats doubled at smaller sizes, at 0 the skin went waxy (Tom 30k, 2026-09-15d)
    // needle term 0.03 (2026-09-15f): needles 5.3 % -> 1.5 % on the face, discs and the
    // visible count untouched — the one shape term with no measured cost
    // the opacity pressure stops at half the run (2026-09-15p): it prunes while the model
    // grows, as a per-iteration pull it scaled with the run length — 30k: visible face
    // 1,383 -> 2,700; 100k: 926 -> 3,370, half-size splats, readable skin
    trainer: { maxSplats: AV_MAX_SPLATS, capMult: AV_CAP_MULT, ...(on('opa') ? { opacityReg: 0.004, opaRegUntil: 0.5 } : {}), ...(on('needle') ? { needleReg: 0.03 } : {}), ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('avsh') != null ? { shDeg: +new URLSearchParams(location.search).get('avsh') } : {}), ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('camopt') ? { camOpt: true, ...(new URLSearchParams(location.search).get('camopt') === 'crop' ? { camOptOnly: 'crop' } : {}) } : {}),
      // ?needle=W[,T] (experiment, 2026-09-15): the needle regularizer — the longest axis over
      // the middle one beyond ratio T (default 3) is pulled in; discs stay ("every needle
      // destroys the illusion in a close-up")
      ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('needle') ? (() => { const [w, t] = new URLSearchParams(location.search).get('needle').split(',').map(Number); return { needleReg: w, ...(t > 1 ? { needleRatio: t } : {}) }; })() : {}),
      // ?orient=W (experiment, 2026-09-15): the orientation regularizer — discs turn to face
      // the cameras that see them (a disc seen edge-on draws a line; a quarter of Tom's
      // visible face splats were edge-on to the frontal camera)
      ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('orient') ? { orientReg: +new URLSearchParams(location.search).get('orient') } : {}),
      // ?blob=R (experiment, 2026-09-15): only train blobs — a hard clamp after every Adam step
      // keeps each splat's longest axis within R x its shortest (the user's rule: no needles, no
      // edge-on discs, a near-round volume never draws a line)
      ...(typeof location !== 'undefined' && +new URLSearchParams(location.search).get('blob') > 1 ? { blobRatio: +new URLSearchParams(location.search).get('blob') } : {}),
      // ?camlr=N (experiment, 2026-09-15): multiplier on the pose learning rates of ?camopt
      ...(typeof location !== 'undefined' && +new URLSearchParams(location.search).get('camlr') > 0 ? { camLr: +new URLSearchParams(location.search).get('camlr') } : {}),
      // ?opuntil=F (experiment, 2026-09-15): the opacity pressure stops at F x the run length
      ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('opuntil') != null ? { opaRegUntil: +new URLSearchParams(location.search).get('opuntil') } : {}),
      // ?opreg=N (experiment, 2026-09-15): the opacity pressure (trainer default 0.01) — the visible face density lever
      ...(typeof location !== 'undefined' && new URLSearchParams(location.search).get('opreg') != null ? { opacityReg: +new URLSearchParams(location.search).get('opreg') } : {}) },
    iters,
  };
}
