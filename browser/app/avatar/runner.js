// runner.js — walks the stages after training and shows them at the TOP of
// the window, where the camera solve's beats were.
//
// Each stage module exports { id, needs, run(ctx, manifest, hooks), review? }.
// The PROGRESS lives in the dock (ctx.mountDock): the nine stages as a beat
// row, the active stage's message, a meter — the same shape the solve uses.
// A centred card covered the very thing it was asking about — the face — and
// squeezed three photos and two buttons into 340 px (the user, 2026-09-18).
//
// NOTHING INTERRUPTS THE RUN. Once Start training is pressed the stages walk
// to the end on their own: the joints and the body-fit checkpoints only ever
// asked "does this look right?", and as the user put it on 2026-09-18, "it
// always did and there's no way to make it different".
//
// What they SHOWED is worth seeing, so a stage with a picture exports
// preview(ctx, manifest, hooks) and hands it to hooks.shots(node, caption):
// it goes onto ONE card under the dock and STAYS there. A stage with no
// picture of its own leaves the last one up — through the minutes of
// isolating and binding you keep looking at the joints, then at the body
// model, and the finished card carries that above its buttons. Nothing waits
// for a click except publish, which is an action (download the package, or put
// it on your account), not a confirmation.
//
// The runner skips what the manifest says is done. The app hands it the
// finished session and the capture record; it hands back the updated manifest
// (the app persists it).
import { STAGES, STAGE_LABEL, STAGE_BEAT, setStage, nextStage } from './manifest.js';

const LOADERS = {
  landmarks: () => import('./stages/landmarks.js'),
  facepass: () => import('./stages/facepass.js'),
  cut: () => import('./stages/cut.js'),
  facepolish: () => import('./stages/facepolish.js'),
  bodyfit: () => import('./stages/bodyfit.js'),
  bind: () => import('./stages/bind.js'),
  publish: () => import('./stages/publish.js'),
};


/** the stage sequence for the dock: the active beat wears the title style,
 *  the done ones fade back, a failed one turns red (.prep-stages in app.css) */
export function beatRow(manifest, active) {
  return STAGES.map((s) => {
    const st = manifest.stages[s] || {};
    const on = s === active ? '1'
      : st.status === 'done' || st.status === 'skipped' ? 'done'
        : st.status === 'failed' || st.status === 'rejected' ? 'failed' : '0';
    return `<span data-on="${on}" title="${STAGE_LABEL[s]}">${STAGE_BEAT[s]}</span>`;
  }).join('');
}

/** ctx: { session, frames, manifest, capture, log, flash, mount, mountDock,
 *  onSession } — onSession(newSession, stageId, result) is called whenever a
 *  stage replaces the model, so the app's viewer follows it. */
export async function afterTraining(ctx) {
  const { manifest, mount, mountDock, log = () => {}, flash = () => {} } = ctx;
  setStage(manifest, 'train', { status: 'done', iters: ctx.session.trainer.iter, splats: ctx.session.trainer.n, note: `${ctx.session.trainer.n.toLocaleString()} splats` });

  // ── the bar in the dock: all nine stages, the message, one meter ──────────
  const bar = document.createElement('div');
  bar.className = 'av-bar';
  bar.id = 'avbar';
  bar.innerHTML = `
    <div class="prep-stages av-beats" id="av-beats"></div>
    <div class="prep-sub" id="av-sub">the trained splat is the halfway point — joints, face, body, binding</div>
    <div class="prep-meter"><i id="av-bar-i" style="width:0%"></i></div>`;
  (mountDock || mount)(bar);
  const el = (id) => bar.querySelector(`#${id}`);
  let active = null;
  const beats = () => { el('av-beats').innerHTML = beatRow(manifest, active); };
  const say = (m) => { if (m != null) el('av-sub').textContent = m; };
  const meter = (frac) => {
    const i = Math.max(0, STAGES.indexOf(active));
    el('av-bar-i').style.width = `${((i + Math.max(0, Math.min(1, frac))) / STAGES.length) * 100}%`;
  };
  beats();

  // ── the card under the dock: the pictures, and publish's buttons ──────────
  // Born with the first picture and kept to the end of the run. Nothing on it
  // blocks; the × abandons the whole thing.
  let card = null;
  const openCard = () => {
    if (!card) {
      card = document.createElement('div');
      card.className = 'upcard av-card';
      card.id = 'avcard';
      card.innerHTML = `
        <button class="card-x" id="av-x" aria-label="Close">&times;</button>
        <div class="vid-head"><b>Your avatar</b></div>
        <div class="av-overlay" id="av-shots"></div>
        <p class="fine av-shotcap" id="av-shotcap"></p>`;
      card.querySelector('#av-x').onclick = () => { ctx.cancelled = true; closeCard(); act?.remove(); act = null; bar.remove(); };
      mount(card);
    }
    return card;
  };
  const closeCard = () => { card?.remove(); card = null; };

  // The action panel (publish): top right, under the view controls, NOT over
  // the avatar. The finished avatar has to be walked around and looked at —
  // "there should be no ui in front of it" (the user, 2026-09-18).
  let act = null;
  const openAct = () => {
    if (!act) {
      act = document.createElement('div');
      act.className = 'av-act';
      act.id = 'avact';
      act.innerHTML = '<div class="av-act-head">Your avatar</div><div id="av-body"></div>';
      mount(act);
    }
    return act;
  };

  const hooks = {
    progress: (d, t, msg) => { meter(t ? d / t : 0); say(msg); },
    body: () => openAct().querySelector('#av-body'),
    // a stage's picture: swapped in complete (no blink), captioned, and left up
    // until another stage has one of its own
    shots: (node, caption) => {
      const c = openCard();
      const box = c.querySelector('#av-shots'), cap = c.querySelector('#av-shotcap');
      box.replaceChildren(node);
      cap.textContent = caption || '';
      // no step counter here: it would name the picture's stage while the dock
      // above is already showing where the RUN is, and the two do not match —
      // the joints picture stays up while the body is being fitted
    },
    log, flash,
  };

  /** a stage's picture onto the card — never fatal, never waited on by anyone */
  const showPicture = async (stage, mod = null) => {
    try {
      const m = mod || (LOADERS[stage] ? await LOADERS[stage]() : null);
      if (m && m.preview) await m.preview(ctx, manifest, hooks);
    } catch (e) { log(`${stage} preview: ${e.message || e}`); }
  };

  // stages that ran before training (landmarks, for the head windows) are
  // already done when we get here: their picture goes up first, so the card is
  // not empty through the isolating
  for (const s0 of STAGES) {
    const st = manifest.stages[s0]; if (!st || !st.reviewPending) continue;
    delete st.reviewPending;
    if (st.status !== 'done' || ctx.cancelled) continue;
    const was = active; active = s0;
    await showPicture(s0);
    active = was; beats();
  }

  let s;
  while ((s = nextStage(manifest)) && !ctx.cancelled) {
    if (s === 'matte' || s === 'train') { setStage(manifest, s, { status: 'done' }); continue; }
    const loader = LOADERS[s];
    if (!loader) { setStage(manifest, s, { status: 'skipped', note: 'not built yet' }); continue; }
    if (s === 'facepolish' && !(typeof location !== 'undefined' && new URLSearchParams(location.search).get('facepolish'))) { setStage(manifest, s, { status: 'skipped', note: 'off' }); continue; }   // experiment: ?facepolish=1
    let mod;
    try { mod = await loader(); } catch (e) { setStage(manifest, s, { status: 'skipped', note: 'not built yet' }); beats(); continue; }
    setStage(manifest, s, { status: 'running' });
    active = s; beats(); say(`${STAGE_LABEL[s]} …`); meter(0);
    // the finished avatar is the thing to look at now: the pictures go, the
    // step bar goes (the user, 2026-09-18: "at the last stage you can remove
    // the top bar with the finished steps"), and publish's buttons open at the
    // top right instead of over the model. What publish has to say goes into
    // its own status line there.
    if (s === 'publish') { closeCard(); bar.remove(); }
    try {
      const wasSession = ctx.session;
      const res = await mod.run(ctx, manifest, hooks);
      setStage(manifest, s, { status: 'done', ...res });
      // isolate / face pass / face polish hand back a NEW model — the one that
      // will be bound and shipped. The viewer has to show that one, not the
      // room it was cut out of (the user, 2026-09-18).
      if (ctx.session && ctx.session !== wasSession) ctx.onSession?.(ctx.session, s, res);
      // bind hands over the rig binding: from here the avatar can be moved, and
      // the app shows it walking while the rest of the run finishes
      if (s === 'bind' && ctx.bindingBytes && ctx.rigGlb) ctx.onBound?.({ binding: ctx.bindingBytes, fit: ctx.fitJson, rigGlb: ctx.rigGlb });
      if (mod.preview) await showPicture(s, mod);
      beats(); say(manifest.stages[s].note || `${STAGE_LABEL[s]} — done`);
    } catch (e) {
      log(`${s} failed: ${e.message || e}`);
      setStage(manifest, s, { status: 'failed', note: String(e.message || e).slice(0, 80) });
      beats(); say(`${STAGE_LABEL[s]} failed — ${String(e.message || e).slice(0, 120)}`);
      flash(`${STAGE_LABEL[s]} failed: ${e.message || e}`, 9000);
      break;
    }
    ctx.persist?.(manifest);
  }
  if (!s && !ctx.cancelled) { active = null; beats(); meter(1); say(manifest.stages.publish?.note ? `done — ${manifest.stages.publish.note}` : 'done'); }
  return manifest;
}
