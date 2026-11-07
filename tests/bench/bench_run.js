// Benchmark matrix runner: one (dataset, cycle-budget) cell per page load.
// Release-default trainer (MCMC set + auto splat budget = cycles x 35,
// capMult 8, shDeg 3, refineEvery 500), eval8 held-out PSNR, train-only
// timing. ?set=truck&iters=20000 — posts bench_<set>_<iters>.json.
// The bar pano cell rebuilds the split rig-aware (hold every 8th PANO's six
// faces, score the four yaw faces — the stock face split leaks centers).
import { createSession } from '/src/index.js';
import { extractSharpFrames } from '/src/io/video.js';
import { solveTierOpts } from '/src/sfm/sfm.js';

const Q = new URLSearchParams(location.search);
const SET = Q.get('set');
const ITERS = +(Q.get('iters') || 20000);
const TAG = `${SET}_${ITERS}` + (Q.has('classic') ? '_classic' : '')
  + (Q.get('dilate') ? `_dil${Q.get('dilate')}` : '')
  + (Q.get('aniso') != null ? `_ar${Q.get('aniso')}` : '')
  + (Q.get('ssim') ? `_ssim${Q.get('ssim')}` : '')
  + (Q.get('maxsplats') ? `_cap${Q.get('maxsplats')}` : '') + (Q.get('res') ? `_r${Q.get('res')}` : '') + (Q.has('evalsplit') ? `_es${Q.get('evalsplit')}` : '') + (Q.get('capmult') ? `_cm${Q.get('capmult')}` : '')
  + (Q.get('entriescap') ? `_ec${Q.get('entriescap')}` : '')
  + (Q.get('minscale') ? `_ms${Q.get('minscale')}` : '')
  + (Q.get('comp') != null ? `_c${Q.get('comp')}` : '')
  + (Q.get('regvis') ? '_rv' : '') + (Q.get('opafloor') ? `_of${Q.get('opafloor')}` : '')
  + (Q.get('refv2') ? '_rv2' : '') + (Q.get('errdon') ? '_ed' : '') + (Q.get('splitv2') ? '_sv2' : '')
  + (Q.get('growrate') ? `_gr${Q.get('growrate')}` : '') + (Q.get('movecap') ? `_mc${Q.get('movecap')}` : '')
  + (Q.get('refevery') ? `_re${Q.get('refevery')}` : '') + (Q.get('growuntil') ? `_gu${Q.get('growuntil')}` : '')
  + (Q.get('relocuntil') ? `_ru${Q.get('relocuntil')}` : '') + (Q.get('reloctaper') ? `_rt${Q.get('reloctaper')}` : '') + (Q.get('lrdecay') ? `_ld${Q.get('lrdecay')}` : '') + (Q.get('lrdecaymax') ? `_ldm${Q.get('lrdecaymax')}` : '')
  + (Q.get('poslr') ? `_pl${Q.get('poslr')}` : '') + (Q.get('shramp') === '0' ? '_nsr' : '')
  + (Q.get('opareg') ? `_or${Q.get('opareg')}` : '') + (Q.get('scalereg') ? `_sr${Q.get('scalereg')}` : '')
  + (Q.get('oregref') ? `_orr${Q.get('oregref')}` : '') + (Q.get('oregrefmax') ? `_orm${Q.get('oregrefmax')}` : '')
  + (Q.get('donor') ? `_dw${Q.get('donor')}` : '')
  + (Q.get('deadthr') ? `_dt${Q.get('deadthr')}` : '') + (Q.get('poolmin') != null ? `_pm${Q.get('poolmin')}` : '')
  + (Q.get('opadecay') ? `_od${Q.get('opadecay')}` : '') + (Q.get('ratiocap') ? `_rc${Q.get('ratiocap')}` : '')
  + (Q.get('econ') ? `_e${Q.get('econ')}` : '') + (Q.get('deadtiny') ? '_dtn' : '') + (Q.get('minutes') ? `_m${Q.get('minutes')}` : '') + (Q.get('evalmin') ? `_ev${Q.get('evalmin')}` : '')
  + (Q.has('classicsolve') ? '_cs' : '') + (Q.get('featres') ? `_fr${Q.get('featres')}` : '') + (Q.get('feats') ? `_nf${Q.get('feats')}` : '') + (Q.get('octave') ? `_oc${Q.get('octave')}` : '') + (Q.get('peak') ? `_pk${Q.get('peak')}` : '') + (Q.get('solve') ? `_sv${Q.get('solve')}` : '') + (Q.get('baratio') != null ? `_br${Q.get('baratio')}` : '') + (Q.get('bapts') != null ? `_bp${Q.get('bapts')}` : '') + (Q.get('pairworkers') === '0' ? '_npw' : '') + (Q.get('exiffocal') === '0' ? '_nxf' : '') + (Q.get('focals') ? `_fs${Q.get('focals').replace(/[^0-9]/g, '')}` : '') + (Q.get('bracket') === '0' ? '_nbk' : '') + (Q.get('inittrials') != null ? `_it${Q.get('inittrials')}` : '') + (Q.get('initpair') ? `_ip${Q.get('initpair').replace(',', '')}` : '') + (Q.get('searchsub') === '0' ? '_nss' : '')
  + (Q.get('compact') === '0' ? '_ncp' : '') + (Q.get('gspread') ? `_gs${Q.get('gspread')}` : '') + (Q.get('gbatch') ? `_gb${Q.get('gbatch')}` : '') + (Q.get('usestats') ? '_us' : '') + (Q.get('camgrads') ? '_cg' : '') + (Q.get('rectbin') ? '_rb' : '') + (Q.get('ipf') ? `_ipf${Q.get('ipf')}` : '') + (Q.get('gzskip') ? '_gz' : '') + (Q.get('pvec') ? '_pv' : '') + (Q.get('sgagg') ? '_sg' : '') + (Q.get('tilegrad') === '0' ? '_ntg' : '') + (Q.get('frommodel') ? '_fm' : '') + (Q.get('shdeg') != null ? `_sh${Q.get('shdeg')}` : '') + (Q.get('aspect') ? '_asp' : '') + (Q.get('asplr') ? `_al${Q.get('asplr')}` : '') + (Q.get('sfmaspect') === '0' ? '_nsa' : Q.get('sfmaspect') ? '_sa' : '') + (Q.get('lockk') ? '_lk' : '') + (Q.get('pairinl') ? `_pi${Q.get('pairinl')}` : '') + (Q.get('pairinladj') ? `_pa${Q.get('pairinladj')}` : '') + (Q.get('relax') === '0' ? '_nrx' : '')
  + (Q.get('video') ? `_v${Q.get('video').split('/').pop().replace(/.[^.]+$/, '')}` : '') + (Q.get('vidmode') ? `_vm${Q.get('vidmode')}` : '') + (Q.get('vidmax') ? `_vx${Q.get('vidmax')}` : '') + (Q.get('vidoverlap') ? `_vo${Q.get('vidoverlap')}` : '') + (Q.get('vidshots') ? `_vs${Q.get('vidshots')}` : '') + (Q.get('vidmaxgap') ? `_vg${Q.get('vidmaxgap').replace('.', '')}` : '') + (Q.get('vidpick') ? `_vp${Q.get('vidpick')}${(Q.get('vidfps') || '').replace('.', '')}` : '') + (Q.get('vidhold') ? '_vh' : '') + (Q.get('vidholdexcl') ? `_hx${Q.get('vidholdexcl').replace('.', '')}` : '')
  + (Q.get('masks') ? '_msk' : '') + (Q.get('maskguard') ? `_mg${Q.get('maskguard')}` : '') + (Q.get('randbg') ? `_rbg${Q.get('randbg')}` : '') + (Q.get('hullkeep') ? `_hk${Q.get('hullkeep')}` : '') + (Q.get('hullres') ? `_hr${Q.get('hullres')}` : '') + (Q.get('covw') ? `_cw${Q.get('covw')}` : '') + (Q.get('covs') ? `_cs${Q.get('covs')}` : '') + (Q.get('maskcut') ? `_mc${Q.get('maskcut')}` : '') + (Q.get('maskbg') ? `_mb${Q.get('maskbg')}` : '')
  + (Q.get('dir') ? `_d${Q.get('dir')}` : '') + (Q.get('tag') ? `_${Q.get('tag')}` : '')   // free suffix: e.g. the recon source, which no flag names
  + (Q.get('seed') ? `_s${Q.get('seed')}` : '');
const t0 = Date.now();
const logEl = document.getElementById('log');
const post = (name, body) => fetch(`/scratch/${name}`, { method: 'POST', body });
const St = { set: SET, iters: ITERS, phase: 'boot', t0 };
const say = async (phase, extra = {}) => {
  Object.assign(St, { phase, ...extra, ts: Date.now(), min: +((Date.now() - t0) / 60000).toFixed(1) });
  console.log('[BENCH]', SET, ITERS, phase, JSON.stringify(extra));
  logEl.textContent += `\n${phase} ${JSON.stringify(extra)}`;
  try { await post(`bench_${TAG}_status.json`, JSON.stringify(St)); } catch {}
};

// ?gradcheck=1: finite-difference validation of the analytic gradients on the
// small rigs (parameters on both accumulation paths, camera/exposure/aspect,
// SH) — no dataset, no session. Posts bench_<TAG>_gradcheck.json and stops.
if (Q.has('gradcheck')) {
  await say('gradcheck');
  const gc = await import('/src/gs/gradcheck.js');
  const out = {};
  const runs = [
    ['paramsTile', () => gc.gradCheckSmall({ trainer: { tileGrad: true } })],
    ['paramsGlobal', () => gc.gradCheckSmall({ trainer: { tileGrad: false } })],
    ['pose', () => gc.gradCheckPose()],
    ['sh3', () => gc.gradCheckSH(3)],
  ];
  for (const [name, fn] of runs) {
    try { const r = await fn(); out[name] = { ok: !!r.ok, summary: r.summary || r.sh || null, worst: r.worstSamples || (r.results && r.results.filter((x) => x.relErr >= 0.05)) || null }; }
    catch (e) { out[name] = { ok: false, error: String((e && e.message) || e) }; }
    console.log('[GRADCHECK]', name, JSON.stringify(out[name]).slice(0, 400));
  }
  out.ok = Object.values(out).every((r) => r.ok !== false);
  await post(`bench_${TAG}_gradcheck.json`, JSON.stringify(out));
  await say('DONE', { gradcheck: out.ok, parts: Object.fromEntries(Object.entries(out).filter(([k]) => k !== 'ok').map(([k, v]) => [k, v.ok])) });
  await new Promise(() => {});   // the runner tears the page down on DONE
}

const SETS = {
  // 12 views: eval8 starves training (10 views) — score the classic single
  // mid-sequence holdout instead, like every historical synthetic number
  synthetic: { dir: 'synthetic', holdout1: true, names: () => Array.from({ length: 12 }, (_, i) => `synthetic_${String(i).padStart(2, '0')}.png`) },
  truck:     { dir: 'truck', res: 979, names: () => Array.from({ length: 251 }, (_, i) => `${String(i + 1).padStart(6, '0')}.jpg`) },
  camping:   { dir: 'camping', names: () => Array.from({ length: 113 }, (_, i) => `frame_${String(i + 1).padStart(5, '0')}.jpg`) },
  train:     { dir: 'train', names: () => Array.from({ length: 301 }, (_, i) => `${String(i + 1).padStart(5, '0')}.jpg`) },
  playroom:  { dir: 'playroom', list: true },
  // person orbit (LisaAvatar.mov, 189 sharp-selected frames at 900x1600).
  // ?masks=1 attaches data/lisa/masks/<name>.png (the RVM person matte) so the
  // loss only sees the person. The photographs themselves stay whole, so the
  // solver still registers off the room.
  lisa:      { dir: 'lisa', list: true, masks: 'masks' },
  lisaface:  { dir: 'lisaface', list: true, masks: 'masks' },   // lisa + native-4K face crops (tests/bench/face_crops.py); train with ?gtrecon=&frommodel=
  bicycle:   { dir: 'bicycle', list: true },
  garden:    { dir: 'garden', list: true },
  statue:    { dir: 'statue_ka', list: true },
  charleston: { dir: 'charleston', names: () => Array.from({ length: 502 }, (_, i) => `frame_${String(i + 1).padStart(3, '0')}.jpg`) },   // 4K walk, 502 frames from the server extractor (2/s) — density baseline for the video extractor
  video:     { dir: '', video: true },   // ?set=video&video=/data/downloads/x.webm [&vidmode=element&vidmax=N]: frames come from the extractor   // user's 34-photo statue set (8064x6048), registration probe 2026-09-08
  bar360:    { dir: 'bar360/images', res: 912, pano: true, names: () => {
    const n = [];
    for (let i = 0; i <= 150; i += 2) n.push(`0_${String(i).padStart(4, '0')}.jpg`);
    for (let i = 0; i <= 50; i += 2) n.push(`1_${String(i).padStart(4, '0')}.jpg`);
    return n;
  } },
};

try {
  const cfg = SETS[SET];
  if (Q.get('dir')) cfg.dir = Q.get('dir');   // a resampled copy of the set (e.g. truck_sq: square pixels for the COLMAP camera)
  if (!cfg) throw new Error(`unknown set ${SET}`);
  let names, evalFrames = null;
  const files = [];
  if (cfg.video) {
    // ---- video input: fetch the file, run the sharp-frame extractor, treat the winners as photos ----
    const vurl = Q.get('video');
    if (!vurl) throw new Error('set=video needs ?video=/data/...');
    await say('fetch', { video: vurl });
    const r = await fetch(vurl);
    if (!r.ok) throw new Error(`missing ${vurl}`);
    const vfile = new File([await r.blob()], vurl.split('/').pop());
    const vT = Date.now();
    const ex = await extractSharpFrames(vfile, {
      engine: Q.get('vidmode') || 'auto',
      ...(Q.get('vidmax') ? { maxFrames: +Q.get('vidmax') } : {}),
      ...(Q.get('vidoverlap') ? { overlap: +Q.get('vidoverlap') } : {}),
      ...(Q.get('vidshots') ? { shots: Q.get('vidshots') } : {}),
      ...(Q.get('vidmaxgap') ? { maxGapSec: +Q.get('vidmaxgap') } : {}),
      ...(Q.get('vidpick') ? { pick: Q.get('vidpick') } : {}),   // vidpick=uniform&vidfps=3: fixed-rate control with no sharpness selection
      ...(Q.get('vidfps') ? { uniformFps: +Q.get('vidfps') } : {}),
      ...(Q.get('vidhold') ? { forceTimes: Q.get('vidhold').split(',').map(Number) } : {}),
      ...(Q.get('vidholdexcl') ? { forceExcludeSec: +Q.get('vidholdexcl') } : {}),   // no training pick within this many seconds of a held-out frame   // vidhold=t1,t2,...: these frames are always extracted and form the test set (evalFrames)   // window time ceiling (s): walks show little image shift, so this paces them   // 'all' keeps every shot of an edited clip
      log: (m) => console.log('[video]', m),
      onProgress: (e) => { if (e.done % 50 === 0 || e.done === e.total) say('video-' + e.stage, { done: e.done, total: e.total }); },
    });
    const vidMin = +((Date.now() - vT) / 60000).toFixed(2);
    await say('video-done', { engine: ex.engine, frames: ex.frames.length, scanned: ex.sampled, shots: ex.shots ? ex.shots.length : 1, shot: ex.shot ? [ex.shot.start, ex.shot.end] : null, duration: +ex.duration.toFixed(1), videoW: ex.videoW, videoH: ex.videoH, fps: ex.fps && +ex.fps.toFixed(2), rotation: ex.rotation, vidMin });
    if (Q.get('postanalysis')) await post(Q.get('postanalysis'), JSON.stringify({ engine: ex.engine, duration: ex.duration, fps: ex.fps, frames: ex.analysis, picked: ex.frames.map((f) => f.t) }));
    for (const f of ex.frames) files.push(new File([f.source], f.name));
    names = ex.frames.map((f) => f.name);
    if (Q.get('vidhold')) {
      const ht = Q.get('vidhold').split(',').map(Number);
      evalFrames = ht.map((t) => ex.frames.reduce((b, f) => (Math.abs(f.t - t) < Math.abs(b.t - t) ? f : b)).name);
      await say('video-hold', { evalFrames });
    }
  } else {
    if (cfg.list) names = await (await fetch(`/data/${cfg.dir}/files.json`)).json();
    else names = cfg.names();
    await say('fetch', { files: names.length });
    const useMasks = Q.get('masks') && cfg.masks;
    for (const n of names) {
      const r = await fetch(`/data/${cfg.dir}/${n}`);
      if (!r.ok) throw new Error(`missing ${n}`);
      const entry = { source: new File([await r.blob()], n), name: n };
      if (useMasks) {
        const mn = n.replace(/\.[^.]+$/, '.png');
        const mr = await fetch(`/data/${cfg.dir}/${cfg.masks}/${mn}`);
        if (!mr.ok) throw new Error(`missing mask ${mn}`);
        entry.mask = await mr.blob();
      }
      files.push(entry);
    }
  }

  const hullOpts = {};
  if (Q.get('hullkeep')) hullOpts.keep = +Q.get('hullkeep');
  if (Q.get('hullres')) hullOpts.res = +Q.get('hullres');
  if (Q.get('hullmad')) hullOpts.kMad = +Q.get('hullmad');
  const ses = window.__ses = createSession({
    ...(Q.get('init') ? { initTarget: +Q.get('init') } : {}),
    // ?ipf=N pins the per-frame batch: the refine trigger is checked once per
    // frame, so the adaptive batch makes runs diverge with kernel speed
    ...(Q.get('ipf') ? { itersPerFrame: +Q.get('ipf') } : {}),
    maxIters: ITERS,
    evalSplit: Q.has('evalsplit') ? +Q.get('evalsplit') : (cfg.holdout1 ? 0 : 8),
    ...(evalFrames ? { evalFrames } : {}),   // ?evalsplit=0 trains on every view (showcase runs; no test PSNR)
    ...(cfg.holdout1 ? { holdout: 'auto' } : {}),
    // benchmark mode pins resolution like ?eval (adaptive budget otherwise
    // shrinks big sets and PSNR at reduced res is not comparable run-to-run)
    frames: { trainMaxDim: +(Q.get('res') || cfg.res || 1600), ...(Q.get('featres') ? { featMaxDim: +Q.get('featres') } : {}),
      ...(Q.get('maskcut') ? { maskCut: +Q.get('maskcut') } : {}), ...(Q.get('maskbg') ? { maskBgCut: +Q.get('maskbg') } : {}),
      ...(Q.get('maskguard') ? { maskGuard: +Q.get('maskguard') } : {}) },   // px ring outside the matte with no loss   // ?res= overrides the set's training resolution
    // ?classic=1: pre-MCMC defaults (A/B for small-set anomalies)
    // refine cadence is a SESSION option: the economy packages carry their
    // own (Brush 200, LichtFeld 100); ?refevery overrides either
    // solver probes: feature-frame resolution, SIFT budget, first octave (-1 = 2x upsampled)
    // desktop solver default since 2026-09-04: 8000 SIFT features from the upsampled
    // first octave (COLMAP's default) — truck +0.38, garden +0.26, camping ±0 at 30k;
    // pose residual vs COLMAP halved. ?feats= / ?octave= override; ?classicsolve restores 3900 / octave 0
    sfm: {
      // ?solve=quick|standard|precise picks a solve tier (SOLVE_TIERS); the bench's
      // own default stays the precise tier the README numbers were measured with
      ...(Q.has('classicsolve') ? {} : solveTierOpts(Q.get('solve') || 'precise')),
      ...(Q.get('sfmaspect') === '0' ? { refineAspect: false } : Q.get('sfmaspect') ? { refineAspect: true } : {}),
      ...(Q.get('lockk') ? { lockIntrinsics: true } : {}),   // rig faces: f, k1/k2, aspect fixed at their exact slice values
      ...(Q.get('pairinl') ? { pairMinInliers: +Q.get('pairinl') } : {}),
      ...(Q.get('pairinladj') ? { pairMinInliersAdj: +Q.get('pairinladj') } : {}),
      ...(Q.get('relax') === '0' ? { pairRelax: false } : {}),   // disable the automatic relaxed-gate retry   // neighbour-pair (|i-j|<=2) absolute gate (default 15; 1e9 = off)   // absolute E-inlier gate for a pair (default 100; 1e9 = ratio only, the pre-09-08 rule)
      ...(Q.get('feats') ? { siftFeats: +Q.get('feats') } : {}),
      ...(Q.get('octave') ? { siftFirstOctave: +Q.get('octave') } : {}),
      ...(Q.get('peak') ? { siftPeak: +Q.get('peak') } : {}),   // SIFT contrast threshold scale (<1 = more, fainter keypoints)
      ...(Q.get('baratio') != null ? { interimBARatio: +Q.get('baratio') } : {}),   // interim-BA cadence (0 = every 6 registrations, the default; >0 geometric — measured to dive)
      ...(Q.get('bapts') != null ? { interimBAMaxPoints: +Q.get('bapts') } : {}),   // interim-BA point cap (0 = all points)
      ...(Q.get('pairworkers') === '0' ? { pairWorkers: false } : {}),   // pair-geometry RANSAC inline on the main thread (A/B)
      ...(Q.get('exiffocal') === '0' ? { exifFocal: false } : {}),   // ignore the EXIF focal prior (A/B against the focal search)
      ...(Q.get('focals') ? { focalScales: Q.get('focals').split(',').map(Number) } : {}),   // focal-search candidates, relative to the 1.2·maxDim guess
      ...(Q.get('bracket') === '0' ? { focalBracket: false } : {}),   // skip the ±12 % bracket refinement around the search winner
      ...(Q.get('inittrials') != null ? { initTrials: +Q.get('inittrials') } : {}),   // init-pair trials on small sets (default 3; 1 = off)
      ...(Q.get('initpair') ? { initPair: Q.get('initpair').split(',').map(Number) } : {}),   // seed from a named image pair
      ...(Q.get('searchsub') === '0' ? { searchSubset: false } : {}),   // focal search on every image instead of the <= 48 subsample
    },
    ...(Q.has('classic') ? {} : { refineEvery: +(Q.get('refevery') || (Q.get('econ') === 'brush' ? 200 : Q.get('econ') === 'lf' ? 100 : 500)) }),
    trainer: Q.has('classic')
      ? { maxSplats: Math.min(600000, Math.round(ITERS * 15)), capMult: 8, shDeg: 3 }
      : {
        // cap = min(seed·capMult, maxSplats): truck's 25,141-point seed × 8 × 8
        // is 1.61M, so a true 2M needs ?capmult=16 (what the app uses ≥ 1M)
        maxSplats: +(Q.get('maxsplats') || Math.min(2000000, Math.round(ITERS * 35))), capMult: +(Q.get('capmult') || 8), shDeg: +(Q.get('shdeg') ?? 3),   // ?shdeg=0: an avatar preset (view-independent colour)
        growRate: 0.05, mcmcNoise: true, scaleReg: 0.01, moveCap: 0.25, shLr: 3e-4,
        // econ=brush: the Brush economy as ONE package (its pieces never
        // transplanted one at a time): no loss-side reg, decay 0.004/200it,
        // dead below 1/255, every dead splat relocated each refine (200 it),
        // donors ∝ opacity among rendered splats. Individual knobs BELOW
        // override it (attribution cells).
        ...(Q.get('econ') === 'brush' ? { opacityReg: 0, opaDecay: 0.004, deadThr: 1 / 255, poolMin: 0,
          moveCap: 1, donorWeight: 'opavis' } : {}),
        // econ=lf: the LichtFeld MCMC economy — reg kept (0.01, mean-scaled
        // there), dead below 0.005, ALL dead relocated every 100 it, donors
        // ∝ error over the whole population, ratio cap 51, growth 5 %/refine
        ...(Q.get('econ') === 'lf' ? { deadThr: 0.005, poolMin: 0, ratioCap: 51, moveCap: 1,
          growRate: 0.05 } : {}),
        // opacity economy knobs (lab log 2026-09-02, source verdict): dead
        // threshold, donor-pool floor, Brush-style opacity decay (per 200 it)
        ...(Q.get('deadthr') ? { deadThr: +Q.get('deadthr') } : {}),
        ...(Q.get('poolmin') != null ? { poolMin: +Q.get('poolmin') } : {}),
        ...(Q.get('opadecay') ? { opaDecay: +Q.get('opadecay') } : {}),
        ...(Q.get('ratiocap') ? { ratioCap: +Q.get('ratiocap') } : {}),
        ...(Q.get('deadtiny') ? { deadTiny: true } : {}),
        // shared pixel-aspect refinement (log fy/fx) during training
        ...(Q.get('aspect') ? { aspectOpt: true } : {}),
        // subgroup-aggregated gradient atomics in the render backward (speed probe)
        ...(Q.get('sgagg') ? { subgroupAgg: true } : {}),
        ...(Q.get('gspread') ? { gradSpread: +Q.get('gspread') } : {}),
        ...(Q.get('gbatch') ? { gradBatch: +Q.get('gbatch') } : {}),
        ...(Q.get('gzskip') ? { gradZeroSkip: true } : {}),
        ...(Q.get('pvec') ? { projVec: true } : {}),
        // speed-plan bisect toggles: keep the compiled-out work compiled in
        ...(Q.get('usestats') ? { useStats: true } : {}),
        ...(Q.get('camgrads') ? { camGrads: true } : {}),
        // continuation runs ramp the lr over the first N iterations (zero Adam moments); ?warmup=0 disables
        ...((Q.get('warmup') || Q.get('frommodel')) ? { lrWarmup: +(Q.get('warmup') ?? 1000) } : {}),
        ...(Q.get('rectbin') ? { rectBin: true } : {}),   // speed plan #5: per-axis tile binning
        // visibility compaction: chain/Adam/SH-Adam over visible splats only
        ...(Q.get('compact') === '0' ? { compact: false } : {}),
        // tileGrad=0: per-pixel global atomics, no per-splat workgroup barriers (speed probe)
        ...(Q.get('tilegrad') === '0' ? { tileGrad: false } : {}),
        ...(Q.get('asplr') ? { aspectLr: +Q.get('asplr') } : {}),
        ...(Q.get('maxscale') ? { maxScale: +Q.get('maxscale') } : {}),
        // per-frame (key,id) entry budget (default maxSplats*24); rung 4 probe
        ...(Q.get('entriescap') ? { entriesCap: +Q.get('entriescap') } : {}),
        ...(Q.get('dilate') ? { dilate: +Q.get('dilate') } : {}),
        ...(Q.get('aniso') != null ? { anisoReg: +Q.get('aniso') } : {}),
        ...(Q.get('minscale') ? { minScale: +Q.get('minscale') } : {}),
        ...(Q.get('seed') ? { seed: +Q.get('seed') } : {}),
        ...(Q.get('comp') != null ? { mipComp: Q.get('comp') !== '0' } : {}),
        ...(Q.get('regvis') ? { regVisOnly: true } : {}),
        ...(Q.get('opafloor') ? { opaFloor: +Q.get('opafloor') } : {}),
        // placement knobs (rung 3 of docs/plan-placement-2026-09-02.md)
        ...(Q.get('refv2') ? { refineV2: true } : {}),
        ...(Q.get('errdon') ? { errDonors: true } : {}),
        ...(Q.get('splitv2') ? { splitV2: true } : {}),
        ...(Q.get('growrate') ? { growRate: +Q.get('growrate') } : {}),
        ...(Q.get('movecap') ? { moveCap: +Q.get('movecap') } : {}),
        ...(Q.get('poslr') ? { posLrScale: +Q.get('poslr') } : {}),
        ...(Q.get('shramp') === '0' ? { shRamp: false } : {}),
        // reg weights are per-splat constants (3DGS-MCMC's are mean()-scaled,
        // i.e. 1/n) — rung 4 tests whether they must shrink with capacity
        ...(Q.get('opareg') ? { opacityReg: +Q.get('opareg') } : {}),
      ...(Q.get('randbg') ? { randomBg: Q.get('randbg') !== '0' } : {}),   // randbg=0 disables the masked-set default
      ...(Q.get('covw') ? { covW: +Q.get('covw') } : {}),          // mask-supervised coverage (empty pixels -> O = 0)
      ...(Q.get('covs') ? { covSubjW: +Q.get('covs') } : {}),      // and subject pixels -> O = 1
        ...(Q.get('scalereg') ? { scaleReg: +Q.get('scalereg') } : {}),
        // opacityReg ∝ 1/n above oregref splats (rung 4: capacity dilution)
        ...(Q.get('oregref') ? { opaRegRefN: +Q.get('oregref') } : {}),
        ...(Q.get('oregrefmax') ? { opaRegRefMax: +Q.get('oregrefmax') } : {}),
        ...(Q.get('donor') ? { donorWeight: Q.get('donor') } : {}),
        ...(Q.get('ssim') ? { ssimWeight: +Q.get('ssim') } : {}),
        ...(Q.get('v2') ? { engine: 'v2' } : {}),
        ...(Q.get('growfrac') ? { growFrac: +Q.get('growfrac') } : {}),
        ...(Q.get('growtau') ? { growTau: +Q.get('growtau') } : {}),
        ...(Q.get('growuntil') ? { growUntil: +Q.get('growuntil') } : {}),
        ...(Q.get('relocuntil') ? { relocUntil: Q.get('relocuntil') === 'all' ? Infinity : +Q.get('relocuntil') } : {}),   // 'all' = relocate to the last step (pre-09-07 default)
        ...(Q.get('reloctaper') ? { relocTaper: +Q.get('reloctaper') } : {}),
        ...(Q.get('lrdecay') ? { lrDecayFrac: +Q.get('lrdecay') } : {}),
        ...(Q.get('lrdecaymax') ? { lrDecayMax: +Q.get('lrdecaymax') } : {}),   // iterations; 0 = no cap
      },
  });
  // refine census lines (dead / relocated / grown per call) are the only
  // record of what the population did mid-run — posted next to the result
  const refineLog = [];
  const sesLog = [];
  ses.on('log', (m) => { console.log('[SES]', m); if (/^refine @/.test(m)) refineLog.push(m); if (Q.get('sfmlog')) sesLog.push(m); });
  let beat = 0;
  ses.on('stage', (e) => { if (Date.now() - beat > 30000) { beat = Date.now(); say('solve-' + e.stage, { done: e.done, total: e.total }); } });
  await ses.load(files);
  const mf = ses.frames.map((f) => f.maskedFrac || 0);
  const ef = ses.frames.map((f) => f.emptyFrac || 0);
  const gf = ses.frames.map((f) => f.guardFrac || 0);
  const mean = (a) => +(100 * a.reduce((x, y) => x + y, 0) / a.length).toFixed(1);
  await say('decoded', { frames: ses.frames.length, maskedPct: mean(mf), emptyPct: mean(ef), guardPct: mean(gf) });

  const solveT = Date.now();
  let recon;
  if (Q.get('gtrecon')) {
    // externally supplied reconstruction (e.g. COLMAP GT parsed to our
    // format) — poses + cloud swap in, everything else identical
    const gj = await (await fetch(Q.get('gtrecon'))).json();
    if (gj.frames && gj.cams) {
      // an app-published recon (buildReconJson): cams index ITS frame list and carry f at
      // ITS feature scale — remap by frame name onto our frames and rescale f to our fw;
      // its sparse cloud is a flat {xyz, rgb} pair, the session wants [{X, rgb}]
      const byName = new Map(ses.frames.map((f, i) => [f.name, i]));
      gj.cams = gj.cams.map((c) => {
        const fr = gj.frames[c.imgIdx]; const i = fr ? byName.get(fr.name) : undefined;
        if (i == null) return null;
        const s = ses.frames[i].fw / fr.fw;
        // crop cams (face_crops.py) carry an off-centre principal point: scale it like f
        return { ...c, imgIdx: i, f: c.f * s, ...(c.fy != null ? { fy: c.fy * s } : {}),
          cx: c.cx != null ? c.cx * s : ses.frames[i].fw / 2, cy: c.cy != null ? c.cy * s : ses.frames[i].fh / 2 };
      }).filter(Boolean);
    }
    if (!gj.points && gj.cloud && gj.cloud.xyz) {
      const xyz = gj.cloud.xyz, rgb = gj.cloud.rgb || [];
      gj.points = [];
      for (let i = 0; i + 2 < xyz.length; i += 3) gj.points.push({ X: [xyz[i], xyz[i + 1], xyz[i + 2]], rgb: [rgb[i] ?? 128, rgb[i + 1] ?? 128, rgb[i + 2] ?? 128] });
    }
    recon = ses.useReconstruction(gj);
    await say('gt-recon', { cams: recon.cams.length, points: recon.points.length });
  } else {
    recon = await ses.solve();
  }
  const solveMin = +((Date.now() - solveT) / 60000).toFixed(1);
  if (Q.get('sfmlog')) await post(`bench_${TAG}_sfmlog.txt`, sesLog.join('\n'));   // ?sfmlog=1: the solver's full log (registration verdicts per image)
  await say('solved', { cams: recon.cams.length, of: ses.frames.length, rms: recon.rmsBA && +recon.rmsBA.toFixed(2), solveMin });
  if (Q.get('postrecon')) {
    await post(Q.get('postrecon'), JSON.stringify({
      cams: recon.cams.map((c) => ({
        imgIdx: c.imgIdx, name: ses.frames[c.imgIdx].name,
        R: c.R, t: c.t, f: c.f, ...(c.fy != null ? { fy: c.fy } : {}), cx: c.cx, cy: c.cy,
      })),
      points: recon.points.map((p) => ({ X: p.X, rgb: p.rgb })),
      k1: recon.k1, k2: recon.k2,
    }));
  }

  if (Q.get('frommodel')) {
    // start from an existing model (.ply / .sog / session zip) instead of the
    // SfM seed — speed cells profile at a realistic splat count in ~1 min
    // (with ?iters=1&gputime=N) instead of growing there for 7 minutes
    const { decodeModel } = await import('../../app/js/session_io.js');
    const bytes = new Uint8Array(await (await fetch(Q.get('frommodel'))).arrayBuffer());
    const { gaussians, state } = await decodeModel(bytes, null);
    // ?stripsh=1: drop the imported SH bands so the continuation trains a
    // view-independent model (with ?shdeg=0) — an SH0 EXPORT of an SH3 model
    // is not that: the DC alone exposes needle splats the bands were hiding
    // (2026-09-12, the face in the app)
    if (Q.get('stripsh')) { gaussians.sh = null; gaussians.shK = 0; }
    // a bare .ply/.sog carries BAKED opacities (undo them); a session zip's state.bin is raw
    await ses.seedFrom(gaussians, { iter: +(Q.get('fromiter') || (state && state.iter) || 0), unbake: !state });   // ?fromiter=N continues the schedules from there
    await say('seeded-from-model', { splats: gaussians.n });
  } else {
    await ses.seed({ hullOpts });
  }
  if (cfg.pano) {
    // rig-aware split (see omni_run.js): all six faces of every 8th pano
    // held out, only the four yaw faces scored
    for (const i of ses.testCams) ses.trainer.excluded.delete(i);
    ses.testCams = [];
    ses.trainer.camMeta.forEach((m, i) => {
      const pano = Math.floor(m.imgIdx / 6), face = m.imgIdx % 6;
      if (pano % 8 === 0) {
        ses.trainer.excluded.add(i);
        if (face < 4) ses.testCams.push(i);
      }
    });
    ses.holdout = ses.testCams[ses.testCams.length >> 1];
    ses.trainer.holdout = ses.holdout;
  }
  await say('seeded', { splats: ses.trainer.n });

  const trainT = Date.now();
  const done = new Promise((res) => ses.on('event', (e) => { if (e.kind === 'train-complete') res(); }));
  const guard = setInterval(() => {
    const lh = ses.lossHistory[ses.lossHistory.length - 1];
    say('train', { iter: ses.trainer.iter, splats: ses.trainer.n, psnr: lh ? +lh[1].toFixed(2) : null });
  }, 60000);
  ses.start();
  // ?minutes=N: a wall-clock training budget — the run ends at N minutes
  // wherever the iteration count is (the schedule still runs on ?iters, so
  // size that to what fits: ~1 min per 1k at the 2M cap on the 5080)
  // ?evalmin=N: held-out (eval8) PSNR every N minutes of TRAINING time for a
  // dB-over-time curve; the evaluation itself is excluded from the clock
  // (evalOverhead) so the minute budget and the curve's x-axis stay train-only
  let evalOverhead = 0, evaluating = false;
  const curve = [];
  const trainClockMin = () => (Date.now() - trainT - evalOverhead) / 60000;
  let curveTimer = null;
  if (Q.get('evalmin') && !cfg.holdout1) {
    curveTimer = setInterval(async () => {
      if (evaluating) return;
      evaluating = true;
      const t = Date.now();
      const min = +trainClockMin().toFixed(2), iter = ses.trainer.iter, splats = ses.trainer.n;
      const r = await ses.evalTestPsnr();
      evalOverhead += Date.now() - t;
      if (r) { curve.push({ min, iter, splats, psnr: +r.psnr.toFixed(3) }); await say('curve', { min, iter, psnr: +r.psnr.toFixed(3) }); }
      evaluating = false;
    }, +Q.get('evalmin') * 60000);
  }
  let clock = null;
  if (Q.get('minutes')) {
    clock = setInterval(() => {
      if (trainClockMin() >= +Q.get('minutes')) { ses.opts.maxIters = ses.trainer.iter; clearInterval(clock); }
    }, 5000);
  }
  await done;
  clearInterval(guard); if (clock) clearInterval(clock); if (curveTimer) clearInterval(curveTimer);
  while (evaluating) await new Promise((r) => setTimeout(r, 200));
  const trainMin = +trainClockMin().toFixed(1);
  // dead census at the horizon (opacity < 1/255: what the export purges)
  let deadPct = null;
  {
    const { data, n } = await ses.trainer.readGaussians();
    let dead = 0;
    for (let i = 0; i < n; i++) if (data[i * 16 + 13] <= Math.log(1 / 254)) dead++;
    deadPct = +(100 * dead / n).toFixed(2);
  }

  let psnrTest = null, heldOut = 0;
  if (cfg.holdout1) {
    psnrTest = +(await ses.trainer.evalCamPsnr(ses.holdout)).toFixed(3);
    heldOut = 1;
  } else {
    const test = await ses.evalTestPsnr();
    psnrTest = test ? +test.psnr.toFixed(3) : null;
    heldOut = test ? test.frames.length : 0;
  }
  if (Q.get('postply')) {
    // dump the trained model for distribution forensics (splat_stats.mjs)
    await say('export-ply');
    const blob = await ses.exportPlyBlob();
    await post(Q.get('postply'), blob);
    await say('ply-posted', { mb: +(blob.size / 1e6).toFixed(0) });
  }
  if (Q.get('poststate')) {
    // the trainer's RAW state (no opacity bake) as a resumable session zip —
    // what a continuation run (?frommodel=) should start from
    const { packState } = await import('../../app/js/session_io.js');
    const { zipStore } = await import('../../app/js/zip.js');
    const zip = zipStore([{ name: 'state.bin', data: await packState(ses) }]);
    await post(Q.get('poststate'), zip);
    await say('state-posted', { mb: +(zip.size / 1e6).toFixed(0) });
  }
  if (Q.get('gputime')) {
    // per-kernel GPU time at the final splat count (timestamp queries, one
    // pass per kernel, +N profiling steps of training)
    await say('gputime');
    const prof = await ses.trainer.profileSteps(+Q.get('gputime') || 100);
    await post(`bench_${TAG}_gputime.json`, JSON.stringify(prof));
    console.log('[GPUTIME]', JSON.stringify(prof));
  }
  if (Q.get('postperf')) {
    // the session's per-frame perf rows [t_ms, iter, batch, splats, enc, view, fence, met, total]
    // — ms per iteration vs splat count over the run (where the time goes as n grows)
    await post(Q.get('postperf'), JSON.stringify({ frames: (ses.perf && ses.perf.frames) || [], marks: (ses.perf && ses.perf.marks) || [] }));
  }
  if (Q.get('postview')) {
    // the viewer's recon JSON (what the app's Share writes): camera path,
    // frames, photo URLs at the deployment's data root, run stats — so a
    // bench export can be published as ?model=<sog>&recon=<this> like the
    // README's signature models. ?viewname= names it.
    const { buildReconJson } = await import('../../app/js/session_io.js');
    const dataRoot = `https://ugc.arrival.space/splatjs/data/${cfg.dir}/`;
    const last = (ses.lossHistory || []).at(-1);
    const S = {
      session: ses, preset: { id: SET, name: Q.get('viewname') || SET },
      loadedFiles: files.map((fl) => ({ name: fl.name, url: dataRoot + fl.name })),
      minutes: Math.round(trainMin), psnrTrain: last ? last[1] : null, psnrHold: null,
      psnrTest: psnrTest != null ? { psnr: psnrTest, frames: heldOut } : null,
      holdHist: [],
    };
    await post(Q.get('postview'), JSON.stringify(buildReconJson(S)));
    await say('view-posted');
  }
  const result = {
    set: SET, iters: ITERS, iterDone: ses.trainer.iter,
    aspect: +Math.exp(ses.trainer.logAspect || 0).toFixed(5),   // refined fy/fx (1 = untouched)
    psnrTest,
    heldOut,
    protocol: cfg.holdout1 ? 'holdout1' : 'eval8',
    trainMin, solveMin, deadPct, ...(curve.length ? { curve } : {}),
    // tiles whose (key,id) entry budget overflowed over the whole run — a
    // non-zero count means whole tiles were silently skipped in training
    overflowTiles: ses.trainer.entryOverflowTiles || 0,
    splats: ses.trainer.n,
    cams: recon.cams.length, of: ses.frames.length,
    rms: recon.rmsBA && +recon.rmsBA.toFixed(3),
    res: cfg.res || 1600, ts: new Date().toISOString(),
  };
  await post(`bench_${TAG}_result.json`, JSON.stringify(result));
  try { await post(`bench_${TAG}_refines.txt`, refineLog.join('\n')); } catch {}
  await say('DONE', result);
} catch (e) {
  console.error(e);
  await say('ERROR', { message: String((e && e.message) || e) });
}
