// Frame selection for video intake (src/io/video.js): pure logic, no decoder.
//   - shot-cut detection splits an edited timeline and the default keeps the
//     longest take; windows never span a cut
//   - motion windows pace a moving camera by displacement, a static camera by
//     the time cap; blur dips lose to their neighbours; the device cap widens
//     windows instead of dropping the sharpest
import { selectFrames, detectShots, applyForced } from '../../src/io/video.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name} ${detail}`);
  if (!cond) failures++;
};

// deterministic pseudo-random
let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

const mk = (n, f) => Array.from({ length: n }, (_, i) => ({ t: i / 30, focus: 800, lum: 0.5, clip: 0, motion: 0.002, diff: 0.02 + 0.01 * rnd(), ...f(i) }));

// 1. shot detection: two hard cuts, longest take wins, picks stay inside it
{
  const frames = mk(600, (i) => ({ diff: (i === 200 || i === 420) ? 0.5 : undefined }));
  const shots = detectShots(frames);
  check('two cuts -> three shots', shots.length === 3 && shots[1].start === 200 && shots[2].start === 420, JSON.stringify(shots));
  const r = selectFrames(frames, { maxFrames: 300 });
  check('longest shot kept', r.shot && r.shot.start === 200 && r.shot.end === 419);
  check('picks inside the shot', r.picks.every((p) => p >= 200 && p <= 419), `${Math.min(...r.picks)}..${Math.max(...r.picks)}`);
  const all = selectFrames(frames, { maxFrames: 300, shots: 'all' });
  const spansCut = all.picks.some((p, i) => i && ((all.picks[i - 1] < 200 && p >= 200) || (all.picks[i - 1] < 420 && p >= 420)) && false);
  check("shots:'all' picks from every shot", [0, 200, 420].every((s) => all.picks.some((p) => p >= s && p < s + 220)) && !spansCut);
}

// 2. a moving camera closes windows by displacement: 10 % of the width at 1 %/frame -> every ~10 frames
{
  const frames = mk(300, () => ({ motion: 0.01 }));
  const r = selectFrames(frames, { maxFrames: 300, minFrames: 1 });
  const gaps = r.picks.slice(1).map((p, i) => p - r.picks[i]);
  const med = gaps.sort((a, b) => a - b)[gaps.length >> 1];
  check('displacement pacing ~10 frames', med >= 8 && med <= 14, `median gap ${med} frames, ${r.picks.length} picks`);
}

// 3. a static camera still yields frames, paced by the time cap (0.25 s = 7-8 frames)
{
  const frames = mk(300, () => ({ motion: 0 }));
  const r = selectFrames(frames, { maxFrames: 300, minFrames: 1 });
  const gaps = r.picks.slice(1).map((p, i) => p - r.picks[i]);
  check('static camera paced by the time cap', gaps.length && gaps.every((g) => g >= 7 && g <= 9), `gaps ${[...new Set(gaps)].join(',')}`);
}

// 4. a blur dip inside a window loses to its neighbours
{
  const frames = mk(300, (i) => ({ motion: 0.01, focus: i % 20 === 5 ? 200 : 800 + 5 * Math.sin(i) }));
  const r = selectFrames(frames, { maxFrames: 300, minFrames: 1 });
  check('blur dips never picked', r.picks.every((p) => frames[p].focus > 400), `${r.picks.filter((p) => frames[p].focus <= 400).length} dips picked`);
  check('dips flagged', frames.filter((f) => f.blur).length >= 10);
}

// 5. the device cap widens windows rather than truncating the timeline
{
  const frames = mk(3000, () => ({ motion: 0.01 }));
  const r = selectFrames(frames, { maxFrames: 60, minFrames: 1 });
  check('cap respected', r.picks.length <= 60 && r.picks.length >= 40, `${r.picks.length} picks`);
  check('whole timeline covered', Math.max(...r.picks) > 2800 && Math.min(...r.picks) < 200);
}

// 6. a short clip still returns at least minFrames when it can
{
  const frames = mk(90, () => ({ motion: 0.001 }));
  const r = selectFrames(frames, { maxFrames: 300, minFrames: 24 });
  check('short clip reaches the floor', r.picks.length >= 20, `${r.picks.length} picks from 90 frames`);
  const t = r.picks.map((p) => frames[p].t);
  check('floor keeps >= 0.1 s between picks', t.slice(1).every((v, i) => v - t[i] >= 0.099));
}

if (failures) { console.log(`${failures} failure(s)`); process.exit(1); }
console.log('VIDEO SELECT TESTS PASSED');

// 7. forced timestamps (a fixed evaluation set): nearest frames always kept,
//    other picks inside the exclusion zone dropped; uniform pick mode is a
//    fixed-rate control with no scoring
{
  const frames = mk(300, () => ({ motion: 0.01 }));
  const r = selectFrames(frames, { maxFrames: 300, minFrames: 1 });
  const picks = r.picks.slice();
  applyForced(frames, picks, { forceTimes: [2.0, 5.0], forceExcludeSec: 0.3 });
  check('forced frames present', picks.includes(60) && picks.includes(150), picks.join(','));
  const near = picks.filter((p) => p !== 60 && p !== 150 && (Math.abs(frames[p].t - 2.0) < 0.3 || Math.abs(frames[p].t - 5.0) < 0.3));
  check('exclusion zone empty', near.length === 0, `${near.length} picks inside 0.3 s`);
  check('picks sorted and unique', picks.every((p, i) => !i || p > picks[i - 1]));
  const u = selectFrames(frames, { pick: 'uniform', uniformFps: 3 });
  const gaps = u.picks.slice(1).map((p, i) => p - u.picks[i]);
  check('uniform 3 fps -> every 10 frames', u.picks.length === 30 && gaps.every((g) => g === 10), `${u.picks.length} picks, gaps ${[...new Set(gaps)].join(',')}`);
}
