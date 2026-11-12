// manifest.js — the one record the avatar stages share.
//
// Avatar mode is a set of stages that each read and write this JSON, kept on
// the capture record (IndexedDB) next to the frames, so a stage can be re-run
// from any point, the cards can render it, and the rest of the app only ever
// sees `manifest.kind === 'avatar'`. Nothing about people leaks into src/.
//
//   {
//     kind: 'avatar', version: 1, created,
//     source: { video, picks, videoW, videoH, duration },
//     stages: {
//       matte:     { status, model, coverage },                   // per-frame alpha lives on the capture files
//       train:     { status, iters, splats, iou? },
//       landmarks: { status, markers, face, cropCams },
//       facepass:  { status, iters, facePsnr },
//       bodyfit:   { status, job, path: 'anny'|'rigmesh' },
//       bind:      { status, leakage },
//       publish:   { status, avatarId, assigned },
//     },
//   }
// status: 'todo' | 'running' | 'done' | 'skipped' | 'failed' | 'rejected'

export const STAGES = ['matte', 'train', 'landmarks', 'facepass', 'cut', 'facepolish', 'bodyfit', 'bind', 'publish'];

export const STAGE_LABEL = {
  matte: 'Cut out',
  train: 'Train',
  landmarks: 'Find the joints',
  facepass: 'Sharpen the face',
  cut: 'Isolate',
  facepolish: 'Polish the face',
  bodyfit: 'Fit the body',
  bind: 'Bind',
  publish: 'Put on your account',
};

// the same stages as one short word each: the dock's beat row shows all nine
// side by side (the long labels are the cards' titles and the beats' tooltips)
export const STAGE_BEAT = {
  matte: 'Cut out',
  train: 'Train',
  landmarks: 'Joints',
  facepass: 'Face',
  cut: 'Isolate',
  facepolish: 'Polish',
  bodyfit: 'Body',
  bind: 'Bind',
  publish: 'Account',
};

export function newManifest(source = {}) {
  const stages = {};
  for (const s of STAGES) stages[s] = { status: 'todo' };
  return { kind: 'avatar', version: 1, created: Date.now(), source, stages };
}

export function isAvatar(m) { return !!m && m.kind === 'avatar'; }

/** Write a stage's result into the manifest (a shallow merge over the stage). */
export function setStage(m, stage, patch) {
  m.stages[stage] = { ...(m.stages[stage] || {}), ...patch, at: Date.now() };
  return m;
}

export function nextStage(m) {
  return STAGES.find((s) => !['done', 'skipped'].includes((m.stages[s] || {}).status)) || null;
}
