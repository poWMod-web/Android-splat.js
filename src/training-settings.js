export const MAX_CUSTOM_CYCLES = 1_000_000;
export const MIN_LOD_SPLATS = 250_000;

export function parseCycleCount(value) {
  const text = String(value ?? '').trim();
  if (!text) return 0; // the app's existing default: 20,000 cycles
  const count = Number(text);
  if (!Number.isSafeInteger(count) || count < 1 || count > MAX_CUSTOM_CYCLES) {
    throw new RangeError(`Cycles must be a whole number from 1 to ${MAX_CUSTOM_CYCLES.toLocaleString('en-US')}, or leave blank for the default.`);
  }
  return count;
}

export function maxLODLevels(splatBudget) {
  if (!Number.isFinite(splatBudget) || splatBudget < 1_000_000) return 0;
  let n = Math.round(splatBudget), levels = 0;
  while (n >= MIN_LOD_SPLATS) { levels += 1; n = Math.round(n / 2); }
  return Math.max(0, levels - 1); // snapshots below the final full-budget model
}

export function parseLODCount(value, splatBudget) {
  const text = String(value ?? '').trim();
  if (!text || text === '0') return 0; // Auto uses the existing halving schedule.
  const count = Number(text), max = maxLODLevels(splatBudget);
  if (!Number.isSafeInteger(count) || count < 1 || count > max) {
    throw new RangeError(max ? `Choose 1–${max} LOD snapshots for this splat budget, or 0 for Auto.` : 'Choose a splat budget of at least 1,000,000 before enabling LOD.');
  }
  return count;
}

export function buildLODPlan(splatBudget, requestedLevels = 0, cycles = 20_000) {
  if (!maxLODLevels(splatBudget)) return null;
  const count = parseLODCount(requestedLevels, splatBudget);
  const all = [];
  for (let n = Math.round(splatBudget); n >= MIN_LOD_SPLATS; n = Math.round(n / 2)) all.unshift(n);
  // A custom count is the number of earlier detail snapshots; always append the
  // requested full-budget target as the final, live model.
  const levels = count ? all.slice(-(count + 1)) : all;
  // Scale polish to the run and requested snapshot count, never extending the chosen horizon.
  const perLevel = Math.floor(cycles * 0.25 / Math.max(1, levels.length - 1));
  const polish = Math.max(1, Math.min(12000, Math.floor(cycles * 0.05), perLevel));
  return { levels, idx: 0, holdUntil: null, polish, snaps: [] };
}
