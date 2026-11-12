import assert from 'node:assert/strict';

// DOM stub: boot browser/app.js in Node (no WebGPU needed — no run starts)
// and drive the settings panel the way a user would.
const listeners = new Map();   // id -> { event: [fn] }
const els = new Map();
function mkEl(id, defaults = {}) {
  const el = {
    id,
    value: '',
    textContent: '',
    disabled: false,
    hidden: false,
    style: {},
    scrollTop: 0,
    scrollHeight: 0,
    classList: { add() {}, remove() {} },
    addEventListener(ev, fn) {
      const m = listeners.get(id) || {};
      (m[ev] ??= []).push(fn);
      listeners.set(id, m);
    },
    querySelectorAll: () => [],
    replaceChildren() {},
    append() {},
    ...defaults,
  };
  els.set(id, el);
  return el;
}
// the panel's HTML defaults (mirrors index.html's selected options)
const DEFAULTS = { 'opt-q': 'standard', 'opt-res': '', 'opt-feat': '', 'opt-tier': 'standard',
  'opt-buf': '1', 'opt-sh': '3', 'opt-splats': '', 'opt-lod': '', 'opt-lod-count': '0',
  'opt-mcmc': '', 'opt-iters': '30000', 'opt-gpu': '100', 'opt-gap': '0' };
globalThis.document = {
  getElementById: (id) => els.get(id) || mkEl(id, { value: DEFAULTS[id] ?? '' }),
  createElement: () => mkEl('x'),
};
Object.defineProperty(globalThis, 'navigator', { value: { gpu: {} }, configurable: true });
globalThis.matchMedia = () => ({ matches: false });
globalThis.screen = { width: 1920, height: 1080 };

function fire(id, ev) {
  for (const fn of (listeners.get(id) || {})[ev] || []) fn();
}

await import('../../browser/app.js');

// boot: the panel defaults read as the Standard macro, LOD count parked
assert.equal(els.get('opt-q').value, 'standard');
assert.equal(els.get('opt-lod-count').disabled, true);
assert.match(els.get('log').textContent, /ready/);

// touching a row drops the macro to Custom
els.get('opt-res').value = '1280';
fire('opt-res', 'change');
assert.equal(els.get('opt-q').value, 'custom');

// Draft applies its rows and the macro label follows
els.get('opt-q').value = 'draft';
fire('opt-q', 'change');
for (const [id, v] of [['opt-res', '480'], ['opt-buf', '1'], ['opt-sh', '0'],
    ['opt-iters', '10000'], ['opt-splats', ''], ['opt-tier', 'quick'],
    ['opt-lod', ''], ['opt-lod-count', '0']]) {
  assert.equal(els.get(id).value, v, `${id} follows Draft`);
}
assert.equal(els.get('opt-q').value, 'draft');

// High / Showcase land on this menu's budgets (40k exact, 60k cap)
els.get('opt-q').value = 'high';
fire('opt-q', 'change');
assert.equal(els.get('opt-iters').value, '40000');
assert.equal(els.get('opt-q').value, 'high');
els.get('opt-q').value = 'showcase';
fire('opt-q', 'change');
assert.equal(els.get('opt-iters').value, '60000');
assert.equal(els.get('opt-res').value, '1280');
assert.equal(els.get('opt-q').value, 'showcase');

// Precise (no macro offers it) reads Custom
els.get('opt-q').value = 'standard';
fire('opt-q', 'change');
els.get('opt-tier').value = 'precise';
fire('opt-tier', 'change');
assert.equal(els.get('opt-q').value, 'custom');

// the LOD snapshot count only enables where a plan can exist
els.get('opt-lod').value = '1';
fire('opt-lod', 'change');
assert.equal(els.get('opt-lod-count').disabled, true, 'still no 1M budget');
els.get('opt-splats').value = '1000000';
fire('opt-splats', 'change');
assert.equal(els.get('opt-lod-count').disabled, false);
assert.equal(els.get('opt-q').value, 'custom', 'LOD always reads Custom');

// the GPU slider labels live
els.get('opt-gpu').value = '50';
fire('opt-gpu', 'input');
assert.equal(els.get('opt-gpu-v').textContent, '50%');

console.log('browser settings panel (macro, rows, LOD gate, GPU slider): passed');
