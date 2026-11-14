import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const browser = readFileSync(new URL('../../browser/app/js/app.js', import.meta.url), 'utf8');

for (const [name, source] of [['Android', app], ['Browser', browser]]) {
  const wireStart = source.indexOf('function wireFrameLongPress(');
  const wireEnd = source.indexOf('function buildStrip(', wireStart);
  assert.ok(wireStart >= 0 && wireEnd > wireStart, `${name}: filmstrip long-press wiring should exist`);
  const wire = source.slice(wireStart, wireEnd);
  assert.match(wire, /setTimeout\([\s\S]*removeTrainingImage\(i\)[\s\S]*1000/,
    `${name}: holding a trainable thumbnail for one second should request removal`);
  assert.match(wire, /Math\.hypot\([\s\S]*> 10\) cancel\(\)/,
    `${name}: scrolling the filmstrip should cancel the hold`);
  assert.match(wire, /if \(!fired\) return;[\s\S]*preventDefault\(\)[\s\S]*stopImmediatePropagation\(\)/,
    `${name}: the release click after a completed hold must not select the image`);

  const removeStart = source.indexOf('async function removeTrainingImage(');
  const removeEnd = source.indexOf('function wireFrameLongPress(', removeStart);
  const remove = source.slice(removeStart, removeEnd);
  assert.match(remove, /confirm\(`Remove frame \$\{i \+ 1\} from training\?[\s\S]*excludeCameraImage\(cam\.ci\)/,
    `${name}: removal requires confirmation and uses stable camera exclusion`);
  assert.match(remove, /cam\.state = 'removed'[\s\S]*checkpointRun\('remove-image'\)/,
    `${name}: removed state should be visible and durable`);
  assert.doesNotMatch(remove, /S\.photos\.splice|S\.scene\.cams\.splice/,
    `${name}: removal must not shift persisted photo/camera indices`);
  assert.match(source, /\['align', 'remove-image'\]\.includes\(reason\)[\s\S]*\['train', 'done'\]\.includes\(S\.state\)/,
    `${name}: removal checkpoints should work during training and on completed models`);
  assert.match(source, /status: S\.state === 'done' \? 'finished'/,
    `${name}: editing a completed model must preserve its finished status`);
}

console.log('filmstrip long-press image removal: passed');
