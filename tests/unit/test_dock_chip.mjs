import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../../app/css/app.css', import.meta.url), 'utf8');

// the stop-and-keep chip sits BELOW the pause circle, in normal flow so the
// top dock contains it — never absolutely overhanging the stage edge
{
  const playwrap = css.match(/\.playwrap \{[^}]*\}/)[0];
  assert.match(playwrap, /flex-direction: column/,
    'the pause circle should stack above the chip');
  const chip = css.match(/#t-finish \{[^}]*\}/)[0];
  assert.match(chip, /position: static/,
    'the chip should sit in flow, inside the dock');
  assert.doesNotMatch(chip, /position: absolute/,
    'the chip should never overhang the dock onto the stage');
  assert.match(chip, /margin-top:/,
    'the chip should clear the circle below it');
}

console.log('dock stop-and-keep placement: passed');
