import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(new URL('../..', import.meta.url).pathname);
const moduleDir = resolve(root, 'android/app');
const gradle = readFileSync(resolve(moduleDir, 'build.gradle'), 'utf8');
assert.match(gradle, /from\('\.\.\/\.\.\/app'\)\s*\{\s*into 'app'/,
  'web assets must be copied from the repository app/ directory');
assert.match(gradle, /from\('\.\.\/\.\.\/src'\)\s*\{\s*into 'src'/,
  'the JS engine must be copied from the repository src/ directory');
assert.equal(resolve(moduleDir, '../../app/index.html'), resolve(root, 'app/index.html'));
assert.equal(resolve(moduleDir, '../../src/index.js'), resolve(root, 'src/index.js'));

const manifest = readFileSync(resolve(moduleDir, 'src/main/AndroidManifest.xml'), 'utf8');
assert.match(manifest, /NativeCaptureActivity[^>]*android:screenOrientation="portrait"/,
  'ARCore capture must stay portrait-locked');
const native = readFileSync(resolve(moduleDir, 'src/main/java/space/arrival/splatxr/NativeCaptureActivity.java'), 'utf8');
assert.match(native, /config\.setFocusMode\(Config\.FocusMode\.AUTO\)/,
  'ARCore capture must request continuous autofocus');
assert.match(native, /text == null\) throw new IllegalStateException/,
  'capture export must report JSON serialization errors rather than dereference null');
assert.match(native, /if \(!finite\(world\[0\], world\[1\], world\[2\], confidence\) \|\| confidence < 0\.25f\)/,
  'non-finite feature points must not be serialized into seed JSON');
assert.match(native, /view\.setPadding\(left, top, right, bottom\)/,
  'all app content must be inset below system bars and display cutouts');
assert.match(native, /getDecorView\(\)\.setSystemUiVisibility\(0\)/,
  'capture UI must not request layout behind system bars');
assert.match(native, /frame\.transformDisplayUvCoords\(uvInput, uvOutput\);[\s\S]*?uvOutput\.position\(0\);/,
  'camera preview must use ARCore-transformed display coordinates');
assert.match(native, /frame\.transformDisplayUvCoords\(uvInput, uvOutput\);[\s\S]*?for \(int row = 0; row < 2; row\+\+\)[\s\S]*?int left = row \* 4, right = left \+ 2;[\s\S]*?uvOutput\.put\(left, uvOutput\.get\(right\)\);[\s\S]*?uvOutput\.put\(right, leftU\);/,
  'only the camera preview UV pairs should be swapped across the display horizontally');
assert.match(native, /Matrix\.multiplyMM\(mvp, 0, projection, 0, view, 0\);/,
  'coverage points must use ARCore projection and view matrices directly');
assert.doesNotMatch(native, /mvp\[offset(?: \+ 1)?\] = -mvp\[offset(?: \+ 1)?\]/,
  'coverage points must not apply a second clip-space axis reflection');
const main = readFileSync(resolve(moduleDir, 'src/main/java/space/arrival/splatxr/MainActivity.java'), 'utf8');
assert.match(main, /applySafeAreaInsets\(host\)/,
  'the main WebView host must apply safe-area insets on edge-to-edge Android');
assert.match(main, /WindowInsets\.Type\.systemBars\(\) \| WindowInsets\.Type\.displayCutout\(\)/,
  'safe-area padding must cover status/navigation bars and display cutouts');
assert.match(native, /float degrees = Float\.isFinite\(maxAngle\)[\s\S]*Math\.toDegrees\(maxAngle\)/,
  'the angular value must be converted from radians to degrees before mapping colors');
assert.match(native, /degrees < 40f[\s\S]*degrees < 80f[\s\S]*degrees - 80f\) \/ 40f/,
  'the color ramp must span red to orange to yellow to green at 0, 40, 80 and 120 degrees');
assert.match(native, /MAX_DIRECTIONS = 16/);
assert.match(native, /closestDot < nearestDots\[redundant\]/,
  'the bounded view-direction sample must replace redundant early views with novel later angles');
assert.match(native, /Math\.acos\(widestDot\)/);
assert.match(native, /clampDot\(directions\[offset\] \* dx/,
  'viewing-direction dot products must be clamped before angular comparison');
assert.match(native, /translation\[0\] - x, translation\[1\] - y, translation\[2\] - z/,
  'viewing directions must run from each world-space point toward the ARCore camera position');
console.log('Android asset paths, portrait lock, capture orientation, safe-area insets, and coverage colors: passed');
