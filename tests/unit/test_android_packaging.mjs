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
assert.doesNotMatch(native, /uvOutput\.put\(i(?: \+ 1)?,\s*1f\s*-\s*uvOutput\.get/,
  'camera preview must not apply an extra axis reflection after ARCore transforms UVs');
assert.match(native, /Matrix\.multiplyMM\(mvp, 0, projection, 0, view, 0\);/,
  'coverage points must use ARCore projection and view matrices directly');
assert.doesNotMatch(native, /mvp\[offset(?: \+ 1)?\] = -mvp\[offset(?: \+ 1)?\]/,
  'coverage points must not apply a second clip-space axis reflection');
const main = readFileSync(resolve(moduleDir, 'src/main/java/space/arrival/splatxr/MainActivity.java'), 'utf8');
assert.match(main, /applySafeAreaInsets\(host\)/,
  'the main WebView host must apply safe-area insets on edge-to-edge Android');
assert.match(main, /WindowInsets\.Type\.systemBars\(\) \| WindowInsets\.Type\.displayCutout\(\)/,
  'safe-area padding must cover status/navigation bars and display cutouts');
assert.match(native, /orangeAt = \(float\)Math\.toRadians\(40\.0\)/);
assert.match(native, /yellowAt = \(float\)Math\.toRadians\(80\.0\)/);
assert.match(native, /greenAt = \(float\)Math\.toRadians\(120\.0\)/);
assert.match(native, /green = 0\.42f \+ 0\.52f \* t/);
assert.match(native, /red = 1f - 0\.92f \* t/);
console.log('Android asset paths, portrait lock, capture orientation, safe-area insets, and coverage colors: passed');
