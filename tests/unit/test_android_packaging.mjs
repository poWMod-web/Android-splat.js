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
assert.match(native, /private static String readText\(File f\)[\s\S]*new FileInputStream\(f\)/,
  'capture JSON loading must use an API-24-compatible file stream');
assert.doesNotMatch(native, /java\.nio\.file|\.toPath\(\)/,
  'capture code must avoid java.nio.file APIs unavailable at minSdk 24');
assert.match(native, /if \(!finite\(world\[0\], world\[1\], world\[2\], confidence\) \|\| confidence < 0\.25f\)/,
  'non-finite feature points must not be serialized into seed JSON');
assert.match(native, /view\.setPadding\(left, top, right, bottom\)/,
  'all app content must be inset below system bars and display cutouts');
assert.match(native, /getDecorView\(\)\.setSystemUiVisibility\(0\)/,
  'capture UI must not request layout behind system bars');
assert.match(native, /frame\.transformDisplayUvCoords\(uvInput, uvOutput\);[\s\S]*?uvOutput\.position\(0\);/,
  'camera preview must use ARCore-transformed display coordinates');
const backgroundDraw = native.slice(native.indexOf('private void drawBackground(Frame frame)'), native.indexOf('private void updateCoverage(Frame frame)'));
assert.match(backgroundDraw, /frame\.transformDisplayUvCoords\(uvInput, uvOutput\);[\s\S]*for \(int row = 0; row < 2; row\+\+\)[\s\S]*int left = row \* 4, right = left \+ 2;[\s\S]*uvOutput\.put\(left, uvOutput\.get\(right\)\);[\s\S]*uvOutput\.put\(right, u\);/,
  'the front-camera preview should be mirrored horizontally by swapping UV pairs within each row');
assert.match(backgroundDraw, /Correct the camera image\x27s left-right mirroring[\s\S]*points and camera overlays keep their world-space poses/,
  'correcting preview mirroring must not alter camera overlay or world-space point orientation');
assert.match(native, /Matrix\.multiplyMM\(mvp, 0, projection, 0, view, 0\);/,
  'coverage points must use ARCore projection and view matrices directly');
assert.match(native, /viewpoints\.add\(new CaptureViewpoint\(matrix, focal\[0\], focal\[1\], principal\[0\], principal\[1\], imageSize\[0\], imageSize\[1\]\)\)/,
  'successful AR captures should retain their exact world-space camera poses and intrinsics for the overlay');
assert.match(native, /private void drawCoverage\(Frame frame\)[\s\S]*drawViewpoints\(mvp\)/,
  'camera viewpoints and the sparse coverage cloud should share the ARCore world projection');
assert.match(native, /private void drawViewpoints\(float\[\] viewProjection\)[\s\S]*glDrawArrays\(GLES20\.GL_LINES, 0, vertexCount\)/,
  'captured viewpoints should be drawn as wireframe camera pyramids');
assert.match(native, /pose\[0\] \* x \+ pose\[4\] \* y \+ pose\[8\] \* z \+ pose\[12\]/,
  'camera frustum corners should be transformed by the captured camera-to-world pose');
assert.match(native, /final float depth = 0\.012f/,
  'capture camera pyramids should be one tenth of the previous 0.12 m size');
assert.match(native, /frames\.clear\(\); viewpoints\.clear\(\);/,
  'reset capture must clear the stored camera viewpoint overlays');
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
