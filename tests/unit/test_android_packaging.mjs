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

const native = readFileSync(resolve(moduleDir, 'src/main/java/space/arrival/splatxr/NativeCaptureActivity.java'), 'utf8');
assert.match(native, /text == null\) throw new IllegalStateException/,
  'capture export must report JSON serialization errors rather than dereference null');
assert.match(native, /if \(!finite\(world\[0\], world\[1\], world\[2\], confidence\) \|\| confidence < 0\.25f\)/,
  'non-finite feature points must not be serialized into seed JSON');
assert.match(native, /View\.SYSTEM_UI_FLAG_LAYOUT_STABLE \| View\.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN/,
  'camera preview should remain full bleed while controls use system-bar insets');
assert.match(native, /guideParams\.bottomMargin = bottom \+ dp\(125\)/);
assert.match(native, /frame\.transformDisplayUvCoords\(uvInput, uvOutput\);[\s\S]*?1f - uvOutput\.get\(i\)/,
  'camera preview UVs must receive the 180-degree orientation correction');
assert.match(native, /Matrix\.rotateM\(projection, 0, 180f, 0f, 0f, 1f\)/,
  'ARCore coverage points must use the same corrected display orientation');
console.log('Android asset paths, export guards, safe-area UI, and camera orientation: passed');
