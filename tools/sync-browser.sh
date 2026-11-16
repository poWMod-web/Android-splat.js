#!/bin/sh
# sync-browser.sh — refresh the browser shell's shared app copies.
#
# browser/ is standalone-servable. The engine, shared UI helpers/styles and
# assets are mirrored from this repository, while the Browser project owns
# the files that contain its viewer and background-policy boundaries:
#
#   app/js/app.js       Browser integration and model restore policy
#   app/js/pcview.js    Browser model viewer
#   app/js/headless.js  Browser-safe compatibility surface used by app.js
#
# Those files are deliberately preserved, not overwritten from app/. The
# Android-only service driver train-headless.js is deliberately omitted.
# This avoids an Android viewer/background change silently entering Browser;
# Browser changes to the monolithic app.js must be merged deliberately.
#
# Run from the repo root after changing mirrored shared source:
#   sh tools/sync-browser.sh
set -eu
cd "$(dirname "$0")/.."

BROWSER_OWNED_JS="app.js pcview.js headless.js"
ANDROID_ONLY_JS="train-headless.js"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# Keep Browser-owned files before rebuilding its standalone app tree.
mkdir -p "$tmp/owned"
for f in $BROWSER_OWNED_JS; do
  if [ ! -f "browser/app/js/$f" ]; then
    echo "MISSING Browser-owned file: browser/app/js/$f" >&2
    exit 1
  fi
  cp "browser/app/js/$f" "$tmp/owned/$f"
done

# 1. The engine, whole (Browser cannot train without it).
rm -rf browser/src
mkdir -p browser/src
cp -r src/. browser/src/

# 2. Rebuild shared app assets, excluding Browser-owned and Android-only JS.
rm -rf browser/app browser/js browser/css browser/avatar browser/vendor browser/models
mkdir -p browser/app/js browser/app/css browser/app/vendor browser/app/models
for src in app/js/*; do
  [ -f "$src" ] || continue
  f="${src##*/}"
  skip=0
  for owned in $BROWSER_OWNED_JS $ANDROID_ONLY_JS; do
    [ "$f" = "$owned" ] && skip=1
  done
  [ "$skip" = 1 ] || cp "$src" "browser/app/js/$f"
done
for f in $BROWSER_OWNED_JS; do cp "$tmp/owned/$f" "browser/app/js/$f"; done
cp -r app/css/. browser/app/css/
cp -r app/avatar/. browser/app/avatar/
cp app/vendor/playcanvas.min.mjs app/vendor/st-worker.mjs app/vendor/webp.wasm browser/app/vendor/
cp app/models/splat-transform.bundle.mjs browser/app/models/
# Avatar ML models stay out: the Browser profile never runs the avatar route.

# 3. Generate the Browser entry: install bits out, paths re-rooted, shell set.
sed -e '/manifest\.webmanifest/d' -e '/icon-192\.png/d' -e '/icon-180\.png/d' \
  -e 's|src="js/|src="app/js/|' -e 's|href="css/|href="app/css/|' \
  -e 's|<script type="module" src="app/js/app\.js"></script>|<script>window.__SPLAT_SHELL="browser";</script><script type="module" src="app/js/app.js"></script>|' \
  app/index.html > browser/index.html

# 4. Verify shared files are identical. Browser-owned files are checked for
# presence only; train-headless.js must remain absent.
fail=0
check() {
  if ! cmp -s "$1" "$2"; then echo "DRIFT: $2 differs from $1"; fail=1; fi
}
check_dir() {
  (cd "$1" && find . -type f | sort) > "$tmp/src-list"
  (cd "$2" && find . -type f | sort) > "$tmp/dst-list"
  if ! cmp -s "$tmp/src-list" "$tmp/dst-list"; then echo "DRIFT: file list differs between $1 and $2"; fail=1; fi
  while IFS= read -r f; do check "$1/$f" "$2/$f"; done < "$tmp/src-list"
}
check_dir src browser/src
check_dir app/css browser/app/css
check_dir app/avatar browser/app/avatar
for src in app/js/*; do
  [ -f "$src" ] || continue
  f="${src##*/}"
  skip=0
  for owned in $BROWSER_OWNED_JS $ANDROID_ONLY_JS; do [ "$f" = "$owned" ] && skip=1; done
  [ "$skip" = 1 ] || check "$src" "browser/app/js/$f"
done
for f in $BROWSER_OWNED_JS; do
  [ -f "browser/app/js/$f" ] || { echo "MISSING Browser-owned file: $f"; fail=1; }
done
[ ! -e browser/app/js/train-headless.js ] || { echo "DRIFT: Android-only train-headless.js entered Browser"; fail=1; }
check app/vendor/playcanvas.min.mjs browser/app/vendor/playcanvas.min.mjs
check app/vendor/st-worker.mjs browser/app/vendor/st-worker.mjs
check app/vendor/webp.wasm browser/app/vendor/webp.wasm
check app/models/splat-transform.bundle.mjs browser/app/models/splat-transform.bundle.mjs
[ "$fail" = 0 ] && echo "browser shared files in sync; viewer/background files preserved ($(find browser/src browser/app -type f | wc -l) files)"
exit "$fail"
