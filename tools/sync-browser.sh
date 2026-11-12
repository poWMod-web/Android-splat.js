#!/bin/sh
# sync-browser.sh — re-vendor the browser shell's copies of the app.
#
# browser/ is standalone-servable: it carries byte-identical copies of the
# engine (src/ -> browser/src/) and of the whole app (app/{js,css,avatar}
# -> browser/app/{js,css,avatar}, plus the viewer vendor files and the
# on-demand SOG codec). Keeping the app/ subtree one level under the root
# preserves every relative import untouched (../avatar, ../../src, ...),
# so the sync is a plain copy — no rewriting, no build step. Its
# index.html is GENERATED from app/index.html (install bits out, asset
# paths re-rooted under app/, shell marker in), so the browser boots the
# same UI with the browser profile (SHELL in app/js/app.js).
#
# Run from the repo root after changing any mirrored source:
#   sh tools/sync-browser.sh
# tests/unit/test_browser_vendor.mjs fails loudly on drift.
set -eu
cd "$(dirname "$0")/.."

# 1. the engine, whole (browser can't train without any of it)
rm -rf browser/src
mkdir -p browser/src
cp -r src/. browser/src/

# 2. the app, whole (retired the old browser/app/js partial mirror)
rm -rf browser/app browser/js browser/css browser/avatar browser/vendor browser/models
mkdir -p browser/app/js browser/app/css browser/app/vendor browser/app/models
cp -r app/js/. browser/app/js/
cp -r app/css/. browser/app/css/
cp -r app/avatar/. browser/app/avatar/
cp app/vendor/playcanvas.min.mjs app/vendor/st-worker.mjs app/vendor/webp.wasm browser/app/vendor/
cp app/models/splat-transform.bundle.mjs browser/app/models/
# the avatar ML models stay out: the browser profile never runs the avatar route

# 3. the generated entry: app/index.html minus install bits, re-rooted
# under app/, plus the shell marker
sed -e '/manifest\.webmanifest/d' -e '/icon-192\.png/d' -e '/icon-180\.png/d' \
  -e 's|src="js/|src="app/js/|' -e 's|href="css/|href="app/css/|' \
  -e 's|<script type="module" src="app/js/app\.js"></script>|<script>window.__SPLAT_SHELL="browser";</script><script type="module" src="app/js/app.js"></script>|' \
  app/index.html > browser/index.html

# 4. verify: every mirrored file byte-identical to its source
fail=0
check() {
  if ! cmp -s "$1" "$2"; then echo "DRIFT: $2 differs from $1"; fail=1; fi
}
check_dir() {
  (cd "$1" && find . -type f | sort) > /tmp/syncb_src
  (cd "$2" && find . -type f | sort) > /tmp/syncb_dst
  if ! cmp -s /tmp/syncb_src /tmp/syncb_dst; then echo "DRIFT: file list differs between $1 and $2"; fail=1; fi
  while IFS= read -r f; do check "$1/$f" "$2/$f"; done < /tmp/syncb_src
}
check_dir src browser/src
check_dir app/js browser/app/js
check_dir app/css browser/app/css
check_dir app/avatar browser/app/avatar
check app/vendor/playcanvas.min.mjs browser/app/vendor/playcanvas.min.mjs
check app/vendor/st-worker.mjs browser/app/vendor/st-worker.mjs
check app/vendor/webp.wasm browser/app/vendor/webp.wasm
check app/models/splat-transform.bundle.mjs browser/app/models/splat-transform.bundle.mjs
[ "$fail" = 0 ] && echo "browser mirror in sync ($(find browser/src browser/app -type f | wc -l) files)"
exit "$fail"
