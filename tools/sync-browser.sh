#!/bin/sh
# sync-browser.sh — re-vendor the browser webapp's mirrored copies.
#
# browser/ is standalone-servable: it carries byte-identical copies of the
# engine (src/ -> browser/src/) and of the app modules it reuses
# (app/js/<f>.js -> browser/app/js/<f>.js), plus the on-demand SOG codec.
# The mirror layout keeps every relative import working unchanged, so the
# sync is a plain copy — no rewriting, no build step.
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

# 2. the app modules the browser reuses (pure logic + guarded fallbacks only:
#    no capture UI, no backend, no native bridges — see the README's table)
mkdir -p browser/app/js
for f in chart data develop download img scene-cloud session_io sog store xr-dataset zip; do
  cp "app/js/$f.js" "browser/app/js/$f.js"
done

# 3. the on-demand SOG codec (loaded only when a SOG export/import needs it)
mkdir -p browser/app/models browser/app/vendor
cp app/models/splat-transform.bundle.mjs browser/app/models/
cp app/vendor/st-worker.mjs app/vendor/webp.wasm browser/app/vendor/

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
for f in chart data develop download img scene-cloud session_io sog store xr-dataset zip; do
  check "app/js/$f.js" "browser/app/js/$f.js"
done
check app/models/splat-transform.bundle.mjs browser/app/models/splat-transform.bundle.mjs
check app/vendor/st-worker.mjs browser/app/vendor/st-worker.mjs
check app/vendor/webp.wasm browser/app/vendor/webp.wasm
[ "$fail" = 0 ] && echo "browser mirror in sync ($(find browser/src browser/app -type f | wc -l) files)"
exit "$fail"
