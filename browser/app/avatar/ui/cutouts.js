// cutouts.js — every frame with the room removed, in a strip that scrolls
// sideways (wheel, drag, swipe).
//
// This used to be a gate: "Looks right / Not a person", and nothing expensive
// had run when the answer was no. The person is chosen one screen earlier now,
// on the start-training card, and a run that has started is never interrupted
// again (the user, 2026-09-18) — so the strip is SHOWN, for as long as the
// caller leaves the card up, and the solve gets under way behind it. A matte
// that finds nobody still bails out on its coverage (index.js prepareCapture).

/** Paint the cut-outs into `card`. Returns at once; the frames fill in as they
 *  decode, and stop as soon as the card is taken off the page. */
export function cutoutsView(card, frames, { coverage }) {
  card.innerHTML = `
    <div class="vid-head"><b>Cut out of the room</b><span class="prep-sub">${frames.length} frames · the person covers ${(coverage * 100).toFixed(0)}% of a frame</span></div>
    <div class="av-strip" id="av-strip"></div>
    <p class="fine av-fine">The avatar is built from what is white here — hair and finger tips should be there, the room should not. Scroll sideways to see every frame; the solve is already running behind this.</p>`;
  const strip = card.querySelector('#av-strip');
  // a vertical wheel scrolls the strip sideways; dragging scrolls it too
  strip.addEventListener('wheel', (e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { strip.scrollLeft += e.deltaY; e.preventDefault(); } }, { passive: false });
  let drag = null;
  strip.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, left: strip.scrollLeft }; strip.setPointerCapture(e.pointerId); });
  strip.addEventListener('pointermove', (e) => { if (drag) strip.scrollLeft = drag.left - (e.clientX - drag.x); });
  strip.addEventListener('pointerup', () => { drag = null; }); strip.addEventListener('pointercancel', () => { drag = null; });
  (async () => {
    for (const f of frames) {
      if (!card.isConnected || !card.contains(strip)) return;   // the card moved on
      const [bmp, mask] = await Promise.all([createImageBitmap(f.source), createImageBitmap(f.mask)]);
      const h = 320, w = Math.round((bmp.width / bmp.height) * h);   // drawn at 2x, shown at 160 px
      const cv = document.createElement('canvas'); cv.width = w; cv.height = h; cv.style.height = '160px'; cv.style.width = `${w / 2}px`;
      const g = cv.getContext('2d');
      // photo × matte over white
      g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
      const tmp = new OffscreenCanvas(w, h); const t = tmp.getContext('2d');
      t.drawImage(bmp, 0, 0, w, h);
      t.globalCompositeOperation = 'destination-in';
      // the matte is grayscale: use its luminance as alpha
      const m = new OffscreenCanvas(w, h); const mg = m.getContext('2d');
      mg.drawImage(mask, 0, 0, w, h);
      const md = mg.getImageData(0, 0, w, h); const ad = mg.createImageData(w, h);
      for (let p = 0; p < w * h; p++) { ad.data[p * 4 + 3] = md.data[p * 4]; }
      mg.putImageData(ad, 0, 0);
      t.drawImage(m, 0, 0);
      g.drawImage(tmp, 0, 0);
      bmp.close(); mask.close();
      cv.title = f.name || ''; strip.appendChild(cv);
      await new Promise((r) => setTimeout(r, 0));
    }
  })().catch(() => {});
}
