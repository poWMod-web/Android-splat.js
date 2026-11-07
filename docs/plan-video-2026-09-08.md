# Video input, done properly — plan 2026-09-08

Users film places; a phone video is the capture path that turns the statue
failure (34 photos, 12° apart) into a truck-class dataset. Today's extractor
(`src/io/video.js`) plays a `<video>` at 3× and scores ~10 presented frames per
second with a 320-px Laplacian variance, then seeks back for JPEGs. It works,
but it is the 2024 recipe. State of the art (2026-09) and what we take from it:

## What the field does

- **Sharp Frames (Reflct)** — the community standard for 3DGS dataset prep
  (free app + `sharp-frames-python`). Extraction: every frame (Mediabunny in the
  app since the 2026 update, 4–5× faster; ffmpeg `fps=10` in the CLI). Scoring:
  `normalized_laplacian_tenengrad_v1` — grayscale, long edge 512 (area
  downsample), Gaussian 5×5 σ 1 denoise, then
  `expm1(0.5·log1p(LaplacianVar) + 0.5·log1p(mean Sobel²))`. Selection: best-N
  with a minimum buffer and a *distribution* term (70 % sharpness / 30 %
  position-in-segment), batched (sharpest of every 5, buffer 2), and outlier
  removal (window 15, sensitivity 60: a frame is dropped when it dips below its
  neighbours, not below a global threshold). Colour: detects iPhone HDR
  (PQ/HLG, BT.2020) and tone-maps to SDR via zscale — HDR video fed raw gives
  washed-out, inconsistent frames.
- **Reflct capture guide**: extract ≥ 10 fps, keep the sharpest of every 5
  (one frame / 0.5 s), ~80 % overlap between kept frames, fixed ISO, shutter
  = 1/(4·fps); "more images is not always better".
- **SLAM-style keyframing** (MonoGS/RTGS/LVI-GS): a frame becomes a keyframe
  on motion (rotation/translation beyond a threshold, or covisibility below
  ~0.9), not on time — redundancy from a slow or paused camera is the other
  half of dataset quality. Cheap proxy without poses: mean optical-flow
  magnitude / phase-correlation shift between consecutive frames.
- **WebCodecs**: `VideoDecoder` + a demuxer gives every frame, exact
  timestamps, hardware decode, rotation metadata. Mediabunny (MPL-2.0, zero
  deps, ESM bundle 290 KB min) wraps demux + decode for MP4/MOV/WebM/MKV with
  `VideoSampleSink.samples()` (all frames) and `samplesAtTimestamps()` (only
  the winners, no duplicate decodes) and `CanvasSink` (rotation-aware
  scaling). The `<video>`/rVFC path stays as fallback.

## Design

1. **Decode** (`src/io/video.js` v2): Mediabunny `Input(BlobSource)`; read
   track metadata (display size after rotation, fps, duration, colour
   transfer). Pass 1 iterates every frame in presentation order, draws each
   into a 512-long-edge canvas (rotation applied), scores it, closes it.
   Pass 2 decodes only the selected timestamps at full resolution and encodes
   JPEG q0.95 (the rest of the pipeline treats them as photo files). HDR
   tracks (PQ/HLG) are drawn through the canvas, which tone-maps in Chrome;
   we log the transfer and flag it in the UI.
2. **Score** every frame: sharp-frames' hybrid metric (Gaussian 5×5 σ1 →
   Laplacian variance + Tenengrad, log-combined), plus luminance mean and
   clipped-pixel fraction (exposure outliers) and a motion proxy: mean
   absolute difference of the 128-px gray against the previous frame (0 =
   camera paused).
3. **Select**: (a) outlier removal against a local window (blur dips, not
   texture); (b) motion-aware windows — walk the timeline accumulating the
   motion proxy and close a window every time the budget for ~80 % overlap is
   spent (with a time cap so a static camera still yields frames sparsely and
   a fast pan yields more); pick the sharpest surviving frame per window;
   (c) device cap (phone 140, desktop 300) applied by widening windows, never
   by dropping the sharpest.
4. **Diagnostics**: return per-frame scores + selection so the app can draw a
   Sharp-Frames-style timeline later; bench mode `?video=/data/downloads/x`
   runs extraction → solve → train so old vs new selection is measured on the
   same file (registration, 30k held-out PSNR).
5. **Fallback**: no WebCodecs / undecodable codec → the existing `<video>`
   path with the new scorer and selector (shared code).

## Measurement

Charleston 4K/30 (`data/downloads/charleston.webm`, VP9, BT.709) and the
camping source if the user has it. A/B: current extractor vs v2 on the same
video, same target count → registered frames, sparse points, 30k eval8 PSNR.
Then the iPhone HDR case (needs a user clip).

## Out of scope for now

Rolling-shutter correction, deblurring (DeblurGS/GeMS), learned frame
selection, 360 video (separate rig path).
