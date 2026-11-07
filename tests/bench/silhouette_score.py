# How sharp is the silhouette? Renders alone cannot say: a black t-shirt on a
# black background is invisible to luminance. So the model is rendered twice
# from each pose, on black and on white, and coverage comes out exactly:
#   C_white - C_black = T   ->  alpha = 1 - T
# Scored against the matte of the same frame.
#   python tests/bench/silhouette_score.py <prefix_black> <prefix_white> <views...>
import sys, os
import numpy as np
from PIL import Image

blk, wht = sys.argv[1], sys.argv[2]
views = [int(v) for v in sys.argv[3].split(',')]
MASKS = 'data/lisa/masks'

def dilate(mask, r):
    """chebyshev dilation by r px - separable max over a (2r+1) window, numpy only"""
    out = mask.copy()
    for ax in (0, 1):
        acc = out.copy()
        for k in range(1, r + 1):
            acc |= np.roll(out, k, axis=ax) | np.roll(out, -k, axis=ax)
        out = acc
    return out

def ring(mask, lo, hi):
    """pixels between lo and hi px outside `mask`"""
    return dilate(mask, hi) & ~dilate(mask, lo)

ious, halos, bands, insides = [], [], [], []
for v in views:
    b = np.asarray(Image.open(f'scratch/{blk}_{v:03d}.png').convert('RGB'), np.float64) / 255
    w = np.asarray(Image.open(f'scratch/{wht}_{v:03d}.png').convert('RGB'), np.float64) / 255
    a = np.clip(1 - (w - b).mean(2), 0, 1)                    # rendered coverage
    m = np.asarray(Image.open(f'{MASKS}/frame_{v+1:05d}.png').convert('L').resize(
        (a.shape[1], a.shape[0]), Image.LANCZOS), np.float64) / 255
    gt = m >= 0.5
    if gt.sum() < 500:
        continue
    r = a >= 0.5
    ious.append((r & gt).sum() / max((r | gt).sum(), 1))
    halos.append(a[ring(gt, 3, 25)].mean())                   # ink outside the body
    insides.append(a[ring(~gt, 3, 25) & gt].mean() if (ring(~gt, 3, 25) & gt).any() else np.nan)
    per = np.abs(np.diff(r.astype(np.int8), axis=0)).sum() + np.abs(np.diff(r.astype(np.int8), axis=1)).sum()
    bands.append(((a > 0.1) & (a < 0.9)).sum() / max(per, 1))

print(f'views scored      {len(ious)}')
print(f'silhouette IoU    {np.mean(ious):.4f}')
print(f'halo (3-25 px out){np.mean(halos):>8.4f}   <- coverage where there should be none')
print(f'edge band         {np.mean(bands):>8.2f} px per perimeter pixel')
print(f'interior coverage {np.nanmean(insides):>8.4f}   <- should be ~1.0')
