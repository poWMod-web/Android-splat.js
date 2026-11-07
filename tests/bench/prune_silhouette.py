# Silhouette prune: drop Gaussians whose centre lands OUTSIDE the person matte
# in the views that see them. Masking the loss leaves empty space unconstrained
# (a splat nothing can see is free to be anything), so the leftovers grow into
# giant flares. The masks already say where the subject is NOT — this applies
# that to the model instead of only to the loss.
import sys, json, os
import numpy as np
from PIL import Image

ply_in, recon_p, mask_dir, ply_out = sys.argv[1:5]
KEEP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5   # min fraction of seeing views that call it subject
SIZE = float(sys.argv[6]) if len(sys.argv) > 6 else 0.04  # max splat axis, as a fraction of the subject's diagonal

# --- PLY ---
with open(ply_in, 'rb') as f:
    hdr = b''
    while not hdr.endswith(b'end_header\n'):
        hdr += f.read(1)
    h = hdr.decode('ascii')
    n = int([l for l in h.split('\n') if l.startswith('element vertex')][0].split()[-1])
    names = [l.split()[-1] for l in h.split('\n') if l.startswith('property float')]
    body = np.frombuffer(f.read(n * len(names) * 4), dtype=np.float32).reshape(n, len(names))
print(f'{n} splats, {len(names)} properties')
X = body[:, :3].astype(np.float64)

# --- cameras ---
rc = json.load(open(recon_p))
frames = rc['frames']
cams = rc['cams']
print(f'{len(cams)} cams')

# splat extent in world units (max axis), and the disc we sample per view
idx0 = {k: j for j, k in enumerate(names)}
smax = np.exp(body[:, [idx0['scale_0'], idx0['scale_1'], idx0['scale_2']]].astype(np.float64)).max(1)
SIG = 2.0                       # sample out to 2 sigma
RIM = [(0, 0)] + [(np.cos(a), np.sin(a)) for a in np.arange(8) * (np.pi / 4)]
RIM_MIN = int(sys.argv[7]) if len(sys.argv) > 7 else 8   # of 9 disc samples that must be subject

seen = np.zeros(n, np.int32)
subj = np.zeros(n, np.int32)
cache = {}
for ci, c in enumerate(cams):
    fr = frames[c['imgIdx']]
    fw, fh = fr['fw'], fr['fh']
    R = np.array(c['R'], np.float64).reshape(3, 3)
    t = np.array(c['t'], np.float64)
    Xc = X @ R.T + t
    z = Xc[:, 2]
    ok = z > 1e-6
    u = np.full(n, -1.0); v = np.full(n, -1.0)
    u[ok] = c['f'] * Xc[ok, 0] / z[ok] + c['cx']
    v[ok] = (c.get('fy', c['f'])) * Xc[ok, 1] / z[ok] + c['cy']
    inf = ok & (u >= 0) & (u < fw) & (v >= 0) & (v < fh)
    if not inf.any():
        continue
    mn = os.path.splitext(fr['name'])[0] + '.png'
    if mn not in cache:
        m = np.asarray(Image.open(os.path.join(mask_dir, mn)).convert('L'))
        cache[mn] = m
        if len(cache) > 200: cache.pop(next(iter(cache)))
    m = cache[mn]
    mh, mw = m.shape
    seen += inf
    idx = np.where(inf)[0]
    # FOOTPRINT test, not a point test. A Gaussian's centre can sit inside the
    # silhouette while the splat itself spills metres past it — that is exactly
    # what a flare or a background curtain is. Sample the projected disc: the
    # centre plus 8 points on its rim, and call the splat subject only if
    # nearly all of them land on the matte.
    rpx = np.clip(c['f'] * SIG * smax[idx] / z[idx], 0.5, 80.0)
    hits = np.zeros(idx.size, np.int32)
    for du, dv in RIM:
        su = (u[inf] + du * rpx) / fw * mw
        sv = (v[inf] + dv * rpx) / fh * mh
        ui = np.clip(su.astype(np.int32), 0, mw - 1)
        vi = np.clip(sv.astype(np.int32), 0, mh - 1)
        hits += (m[vi, ui] >= 230)
    subj[idx] += (hits >= RIM_MIN)

frac = np.where(seen > 0, subj / np.maximum(seen, 1), 0.0)
keep = (seen > 0) & (frac >= KEEP)
print(f'silhouette test kept {keep.sum()} ({100*keep.mean():.1f}%)')

# Size cap. The silhouette test alone misses the worst offenders: a flare
# CENTRED on the subject but metres across still passes it. With the background
# masked out nothing bounds a splat's extent into empty space, so cap it
# against the subject's own size (the central 96% of the cloud IS the subject).
sc = smax
lo, hi = np.percentile(X[keep], [2, 98], axis=0)
span = float(np.linalg.norm(hi - lo))
cap = SIZE * span
big = sc > cap
print(f'size cap {cap:.3f} ({SIZE:.0%} of the {span:.2f} subject diagonal): drops {(big & keep).sum()}')
keep &= ~big
print(f'kept: {keep.sum()} ({100*keep.mean():.1f}%)')

out = body[keep]
h2 = h.replace(f'element vertex {n}', f'element vertex {keep.sum()}')
with open(ply_out, 'wb') as f:
    f.write(h2.encode('ascii'))
    f.write(out.tobytes())
print('wrote', ply_out, os.path.getsize(ply_out) // 10**6, 'MB')
