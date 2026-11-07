# Register the fitted body-model head to the triangulated face landmarks.
# The body fit only saw pose landmarks + a Chamfer pull, so the head is a
# generic head roughly in place. Point-to-surface ICP of the 468 skin landmarks
# (iris points excluded) against the head region solves a similarity transform
# (scale, rotation, translation) that puts nose on nose and chin on chin; it is
# blended into the surface by the head bone weight so the neck stays attached.
#   python tests/bench/head_register.py --surface scratch/lisa_anny_surface3_clean.json \
#          --face scratch/face_canon3d.json --out scratch/lisa_anny_surface3_head.json
import sys, json, numpy as np
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
S = json.load(open(arg('--surface'))); F = json.load(open(arg('--face')))
V = np.array(S['vertices'], np.float64); T = np.array(S['faces'], np.int64)
labels = S['boneLabels']; bi = np.array(S['boneIndices']); bw = np.array(S['boneWeights'], np.float64)
hb = labels.index('head'); wh = (bw * (bi == hb)).sum(1)                     # head-bone weight per vertex
P = np.array([F['landmarks'][k] for k in sorted(F['landmarks'], key=int) if int(k) < 468], np.float64)
headV = np.where(wh > 0.5)[0]; headT = T[np.isin(T, headV).all(1)]
print(f'head region: {len(headV)} verts, {len(headT)} tris; {len(P)} skin landmarks')

def closest_on_tris(pts, verts, tris):
    # closest point on a triangle soup for each query (vectorised per chunk)
    A, B, C = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    out = np.zeros_like(pts); dist = np.full(len(pts), np.inf)
    for i0 in range(0, len(pts), 64):
        p = pts[i0:i0 + 64][:, None, :]
        ab, ac, ap = B - A, C - A, p - A
        d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
        bp = p - B; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
        cp = p - C; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        denom = va + vb + vc
        v = np.clip(vb / np.where(denom == 0, 1, denom), 0, 1); w = np.clip(vc / np.where(denom == 0, 1, denom), 0, 1)
        q = A + ab * v[..., None] + ac * w[..., None]
        # edge/vertex regions
        e = np.where(((d1 <= 0) & (d2 <= 0))[..., None], A, q)
        e = np.where(((d3 >= 0) & (d4 <= d3))[..., None], B, e)
        e = np.where(((d6 >= 0) & (d5 <= d6))[..., None], C, e)
        t_ab = np.clip(d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0, 1); e = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], A + ab * t_ab[..., None], e)
        t_ac = np.clip(d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0, 1); e = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], A + ac * t_ac[..., None], e)
        t_bc = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)), 0, 1)
        e = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], B + (C - B) * t_bc[..., None], e)
        dd = np.linalg.norm(e - p, axis=-1); j = dd.argmin(1)
        out[i0:i0 + 64] = e[np.arange(len(j)), j]; dist[i0:i0 + 64] = dd[np.arange(len(j)), j]
    return out, dist

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src)); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    R = U @ Sg @ Vt; sc = (D * np.diag(Sg)).sum() / (s0 ** 2).sum(0).sum() * len(src)
    t = md - sc * R @ ms
    return sc, R, t

Vh = V.copy()
q0, d0 = closest_on_tris(P, Vh, headT); print(f'before: landmark->surface distance mean {d0.mean()*1000:.1f} mm, p90 {np.percentile(d0,90)*1000:.1f} mm')
sc_all, R_all, t_all = 1.0, np.eye(3), np.zeros(3)
for it in range(25):
    q, d = closest_on_tris(P, Vh, headT)
    keep = d < max(0.006, 2.5 * np.median(d))                                  # robust: ignore hair/occluded outliers
    sc, R, t = umeyama(q[keep], P[keep])
    Vh = (sc * (R @ Vh.T)).T + t
    sc_all, R_all, t_all = sc * sc_all, R @ R_all, sc * R @ t_all + t
q1, d1 = closest_on_tris(P, Vh, headT)
ang = np.degrees(np.arccos(np.clip((np.trace(R_all) - 1) / 2, -1, 1)))
print(f'after:  mean {d1.mean()*1000:.1f} mm, p90 {np.percentile(d1,90)*1000:.1f} mm | head similarity: scale {sc_all:.3f}, rotation {ang:.1f} deg, shift {np.linalg.norm(t_all + (sc_all*R_all - np.eye(3)) @ V[headV].mean(0))*1000:.0f} mm')
# blend by head weight so the neck stays attached
Vout = V + wh[:, None] * (Vh - V)
S2 = dict(S); S2['vertices'] = Vout.round(5).tolist()
S2['headRegistered'] = {'scale': sc_all, 'rotationDeg': float(ang), 'landmarkDistMm': {'before': float(d0.mean()*1000), 'after': float(d1.mean()*1000)}}
json.dump(S2, open(arg('--out'), 'w')); print('wrote', arg('--out'))
