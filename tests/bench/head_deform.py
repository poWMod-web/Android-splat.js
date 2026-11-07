# Put the body-model head ON the person's face: similarity + Laplacian deformation
# of the head region pinning the 468 landmark correspondences (head_correspond.py)
# to the triangulated landmarks (face_crops.py). Boundary (neck) stays fixed.
#   python tests/bench/head_deform.py --surface scratch/lisa_anny_surface3_head.json --corr scratch/anny_face_corr.json --face scratch/face_canon3d.json --out scratch/lisa_anny_surface3_face.json
import sys, json, numpy as np, cv2
import scipy.sparse as sp, scipy.sparse.linalg as spl
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
S = json.load(open(arg('--surface'))); V = np.array(S['vertices'], np.float64); T = np.array(S['faces'], np.int64)
labels = S['boneLabels']; bi = np.array(S['boneIndices']); bw = np.array(S['boneWeights'], np.float64)
wh = (bw * (bi == labels.index('head'))).sum(1)
C = json.load(open(arg('--corr')))['corr']; F = json.load(open(arg('--face')))['landmarks']
ids = [int(i) for i in C if i in F or int(i) in F or str(i) in F]
ids = [i for i in ids if str(i) in F]
tri = np.array([C[str(i)]['tri'] for i in ids]); bar = np.array([C[str(i)]['bary'] for i in ids]); P = np.array([F[str(i)] for i in ids])
src = (V[tri] * bar[..., None]).sum(1)
print(f'{len(ids)} correspondences; before: mean |src - target| {np.linalg.norm(src - P, axis=1).mean()*1000:.1f} mm')
# --- A. similarity on the head region (rigid + scale) --------------------------
def umeyama(s, d):
    ms, md = s.mean(0), d.mean(0); s0, d0 = s - ms, d - md
    U, D, Vt = np.linalg.svd(d0.T @ s0 / len(s)); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    Rm = U @ Sg @ Vt; sc = (D * np.diag(Sg)).sum() / (s0 ** 2).sum() * len(s); return sc, Rm, md - sc * Rm @ ms
sc, Rm, tt = umeyama(src, P)
Vs = V + wh[:, None] * (((sc * (Rm @ V.T)).T + tt) - V)
src2 = (Vs[tri] * bar[..., None]).sum(1); res2 = np.linalg.norm(src2 - P, axis=1)
print(f'after similarity (scale {sc:.3f}): mean {res2.mean()*1000:.1f} mm, p90 {np.percentile(res2,90)*1000:.1f} mm')
# robust: a landmark whose target sits far from the aligned head is a bad
# triangulation or a bad trace on the render (hair, jaw silhouette) - not a pin
cut = max(0.012, 2.5 * np.median(res2)); ok = res2 < cut
print(f'pins: {ok.sum()}/{len(ids)} kept (cut {cut*1000:.0f} mm); dropped ids {[ids[k] for k in np.where(~ok)[0]][:20]}')
ids = [i for i, k in zip(ids, ok) if k]; tri = tri[ok]; bar = bar[ok]; P = P[ok]
# --- B. Laplacian deformation of the head, landmarks as soft pins ------------
free = np.where(wh > 0.3)[0]; idx = {v: k for k, v in enumerate(free)}; nf = len(free)
rows, cols, vals = [], [], []
adj = [set() for _ in range(len(V))]
for a, b, c in T: adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
r = 0
for v in free:
    nb = list(adj[v]); rows.append(r); cols.append(idx[v]); vals.append(1.0)
    for u in nb:
        if u in idx: rows.append(r); cols.append(idx[u]); vals.append(-1.0 / len(nb))
    r += 1
Lm = sp.csr_matrix((vals, (rows, cols)), shape=(r, nf))
delta = Lm @ Vs[free]                                              # keep the local shape (Laplacian coordinates)
# boundary: neighbours outside the free set contribute their fixed positions
bfix = np.zeros((r, 3)); rr = 0
for v in free:
    nb = list(adj[v])
    for u in nb:
        if u not in idx: bfix[rr] += Vs[u] / len(nb)
    rr += 1
W_PIN = 1.0
prow, pcol, pval = [], [], []
for k in range(len(ids)):
    for j in range(3):
        if tri[k, j] in idx: prow.append(k); pcol.append(idx[tri[k, j]]); pval.append(bar[k, j] * W_PIN)
Pm = sp.csr_matrix((pval, (prow, pcol)), shape=(len(ids), nf))
# weak positional prior: vertices far from any pin (back of the head, hair cap)
# otherwise float freely and a near-singular system blows up
W_PRIOR = 0.1
Im = sp.identity(nf, format='csr') * W_PRIOR
# delta = Lm @ Vs[free] already carries the fixed neighbours' share (the rows only
# hold inside weights), so the right-hand side is delta itself, not delta + bfix
A = sp.vstack([Lm, Pm, Im]).tocsr(); B = np.vstack([delta, P * W_PIN, Vs[free] * W_PRIOR])
AtA = (A.T @ A).tocsc(); Vnew = Vs.copy()
for j in range(3): Vnew[free, j] = spl.spsolve(AtA, A.T @ B[:, j])
src3 = (Vnew[tri] * bar[..., None]).sum(1); res3 = np.linalg.norm(src3 - P, axis=1)
print(f'after Laplacian deformation: mean {res3.mean()*1000:.1f} mm, p90 {np.percentile(res3,90)*1000:.1f} mm; max vertex move {np.linalg.norm(Vnew - Vs, axis=1).max()*1000:.0f} mm')
S2 = dict(S); S2['vertices'] = Vnew.round(5).tolist(); S2['faceDeformed'] = {'pins': len(ids), 'residualMm': float(res3.mean() * 1000)}
json.dump(S2, open(arg('--out'), 'w')); print('wrote', arg('--out'))
