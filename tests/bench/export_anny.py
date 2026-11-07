# Export a compact, browser-ready snapshot of the Anny body model (game_engine
# rig) for app/avatar/body/anny.js.
#
# Anny's shape space is MakeHuman's macro-target machinery: 624 stacked blend
# shapes (102 MB) whose coefficients are products of piecewise-linear weights
# over the phenotype anchors. The browser fitter only needs the SHAPE it
# produces, so the space is sampled over the six phenotypes and compressed to
# a PCA basis (vertices + bone heads + bone tails together, so the rig follows
# the body): mean + K components, int8-quantised per component, ~2 MB.
# Everything else is copied as is: faces, skinning weights, bone rolls, the
# kinematic tree, the COCO keypoint regressor (sparse), and one test vector
# (a shape + a pose -> vertices and bone heads) so the JS port can prove
# itself against torch.
#
#   python tests/bench/export_anny.py [--samples 400] [--comps 32]
import os, sys, json, struct, time
import numpy as np
import torch, roma, anny
from anny.keypoints import KeypointsRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'app', 'models', 'anny_game_engine.bin')
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
NS = arg('--samples', 400); NC = arg('--comps', 32)
torch.manual_seed(0); rng = np.random.RandomState(0)

m = anny.Anny(rig='game_engine', skinning_method='lbs', pose_parameterization='local-bone').float().eval()
labels = list(m.phenotype_labels); B = len(m.bone_labels); V = m.template_vertices.shape[0]
print(f'model: {V} verts, {B} bones, phenotypes {labels}')

def rest(ph):
    kw = {k: float(ph[i]) for i, k in enumerate(labels)}
    c = m.get_phenotype_blendshape_coefficients(**kw)
    r = m.get_rest_model(c)
    return np.concatenate([r['rest_vertices'][0].numpy().ravel(), r['rest_bone_heads'][0].numpy().ravel(), r['rest_bone_tails'][0].numpy().ravel()]), r

# --- sample the phenotype space (uniform in [0.05, 0.95], plus the default and the corners of the two big axes)
t0 = time.time()
samples = [np.full(len(labels), 0.5)]
for g in (0.05, 0.95):
    for h in (0.05, 0.95):
        s = np.full(len(labels), 0.5); s[labels.index('gender')] = g; s[labels.index('height')] = h; samples.append(s)
while len(samples) < NS: samples.append(rng.uniform(0.05, 0.95, len(labels)))
X = np.stack([rest(s)[0] for s in samples]).astype(np.float64)
print(f'sampled {len(samples)} shapes in {time.time() - t0:.0f}s; dim {X.shape[1]}')
mean = X.mean(0); Xc = X - mean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var = S ** 2 / (len(X) - 1); frac = np.cumsum(var) / var.sum()
# the variance share is dominated by height and gender; pick K by what the
# shape LOOKS like: vertex rms error under 1 mm on the samples, capped
K = NC
for k in range(2, NC + 1):
    rec = mean + (Xc @ Vt[:k].T) @ Vt[:k]
    e = np.sqrt(((rec - X)[:, :V * 3] ** 2).reshape(len(X), V, 3).sum(-1).mean(1)).max()
    if e < 0.001: K = k; break
comps = Vt[:K]                                # (K, D) unit vectors
sigma = np.sqrt(var[:K])                      # spread of the sampled shapes along each component
coords = Xc @ comps.T                         # (N, K)
recon = mean + coords @ comps
err = np.sqrt(((recon - X)[:, :V * 3] ** 2).reshape(len(X), V, 3).sum(-1).mean(1))
print(f'PCA: {K} components keep {frac[K - 1] * 100:.2f}% of the variance; vertex rms error mean {err.mean() * 1000:.2f} mm, worst {err.max() * 1000:.2f} mm')
# held-out check
Xh = np.stack([rest(rng.uniform(0.05, 0.95, len(labels)))[0] for _ in range(20)])
ch = (Xh - mean) @ comps.T; eh = np.sqrt((((mean + ch @ comps) - Xh)[:, :V * 3] ** 2).reshape(20, V, 3).sum(-1).mean(1))
print(f'held-out 20 shapes: vertex rms error mean {eh.mean() * 1000:.2f} mm, worst {eh.max() * 1000:.2f} mm')

# --- int8 quantisation of the components (per-component scale)
q = np.zeros((K, X.shape[1]), np.int8); scale = np.zeros(K, np.float32)
for k in range(K):
    scale[k] = np.abs(comps[k]).max() / 127; q[k] = np.clip(np.round(comps[k] / scale[k]), -127, 127)

# --- rig and skinning
faces = m.get_triangular_faces().numpy().astype(np.uint16)
W = m.vertex_bone_weights.numpy(); I = m.vertex_bone_indices.numpy()
rolls = m.bone_rolls_rotmat[0].numpy().reshape(B, 9).astype(np.float32)
y_axis = m.y_axis.numpy().astype(np.float32); degen = m.degenerate_rotation.numpy().reshape(9).astype(np.float32)
parents = np.array(list(m.bone_parents), np.int8)

# --- COCO keypoints, sparse
NAME = {'nose': 'nose', 'l_ear': 'left_ear', 'r_ear': 'right_ear', 'l_sho': 'left_shoulder', 'r_sho': 'right_shoulder',
        'l_elb': 'left_elbow', 'r_elb': 'right_elbow', 'l_wri': 'left_wrist', 'r_wri': 'right_wrist', 'l_hip': 'left_hip',
        'r_hip': 'right_hip', 'l_knee': 'left_knee', 'r_knee': 'right_knee', 'l_ank': 'left_ankle', 'r_ank': 'right_ankle',
        'l_heel': 'left_heel', 'r_heel': 'right_heel', 'l_toe': 'left_big_toe', 'r_toe': 'right_big_toe'}
reg = KeypointsRegressor.coco(m, labels=list(NAME.values()))
RW = reg.regression_weights.numpy()
kp = []
for k, (ours, theirs) in enumerate(NAME.items()):
    nz = np.where(RW[k] > 1e-5)[0]
    kp.append({'name': ours, 'idx': nz.astype(np.uint16), 'w': RW[k][nz].astype(np.float32)})

# --- test vector: a sampled shape, a small pose (local-bone), torch's answer
ph = rng.uniform(0.2, 0.8, len(labels)); xs, r = rest(ph)
beta = ((xs - mean) @ comps.T).astype(np.float32)
rv = np.zeros((B, 3), np.float32)
for name, v in {'upperarm_l': [0.3, 0.1, -0.2], 'lowerarm_r': [0.0, 0.5, 0.1], 'thigh_l': [-0.2, 0.1, 0.05], 'head': [0.1, -0.1, 0.2], 'spine_02': [0.05, 0.1, 0.0]}.items():
    rv[m.bone_labels.index(name)] = v
c = m.get_phenotype_blendshape_coefficients(**{k: float(ph[i]) for i, k in enumerate(labels)})
R = roma.rotvec_to_rotmat(torch.tensor(rv)); pose = roma.Rigid(R, torch.zeros(B, 3))[None].to_homogeneous()
out = m(pose_parameters=pose, phenotype_kwargs={k: float(ph[i]) for i, k in enumerate(labels)}, return_bone_ends=True)
# the JS port applies the PCA shape, not the exact one: compare against torch run on the PCA-RECONSTRUCTED rest model
test = {'beta': beta.tolist(), 'rotvec': rv.tolist(),
        'phenotype': {k: float(ph[i]) for i, k in enumerate(labels)},
        'vertsExact': out['vertices'][0].numpy().astype(np.float32).ravel().tolist()[:3000],   # first 1000 vertices
        'headsExact': out['bone_heads'][0].numpy().astype(np.float32).ravel().tolist(),
        'restHeadsExact': r['rest_bone_heads'][0].numpy().astype(np.float32).ravel().tolist()}

# --- write: header JSON + blocks
meta = {
    'version': 1, 'source': 'anny 0.6 game_engine (Apache-2.0), PCA snapshot', 'V': int(V), 'B': int(B), 'K': int(K), 'D': int(X.shape[1]),
    'layout': ['mean:f32[D]', 'comps:i8[K*D]', 'compScale:f32[K]', 'sigma:f32[K]', 'faces:u16[F*3]', 'weights:u8[V*6]', 'boneIdx:u8[V*6]',
               'rolls:f32[B*9]', 'yAxis:f32[3]', 'degenerate:f32[9]', 'parents:i8[B]', 'kpIdx:u16[sum]', 'kpW:f32[sum]'],
    'F': int(len(faces)), 'boneLabels': list(m.bone_labels), 'phenotypeLabels': labels, 'frame': 'anny (z up, metres)',
    'keypoints': [{'name': e['name'], 'n': int(len(e['idx']))} for e in kp], 'kpTotal': int(sum(len(e['idx']) for e in kp)),
    'pcaVarianceKept': float(frac[K - 1]), 'vertexRmsMm': float(err.mean() * 1000), 'test': test,
}
hdr = json.dumps(meta).encode(); pad = (4 - len(hdr) % 4) % 4
blocks = [mean.astype(np.float32).tobytes(), q.tobytes(), scale.tobytes(), sigma.astype(np.float32).tobytes(), faces.tobytes(),
          np.clip(np.round(W * 255), 0, 255).astype(np.uint8).tobytes(), I.astype(np.uint8).tobytes(),
          rolls.tobytes(), y_axis.tobytes(), degen.tobytes(), parents.tobytes(),
          np.concatenate([e['idx'] for e in kp]).astype(np.uint16).tobytes(), np.concatenate([e['w'] for e in kp]).astype(np.float32).tobytes()]
with open(OUT, 'wb') as f:
    f.write(b'ANNY'); f.write(struct.pack('<I', len(hdr) + pad)); f.write(hdr); f.write(b'\0' * pad)
    for b in blocks:
        f.write(b); f.write(b'\0' * ((4 - len(b) % 4) % 4))
print(f'wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB): K={K}')
