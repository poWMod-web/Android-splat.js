# Face-crop evaluation: PSNR and sharpness of rendered held-out face crops
# (render_views.js, white background) against the native-resolution crops.
#   python tests/bench/face_eval.py --recon scratch/lisa_face_recon_all.json --data data/lisaface \
#          --renders v12=scratch/lisa_v12_face v1=scratch/lisa_face_v1_face --out scratch/face_eval
import os, sys, json
import numpy as np, cv2
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
rc = json.load(open(arg('--recon'))); data = arg('--data'); out = arg('--out', 'scratch/face_eval')
renders = dict(kv.split('=', 1) for kv in sys.argv[sys.argv.index('--renders') + 1:] if '=' in kv)
held = [rc['cams'][i] for i in rc['heldOut']]
lap = lambda g: cv2.Laplacian(g, cv2.CV_32F, ksize=3)
res = {k: [] for k in renders}; strips = []
for c in held:
    gt = cv2.imread(os.path.join(data, c['name'])).astype(np.float32) / 255
    m = cv2.imread(os.path.join(data, 'masks', c['name'].replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    comp = gt * m[..., None] + (1 - m[..., None])
    core = m > 0.9
    # the face sits around the crop centre (face_crops.py centres it, shifted up a quarter face for hair)
    S = gt.shape[0]; f0, f1 = S // 2 - S // 4, S // 2 + S // 4
    row = [comp[f0:f1, f0:f1]]
    gl = lap(cv2.cvtColor(comp[f0:f1, f0:f1], cv2.COLOR_BGR2GRAY))
    for k, pre in renders.items():
        p = f'{pre}_{c["imgIdx"]:03d}.png'
        r = cv2.imread(p)
        if r is None: print('missing', p); continue
        r = r[..., :3].astype(np.float32) / 255
        err = ((r - comp) ** 2)[core].mean()
        psnr = 10 * np.log10(1 / max(err, 1e-10))
        rl = lap(cv2.cvtColor(r[f0:f1, f0:f1], cv2.COLOR_BGR2GRAY))
        fm = core[f0:f1, f0:f1]
        sharp = float(rl[fm].var() / max(gl[fm].var(), 1e-9))
        errf = ((r - comp) ** 2)[f0:f1, f0:f1][fm].mean()
        res[k].append({'name': c['name'], 'psnr': float(psnr), 'psnrFace': float(10 * np.log10(1 / max(errf, 1e-10))), 'sharp': sharp})
        row.append(r[f0:f1, f0:f1])
    strips.append(np.hstack(row))
summary = {k: {'psnr': float(np.mean([r['psnr'] for r in v])), 'psnrFace': float(np.mean([r['psnrFace'] for r in v])),
               'sharp': float(np.median([r['sharp'] for r in v])), 'n': len(v)} for k, v in res.items() if v}
for k, s in summary.items():
    print(f'{k:10s} subject PSNR {s["psnr"]:.2f}  face PSNR {s["psnrFace"]:.2f}  sharpness ratio (render/GT Laplacian var) {s["sharp"]:.3f}  n={s["n"]}')
json.dump({'summary': summary, 'rows': res}, open(out + '.json', 'w'), indent=1)
cv2.imwrite(out + '_cmp.jpg', (np.vstack(strips) * 255).clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('wrote', out + '_cmp.jpg', '(columns: GT,', ', '.join(renders), ')')
