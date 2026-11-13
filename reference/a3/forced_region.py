"""Teacher-forced local errors restricted to a prefix row range, plus coherence of (sys - ref) and of errors."""
import sys, numpy as np
from safetensors.numpy import load_file
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
sysdir, rid, a, b_ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = load_file(sysdir + '/hidden_%s.safetensors' % rid)
b = load_file(A + 'forced/bf16_%s.safetensors' % rid)
f = load_file(A + 'forced/fp32_%s.safetensors' % rid)
rows = f['prefix_rows']
m = (rows >= a) & (rows < b_)
g = lambda x: np.exp(np.mean(np.log(np.maximum(x, 1e-30))))


def rel(x, t):
    return np.sqrt(((x - t) ** 2).sum(1) / np.maximum((t ** 2).sum(1), 1e-300))


def coh(e):
    return np.linalg.norm(e.mean(0)) * np.sqrt(e.shape[0]) / np.sqrt((e ** 2).sum(1).mean())


print('rows', a, b_, 'n', m.sum())
print('L  ty   | mix ratio  coh sys ref s-r | layer-delta ratio coh sys ref s-r')
acc = []
for L in range(32):
    ty = 'attn' if L % 4 == 3 else 'gdn '
    xin = s['full/layer%d' % (L - 1)][rows][m].astype(np.float64) if L else 0
    o = []
    for k in ['mix', 'layer']:
        sv = s['full/%s%d' % (k, L)][rows][m].astype(np.float64); bv = b['%s%d' % (k, L)][m].astype(np.float64)
        fv = f['%s%d' % (k, L)][m].astype(np.float64)
        if k == 'layer':
            sv, bv, fv = sv - xin, bv - xin, fv - xin
        r = g(rel(sv, fv)) / g(rel(bv, fv))
        o.append('%.3f %6.2f %6.2f %6.2f' % (r, coh(sv - fv), coh(bv - fv), coh(sv - bv)))
        if k == 'layer': acc.append(np.log(r))
    print('%2d %s | %s | %s' % (L, ty, o[0], o[1]))
print('mean log ratio layer-delta: %+.4f' % np.mean(acc))
