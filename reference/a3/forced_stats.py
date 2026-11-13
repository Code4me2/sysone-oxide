"""Teacher-forced single layers: on the same input (sysone's stream after layer L-1), the local error of sysone's
layer L and of the bf16 reference's layer L against the fp32 reference's layer L, per op group:
  mix   mixer output (GDN / attention incl. out_proj), relative to the fp32 mixer output
  mlp   MLP output, relative to the fp32 MLP output (note: its input differs per impl: their own mid)
  dlay  layer delta (layer_L - input), relative to the fp32 delta
Also the fraction of elements bit-identical between sysone and the bf16 reference."""
import sys, numpy as np
from safetensors.numpy import load_file
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
sysdir, rid = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else ''
s = load_file(sysdir + '/hidden_%s.safetensors' % rid)
b = load_file(A + 'forced/bf16%s_%s.safetensors' % (tag, rid))
f = load_file(A + 'forced/fp32_%s.safetensors' % rid)
rows = f['prefix_rows']
assert (b['prefix_rows'] == rows).all()


def rel(a, t):
    a = a.astype(np.float64); t = t.astype(np.float64)
    return np.sqrt(((a - t) ** 2).sum(1) / np.maximum((t ** 2).sum(1), 1e-300))


def S(name, L):
    return s['full/%s%d' % (name, L)][rows] if L >= 0 else None


tot = {}
print('L   ty   | mix: sys ref ratio same | mid: ratio | mlp: sys ref ratio | layer-delta: sys ref ratio  medlog  up/down')
for L in range(32):
    ty = 'attn' if L % 4 == 3 else 'gdn '
    xin = S('layer', L - 1) if L > 0 else None
    out = []
    for k in ['mix', 'mid', 'mlp', 'layer']:
        sv = S(k, L); bv = b['%s%d' % (k, L)]; fv = f['%s%d' % (k, L)]
        if k in ('layer', 'mid') and xin is not None:
            sv, bv, fv = sv - xin, bv - xin, fv - xin
        es, eb = rel(sv, fv), rel(bv, fv)
        same = np.mean(S(k, L) == b['%s%d' % (k, L)])
        out.append((es, eb, same))
        tot.setdefault(k, []).append((np.log(es / eb)))
    (ms, mb, msame), (ds, db, _), (ls, lb, _), (ys, yb, _) = out
    g = lambda x: np.exp(np.mean(np.log(np.maximum(x, 1e-30))))
    lr = np.log(ys / yb)
    print('%2d %s | %.2e %.2e %.3f %.2f | %.3f | %.2e %.2e %.3f | %.2e %.2e %.3f %+.3f %3d/%-3d' % (
        L, ty, g(ms), g(mb), g(ms) / g(mb), msame, g(ds) / g(db), g(ls), g(lb), g(ls) / g(lb), g(ys), g(yb),
        g(ys) / g(yb), np.median(lr), (ys > 1.5 * yb).sum(), (yb > 1.5 * ys).sum()))
for k, v in tot.items():
    v = np.array(v)
    gd = v[[L for L in range(32) if L % 4 != 3]]; at = v[[L for L in range(32) if L % 4 == 3]]
    print('%-5s mean log ratio: gdn layers %+.4f, attn layers %+.4f' % (k, gd.mean(), at.mean()))
