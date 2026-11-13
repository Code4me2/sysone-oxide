"""Mean log(sys_err / ref_err) by prefix position bin, per request, at chosen layers."""
import sys, numpy as np
from safetensors.numpy import load_file
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
sysdir = sys.argv[1]
ids = sys.argv[2].split(',')
taps = sys.argv[3].split(',')
bins = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [0, 16, 58, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
refrun = sys.argv[5] if len(sys.argv) > 5 else 'bf16_served'


def rel(a, t):
    a = a.astype(np.float64); t = t.astype(np.float64)
    return np.sqrt(((a - t) ** 2).sum(1) / (t ** 2).sum(1))


for i in ids:
    f = load_file(A + 'fp32_served/hidden_%s.safetensors' % i)
    r = load_file(A + refrun + '/hidden_%s.safetensors' % i)
    s = load_file(sysdir + '/hidden_%s.safetensors' % i)
    pr = f['prefix_rows']
    print('==', i, 'P rows', len(pr), 'bins', bins)
    for t in taps:
        es, er = rel(s[t + '/prefix'], f[t + '/prefix']), rel(r[t + '/prefix'], f[t + '/prefix'])
        lr = np.log(es / er)
        out = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (pr >= a) & (pr < b)
            if m.sum():
                out.append('%+.3f(%d)' % (lr[m].mean(), m.sum()))
        ea = rel(s[t + '/answer'], f[t + '/answer']); ra = rel(r[t + '/answer'], f[t + '/answer'])
        print('  %-7s' % t, ' '.join(out), ' ans', ' '.join('%+.2f' % x for x in np.log(ea / ra)))
