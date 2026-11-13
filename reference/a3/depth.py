"""Per-layer paired asymmetry: sysone vs bf16 reference, errors vs fp32, at every layer."""
import sys, numpy as np
from safetensors.numpy import load_file
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
ids = ['m000_noul_default', 'm005_choice_5', 'm008_choice_27', 'm010_choice_255', 'm013_score_3', 'm019_state_json',
       'm026_unicode_state', 'm037_len1024_debias1', 'm040_len4096_debias1', 'm044_len16000_debias1']
sysdir = sys.argv[1]
refrun = sys.argv[2] if len(sys.argv) > 2 else 'bf16_served'
sel = sys.argv[3].split(',') if len(sys.argv) > 3 else ids
taps = ['layer%d' % i for i in range(32)] + ['norm']


def rel(a, t):
    a = a.astype(np.float64); t = t.astype(np.float64)
    return np.sqrt(((a - t) ** 2).sum(1) / (t ** 2).sum(1))


ES = {t: [] for t in taps}; ER = {t: [] for t in taps}; RID = []
for i in sel:
    f = load_file(A + 'fp32_served/hidden_%s.safetensors' % i)
    r = load_file(A + refrun + '/hidden_%s.safetensors' % i)
    s = load_file(sysdir + '/hidden_%s.safetensors' % i)
    n = None
    for t in taps:
        tr = np.concatenate([f[t + '/prefix'], f[t + '/answer']])
        ES[t].append(rel(np.concatenate([s[t + '/prefix'], s[t + '/answer']]), tr))
        ER[t].append(rel(np.concatenate([r[t + '/prefix'], r[t + '/answer']]), tr))
        n = len(tr)
    RID += [i] * n
RID = np.array(RID)
print('layer  type  sys_geo ref_geo  medlog   up/down   | per request geo ratio sys/ref')
for t in taps:
    es, er = np.concatenate(ES[t]), np.concatenate(ER[t])
    L = int(t[5:]) if t != 'norm' else 32
    ty = 'norm' if t == 'norm' else ('attn' if L % 4 == 3 else 'gdn')
    lr = np.log(es / er)
    per = ' '.join('%+.3f' % np.mean(lr[RID == i]) for i in sel)
    print('%-7s %-4s %.3e %.3e %+.4f %4d/%-4d | %s' % (t, ty, np.exp(np.mean(np.log(es))), np.exp(np.mean(np.log(er))),
                                              np.median(lr), (es > 1.5 * er).sum(), (er > 1.5 * es).sum(), per))
print('requests:', ' '.join(sel))
