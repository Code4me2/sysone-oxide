"""sysone realizations (served + hidden-basis permutations) vs bf16 reference realizations, errors vs fp32."""
import sys, itertools, numpy as np
from safetensors.numpy import load_file
H = '/home/velvet/Desktop/sysone-ref/golden/dev/hidden_dense/'
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
ids = ['m000_noul_default', 'm005_choice_5', 'm008_choice_27', 'm010_choice_255', 'm013_score_3', 'm019_state_json',
       'm026_unicode_state', 'm037_len1024_debias1', 'm040_len4096_debias1', 'm044_len16000_debias1']
sysruns = sys.argv[1].split(',')  # dirs under a3_deep
refs = ['bf16_served', 'bf16_perq', 'bf16_perm1', 'bf16_perm2', 'bf16_perm3']
taps = ['layer0', 'layer3', 'layer4', 'layer15', 'layer31', 'norm']


def rel(a, t):
    a = a.astype(np.float64); t = t.astype(np.float64)
    return np.sqrt(((a - t) ** 2).sum(1) / (t ** 2).sum(1))


rows = lambda d, t: np.concatenate([d[t + '/prefix'], d[t + '/answer']])
E = {}
rid = []
for i in ids:
    f = load_file(H + 'fp32_served/hidden_%s.safetensors' % i)
    n = None
    for r in refs:
        d = load_file(H + r + '/hidden_%s.safetensors' % i)
        for t in taps:
            E.setdefault((r, t), []).append(rel(rows(d, t), rows(f, t)))
    for r in sysruns:
        d = load_file(A + r + '/hidden_%s.safetensors' % i)
        for t in taps:
            E.setdefault((r, t), []).append(rel(rows(d, t), rows(f, t)))
    rid += [i] * len(rows(f, 'norm'))
rid = np.array(rid)
E = {k: np.concatenate(v) for k, v in E.items()}
allr = refs + sysruns
for t in taps:
    print('==', t)
    print('  geo-mean error x1e-3 (all rows):', ' '.join('%s %.4f' % (r.replace('sysone_', 's_').replace('bf16_', 'b_'), 1e3 * np.exp(np.mean(np.log(E[(r, t)])))) for r in allr))
    print('  median ln(e_x/e_served):', ' '.join('%s %+.4f' % (r.replace('sysone_', 's_').replace('bf16_', 'b_'), np.median(np.log(E[(r, t)] / E[('bf16_served', t)]))) for r in allr if r != 'bf16_served'))

    def pair(a, b):
        ea, eb = E[(a, t)], E[(b, t)]
        return int((ea > 1.5 * eb).sum()), int((eb > 1.5 * ea).sum()), np.median(np.log(ea / eb))
    for name, prs in [('bf16 vs bf16', list(itertools.permutations(refs, 2))),
                      ('sys vs bf16', [(a, b) for a in sysruns for b in refs]),
                      ('sys vs sys', list(itertools.permutations(sysruns, 2)))]:
        v = [pair(a, b) for a, b in prs]
        print('  %-12s pairs %2d: up-down mean %+.1f range [%d, %d]; median log ratio mean %+.4f range [%+.4f, %+.4f]' % (
            name, len(v), np.mean([u - d for u, d, _ in v]), min(u - d for u, d, _ in v), max(u - d for u, d, _ in v),
            np.mean([m for *_, m in v]), min(m for *_, m in v), max(m for *_, m in v)))
    print('  per request geo-mean x1e-3: bf16 [served perq p1 p2 p3] | sysone', sysruns)
    for i in ids:
        m = rid == i
        g = lambda r: 1e3 * np.exp(np.mean(np.log(E[(r, t)][m])))
        print('    %-22s %s | %s' % (i, ' '.join('%.3f' % g(r) for r in refs), ' '.join('%.3f' % g(r) for r in sysruns)))
