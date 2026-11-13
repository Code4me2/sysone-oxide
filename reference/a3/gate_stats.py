"""Gate (a) statistics (hidden_dense.evaluate, committed design effects) for sysone dump dirs vs bf16_served."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import hidden_dense as hd
from safetensors.numpy import load_file
rep = json.load(open(hd.REPORT))
g = rep['gate']
A = '/home/velvet/Desktop/sysone-ref/golden/dev/a3_deep/'
truth, _, _ = hd.load_rows('fp32_served')
ref, _, _ = hd.load_rows('bf16_served')
for run in sys.argv[1].split(','):
    print('==', run)
    per = {t: [] for t in hd.TAPS}
    for item in hd.dump_requests():
        d = load_file(A + run + '/hidden_%s.safetensors' % item['id'])
        for t in hd.TAPS:
            per[t].append(np.concatenate([d[t + '/prefix'], d[t + '/answer']]))
    for t in hd.TAPS:
        ec = hd.rel_rows(np.concatenate(per[t]), truth[t]); eb = hd.rel_rows(ref[t], truth[t])
        s = hd.evaluate(ec, eb, g['factor'], g['design_effect'][t])
        print('  %-7s p99 %.3e (ref %.3e) med %+.4f up/down %d/%d sign p %.3f %s' % (
            t, s['cand_p99'], s['ref_p99'], s['median_log_ratio'], s['n_up'], s['n_down'], s['sign_p'],
            'PASS' if not s['failed'] else s['failed']))
