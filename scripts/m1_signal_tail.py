import pandas as pd, numpy as np
df = pd.read_csv('/home/zz865/pythonProject/SimOPD/docs/data/training_metrics_16k_allkeys.csv.gz')
P = 'actor/distillation/'
ARMS = ['vanilla','f2_hard_clip','f1_soft_log','b1_skew_kl','f3_power']
LOCK = {'vanilla':122,'f2_hard_clip':198,'f1_soft_log':208,'b1_skew_kl':247,'f3_power':np.inf}
for lo, hi in [(25,60),(25,100)]:
    w = df[(df.arm.isin(ARMS)) & (df.step.between(lo,hi))]
    print(f'=== window {lo}-{hi} (median over steps x 3 seeds) ===')
    out=[]
    for arm,g in w.groupby('arm'):
        lm  = g[P+'loss_max'].median()
        p5  = g[P+'delta_ell_p5'].median()
        am  = g[P+'delta_ell_absmean'].median()
        tr  = g['response_length/clip_ratio'].median()
        # per-arm scale correction -> effective update-signal extreme u_max
        if arm=='vanilla':      umax=lm; raw=lm
        elif arm=='f2_hard_clip': umax=min(lm,10.0); raw=lm       # panel pre-clip
        elif arm=='f1_soft_log':  umax=lm; raw=np.expm1(abs(lm))  # loss IS softlog(r)
        elif arm=='b1_skew_kl':   umax=lm; raw=np.nan             # estimator's own bound ln10=2.303
        elif arm=='f3_power':     umax=lm; raw=np.nan             # loss bounded [-1,1]; delta_ell panel = raw k1 (F6)
        out.append(dict(arm=arm, lock=LOCK[arm], loss_max=lm, u_max=umax, raw_r_max=raw,
                        del_p5=p5, absmean=am, trunc=tr))
    t=pd.DataFrame(out).set_index('arm').reindex(ARMS)
    pd.set_option('display.width',200); pd.set_option('display.float_format',lambda x:f'{x:.3f}')
    print(t, '\n')
# per-seed u_max at 25-60 for spread
w = df[(df.arm.isin(ARMS)) & (df.step.between(25,60))]
print('per-seed loss_max median (25-60):')
print(w.groupby(['arm','seed'])[P+'loss_max'].median().unstack().reindex(ARMS))
