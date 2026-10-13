"""L1-adjusted robustness check for H1 (additive phone + L1 model, within-phone permutation test).
Run from the project folder after `analyze`:  python src/robustness_check.py"""
import pandas as pd, numpy as np, json, sys
from scipy.stats import spearmanr
inv=pd.read_csv('results/inventory_phone_by_L1.csv',index_col=0)
summ=pd.read_csv('results/inventory_summary.csv').set_index('L1')
res={}
for tag in ['','_perceived']:
    pe=pd.read_csv(f'results/phone_error_by_L1{tag}.csv'); pe=pe[pe.n>=30].copy()
    pe['missing']=[inv.loc[p,l] in ('absent','allophone','parts') for p,l in zip(pe.phone,pe.L1)]
    pe['cons']=~pe.phone.str[0].isin(list('aeiouæɑɒɔəɚɛɜɝɪʊʌ'))
    # two-way adjustment: remove phone mean and L1 mean (additive model)
    def adj(d):
        d=d.copy(); d['r']=d.error_rate-d.groupby('phone').error_rate.transform('mean')-d.groupby('L1').error_rate.transform('mean')+d.error_rate.mean(); return d
    def contrast(d,col):
        diffs=[g[g.missing][col].mean()-g[~g.missing][col].mean() for _,g in d.groupby('phone') if 0<g.missing.sum()<len(g)]
        return np.mean(diffs),len(diffs)
    rng=np.random.default_rng(0)
    out={}
    for name,d in [('all',pe),('consonants',pe[pe.cons]),('vowels',pe[~pe.cons])]:
        a=adj(d); obs,k=contrast(a,'r'); raw,_=contrast(d,'error_rate')
        null=[]
        for _ in range(2000):
            s=d.copy(); s['missing']=s.groupby('phone').missing.transform(lambda x: rng.permutation(x.values)); null.append(contrast(adj(s),'r')[0])
        p=(np.sum(np.abs(null)>=abs(obs))+1)/(len(null)+1)
        out[name]={'raw_within_phone_diff':round(raw,4),'L1_adjusted_diff':round(obs,4),'phones':k,'perm_p':round(p,4)}
    # largest per-phone excesses
    rows=[]
    for ph,g in pe.groupby('phone'):
        if 0<g.missing.sum()<len(g):
            rows.append((ph, g[g.missing].error_rate.mean()-g[~g.missing].error_rate.mean(), ','.join(g[g.missing].L1)))
    out['top_phones']=sorted([(a,round(b,3),c) for a,b,c in rows],key=lambda x:-x[1])[:10]
    out['bottom_phones']=sorted([(a,round(b,3),c) for a,b,c in rows],key=lambda x:x[1])[:5]
    res[tag or 'canonical']=out
# L1-level association
l1=pd.read_csv('results/pfer_by_L1.csv').set_index('L1')
x=summ.loc[l1.index,'n_consonants_not_phonemic']; j=summ.loc[l1.index,'jaccard_loose_corrected']
res['L1_level']={'spearman_PFER_vs_n_missing_consonants':spearmanr(x,l1.PFER_mean),'spearman_PFER_vs_jaccard':spearmanr(j,l1.PFER_mean)}
print(json.dumps(res,indent=1,ensure_ascii=False,default=lambda o: [float(v) for v in o]))
# specific predictions
pe=pd.read_csv('results/phone_error_by_L1.csv')
for l1,phs in [('Arabic',['p','v','b','f']),('Korean',['f','p','z','v']),('Hindi',['v','w']),('Spanish',['ʃ','dʒ','z','v']),('Mandarin',['θ','ð','v','z']),('Vietnamese',['θ','ð','ʃ'])]:
    s=pe[(pe.L1==l1)&pe.phone.isin(phs)][['phone','n','error_rate','most_common_outputs']]
    s['others_mean']=[pe[(pe.phone==p)&(pe.L1!=l1)].error_rate.mean() for p in s.phone]
    print('##',l1); print(s.round(3).to_string(index=False))
