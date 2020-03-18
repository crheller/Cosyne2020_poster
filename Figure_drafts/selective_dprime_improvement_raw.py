"""
Show selective ref/Tar enhancement by active engagement 
and ref/ref enhancement by arousal.

Do ref/Tar with grouping ref, but could also do refs independently...
"""

import loading as ld
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import charlieTools.plotting as cplt

ap_df = ld.load_dprime('refTar_pca4_dprime_act_pass_binPreLick')
apb_df = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_binPreLick')
aps_df = ld.load_dprime('refTar_pca4_dprime_act_smallPass_binPreLick')

# remove outliers
all_dprime = pd.concat([ap_df['dprime'], apb_df['dprime'], aps_df['dprime']], axis=1)
iv_mask = (all_dprime > 3 * np.nanstd(all_dprime.values)).sum(axis=1) | np.isnan(all_dprime).sum(axis=1).astype(bool)
print("removing {} percent of results".format(round(100 * iv_mask.sum() / all_dprime.shape[0], 3)))

a_axis = 'ssa'
p_axis = 'ssa'
pb_axis = 'ssa'
ps_axis = 'ssa'
metric = 'dprime'

ap_df = ap_df[~iv_mask]
apb_df = apb_df[~iv_mask]
aps_df = aps_df[~iv_mask]

sites = ap_df['site'].unique()
bad_pup_sites = ['BRT033b', 'BRT034f', 'BRT036b', 'BRT037b', 'BRT039c', 'bbl102d']

rr_idx = [i for i in ap_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rr_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df['axis']==a_axis) & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_passive_bysite = ap_df[(ap_df['state']=='passive') & (ap_df['axis']==p_axis) & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_bigPassive_bysite = apb_df[(apb_df['state']=='big_passive') & (apb_df['axis']==pb_axis) & (apb_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_smallPassive_bysite = apb_df[(aps_df['state']=='small_passive') & (apb_df['axis']==ps_axis) & (apb_df.index.isin(rr_idx))].groupby(by='site').mean()

rt_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df['axis']==a_axis) & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_passive_bysite = ap_df[(ap_df['state']=='passive') & (ap_df['axis']==p_axis) & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_bigPassive_bysite = apb_df[(apb_df['state']=='big_passive') & (apb_df['axis']==pb_axis) & (apb_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_smallPassive_bysite = aps_df[(apb_df['state']=='small_passive') & (apb_df['axis']==ps_axis) & (apb_df.index.isin(rt_idx))].groupby(by='site').mean()

f, ax = plt.subplots(1, 2)

# REF/REF plot (w/in category)
norm = rr_active_bysite[metric]
ax[0].bar([0, 1, 2], [(rr_active_bysite[metric] / norm).mean(),
                      (rr_bigPassive_bysite[metric] / norm).mean(),
                      (rr_smallPassive_bysite[metric] / norm).mean()], 
                      edgecolor='k', color='darkslateblue')

for s in bad_pup_sites:
    n = norm.loc[s]
    ax[0].plot([0, 1, 2], [rr_active_bysite[metric].loc[s] / n,
                    rr_bigPassive_bysite[metric].loc[s] / n,
                    rr_smallPassive_bysite[metric].loc[s] / n], 
                    'o-', color='grey')

for s in set(sites) - set(bad_pup_sites):
    n = norm.loc[s]
    ax[0].plot([0, 1, 2], [rr_active_bysite[metric].loc[s] / n,
                      rr_bigPassive_bysite[metric].loc[s] / n,
                      rr_smallPassive_bysite[metric].loc[s] / n], 
                      'o-', color='k')

ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# TAR/REF plot (across category)
norm = rt_active_bysite[metric]
ax[1].bar([0, 1, 2], [(rt_active_bysite[metric] / norm).mean(),
                      (rt_bigPassive_bysite[metric] / norm).mean(),
                      (rt_smallPassive_bysite[metric] / norm).mean()], 
                      edgecolor='k', color='violet')

for s in bad_pup_sites:
    n = norm.loc[s]
    ax[1].plot([0, 1, 2], [rt_active_bysite[metric].loc[s] / n,
                      rt_bigPassive_bysite[metric].loc[s] / n,
                      rt_smallPassive_bysite[metric].loc[s] / n], 
                      'o-', color='grey')

for s in set(sites) - set(bad_pup_sites):
    n = norm.loc[s]
    ax[1].plot([0, 1, 2], [rt_active_bysite[metric].loc[s] / n,
                      rt_bigPassive_bysite[metric].loc[s] / n,
                      rt_smallPassive_bysite[metric].loc[s] / n], 
                      'o-', color='k')

ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()