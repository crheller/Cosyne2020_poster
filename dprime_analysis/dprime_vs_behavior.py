
import loading as ld
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import charlieTools.plotting as cplt
import pandas as pd

# ===================== load behavior for each site ===================================
behavior = pd.read_pickle('/auto/users/hellerc/code/projects/Cosyne2020_poster/behavior/results.pickle')
DI_collapse = [np.mean(di) for di in behavior['all_DI']]
dprime_collapse = [np.mean(di) for di in behavior['all_dprime']]
behavior['DI_collapse'] = DI_collapse
behavior['dprime_collapse'] = dprime_collapse

# ====================== load dprime for each site ====================================
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
# ===============================================================================================================

# now, compare the change in target reference decoding to behavior performance
behave_metric = 'DI_collapse'
f, ax = plt.subplots(1, 2)

# for active vs. passive
rt_active_bysite['performance'] = behavior[behave_metric]
frac_improvement = (rt_active_bysite['dprime'] - rt_passive_bysite['dprime']) / rt_active_bysite['dprime']
frac_improvement2 = (rt_active_bysite['dprime'] - rt_bigPassive_bysite['dprime']) / rt_active_bysite['dprime']
ymin = np.concatenate((frac_improvement, frac_improvement2)).min() - 0.1
ymax = np.concatenate((frac_improvement, frac_improvement2)).max() + 0.1

ax[0].scatter(rt_active_bysite['performance'], 
                frac_improvement, 
                c='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.65, linestyle='--', color='k')
ax[0].set_ylim((ymin, ymax))
ax[0].set_xlabel(behave_metric)
ax[0].set_ylabel('Normalized dprime change')
ax[0].set_title('active vs. passive')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# for active vs. big pupil
ax[1].scatter(rt_active_bysite['performance'], 
                frac_improvement2, 
                c='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.65, linestyle='--', color='k')
ax[1].set_ylim((ymin, ymax))
ax[1].set_xlabel(behave_metric)
ax[1].set_ylabel('Normalized dprime change')
ax[1].set_title('active vs. big pupil')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

f.canvas.set_window_title('Target vs. Reference')

# compare the reduction in reference discrimination to behavior performance
f, ax = plt.subplots(1, 2)

rr_active_bysite['performance'] = behavior[behave_metric]
frac_improvement = (rr_active_bysite['dprime'] - rr_passive_bysite['dprime']) / rr_active_bysite['dprime']
frac_improvement2 = (rr_active_bysite['dprime'] - rr_bigPassive_bysite['dprime']) / rr_active_bysite['dprime']
ymin = np.concatenate((frac_improvement, frac_improvement2)).min() - 0.1
ymax = np.concatenate((frac_improvement, frac_improvement2)).max() + 0.1

ax[0].scatter(rt_active_bysite['performance'], 
                frac_improvement, 
                c='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.65, linestyle='--', color='k')
ax[0].set_ylim((ymin, ymax))
ax[0].set_xlabel(behave_metric)
ax[0].set_ylabel('Normalized dprime change')
ax[0].set_title('active vs. passive')
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# for active vs. big pupil
ax[1].scatter(rt_active_bysite['performance'], 
                frac_improvement2, 
                c='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.65, linestyle='--', color='k')
ax[1].set_ylim((ymin, ymax))
ax[1].set_xlabel(behave_metric)
ax[1].set_ylabel('Normalized dprime change')
ax[1].set_title('active vs. big pupil')
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

f.canvas.set_window_title('Reference vs. Reference')

plt.show()