
import loading as ld
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import charlieTools.plotting as cplt
import pandas as pd
import scipy.stats as ss

# ===================== load behavior for each site ===================================
behavior = pd.read_pickle('/auto/users/hellerc/code/projects/Cosyne2020_poster/behavior/results.pickle')
DI_collapse = [np.mean(di) for di in behavior['all_DI']]
dprime_collapse = [np.mean(di) for di in behavior['all_dprime']]
behavior['DI_collapse'] = DI_collapse
behavior['dprime_collapse'] = dprime_collapse

# ====================== load dprime for each site ====================================
ap_df = ld.load_dprime('refTar_pca4_dprime_act_pass_binPreLick')
apb_df = ld.load_dprime('refTar_pca4_dprime_act_bigPass_binPreLick')
pbps_df = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_binPreLick')

sites = ap_df['site'].unique()

axis = 'ssa'
metric = 'dprime'
behave_metric = 'DI_collapse'
sd_cut = 3.5
normalize = False

ap_df = ap_df[ap_df['axis']==axis]
apb_df = apb_df[apb_df['axis']==axis]
pbps_df = pbps_df[pbps_df['axis']==axis]

# compare active vs. passive
iv_mask = ((ap_df[metric] > sd_cut * np.nanstd(ap_df[metric].values)) | np.isnan(ap_df[metric])).astype(bool)
ap_df = ap_df[~iv_mask]

rr_idx = [i for i in ap_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_passive_bysite = ap_df[(ap_df['state']=='passive') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rr_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_passive_bysite = ap_df[(ap_df['state']=='passive') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()

s = [s for s in sites if (s in rt_active_bysite.index) & (s in rt_passive_bysite.index)]
rt_active_bysite = rt_active_bysite.loc[s]
rt_passive_bysite = rt_passive_bysite.loc[s]
s = [s for s in sites if (s in rr_active_bysite.index) & (s in rr_passive_bysite.index)]
rr_active_bysite = rr_active_bysite.loc[s]
rr_passive_bysite = rr_passive_bysite.loc[s]


# compare active vs. big pupil
iv_mask = ((apb_df[metric] > sd_cut * np.nanstd(apb_df[metric].values)) | np.isnan(apb_df[metric])).astype(bool)
apb_df = apb_df[~iv_mask]

rr_idx = [i for i in apb_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in apb_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in apb_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active2_bysite = apb_df[(apb_df['state']=='active') & (apb_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb_bysite = apb_df[(apb_df['state']=='big_passive') & (apb_df.index.isin(rt_idx))].groupby(by='site').mean()
rr_active2_bysite = apb_df[(apb_df['state']=='active') & (apb_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb_bysite = apb_df[(apb_df['state']=='big_passive') & (apb_df.index.isin(rr_idx))].groupby(by='site').mean()

s = [s for s in sites if (s in rt_active2_bysite.index) & (s in rt_pb_bysite.index)]
rt_active2_bysite = rt_active2_bysite.loc[s]
rt_pb_bysite = rt_pb_bysite.loc[s]
s = [s for s in sites if (s in rr_active2_bysite.index) & (s in rr_pb_bysite.index)]
rr_active2_bysite = rr_active2_bysite.loc[s]
rr_pb_bysite = rr_pb_bysite.loc[s]

# compare big pupil vs. small pupil
iv_mask = ((pbps_df[metric] > sd_cut * np.nanstd(pbps_df[metric].values)) | np.isnan(pbps_df[metric])).astype(bool)
pbps_df = pbps_df[~iv_mask]

rr_idx = [i for i in pbps_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in pbps_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in pbps_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rt_idx))].groupby(by='site').mean()
rr_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rr_idx))].groupby(by='site').mean()

s = [s for s in sites if (s in rt_pb2_bysite.index) & (s in rt_ps_bysite.index)]
rt_pb2_bysite = rt_pb2_bysite.loc[s]
rt_ps_bysite = rt_ps_bysite.loc[s]
s = [s for s in sites if (s in rr_pb2_bysite.index) & (s in rr_ps_bysite.index)]
rr_pb2_bysite = rr_pb2_bysite.loc[s]
rr_ps_bysite = rr_ps_bysite.loc[s]

# ===============================================================================================================

# now, compare the change in target reference decoding to behavior performance
f, ax = plt.subplots(1, 3, figsize=(8, 6))

# for active vs. passive
rt_active_bysite['performance'] = behavior[behave_metric]
if normalize:
    frac_improvement = (rt_active_bysite['dprime'] - rt_passive_bysite['dprime']) / (rt_active_bysite['dprime'] + rt_passive_bysite['dprime'])
    frac_improvement2 = (rt_active2_bysite['dprime'] - rt_pb_bysite['dprime']) / (rt_active2_bysite['dprime'] + rt_pb_bysite['dprime'])
    frac_improvement3 = (rt_pb2_bysite['dprime'] - rt_ps_bysite['dprime']) / (rt_pb2_bysite['dprime'] + rt_ps_bysite['dprime'])
else:
    frac_improvement = (rt_active_bysite['dprime'] - rt_passive_bysite['dprime'])
    frac_improvement2 = (rt_active2_bysite['dprime'] - rt_pb_bysite['dprime'])
    frac_improvement3 = (rt_pb2_bysite['dprime'] - rt_ps_bysite['dprime'])
ymin = np.concatenate((frac_improvement, frac_improvement2, frac_improvement3)).min() - 0.1
ymax = np.concatenate((frac_improvement, frac_improvement2, frac_improvement3)).max() + 0.1

c, pval = ss.pearsonr(behavior[behave_metric].loc[frac_improvement.index], frac_improvement)
pval = round(pval, 3)
ax[0].scatter(behavior[behave_metric].loc[frac_improvement.index], 
                frac_improvement, 
                c='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.6, linestyle='--', color='k')
ax[0].set_ylim((ymin, ymax))
ax[0].set_xlabel(behave_metric)
ax[0].set_ylabel('Normalized dprime change')
ax[0].set_title('active vs. passive, pval: {}'.format(pval), fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# for active vs. big pupil
c, pval = ss.pearsonr(behavior[behave_metric].loc[frac_improvement2.index], frac_improvement2)
pval = round(pval, 3)
ax[1].scatter(behavior[behave_metric].loc[frac_improvement2.index], 
                frac_improvement2, 
                c='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.6, linestyle='--', color='k')
ax[1].set_ylim((ymin, ymax))
ax[1].set_xlabel(behave_metric)
ax[1].set_ylabel('Normalized dprime change')
ax[1].set_title('active vs. big pupil pval: {}'.format(pval), fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

# for big pupil vs small pupil
c, pval = ss.pearsonr(behavior[behave_metric].loc[frac_improvement3.index], frac_improvement3)
pval = round(pval, 3)
ax[2].scatter(behavior[behave_metric].loc[frac_improvement3.index], 
                frac_improvement3, 
                c='k', edgecolor='white')
ax[2].axhline(0, linestyle='--', color='k')
ax[2].axvline(0.6, linestyle='--', color='k')
ax[2].set_ylim((ymin, ymax))
ax[2].set_xlabel(behave_metric)
ax[2].set_ylabel('Normalized dprime change')
ax[2].set_title('big pupil vs. small pupil, pval: {}'.format(pval), fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))


f.tight_layout()

f.canvas.set_window_title('Target vs. Reference')


# compare the reduction in reference discrimination to behavior performance
f, ax = plt.subplots(1, 3, figsize=(8, 6))


rr_active_bysite['performance'] = behavior[behave_metric]
if normalize:
    frac_improvement = (rr_active_bysite['dprime'] - rr_passive_bysite['dprime']) / (rr_active_bysite['dprime'] + rr_passive_bysite['dprime'])
    frac_improvement2 = (rr_active2_bysite['dprime'] - rr_pb_bysite['dprime']) / (rr_active2_bysite['dprime'] + rr_pb_bysite['dprime'])
    frac_improvement3 = (rr_pb2_bysite['dprime'] - rr_ps_bysite['dprime']) / (rr_pb2_bysite['dprime'] + rr_ps_bysite['dprime'])
else:
    frac_improvement = (rr_active_bysite['dprime'] - rr_passive_bysite['dprime'])
    frac_improvement2 = (rr_active2_bysite['dprime'] - rr_pb_bysite['dprime'])
    frac_improvement3 = (rr_pb2_bysite['dprime'] - rr_ps_bysite['dprime'])
ymin = np.concatenate((frac_improvement, frac_improvement2, frac_improvement2)).min() - 0.1
ymax = np.concatenate((frac_improvement, frac_improvement2, frac_improvement2)).max() + 0.1

c, pval = ss.pearsonr(rr_active_bysite['performance'], frac_improvement)
pval = round(pval, 3)
ax[0].scatter(rr_active_bysite['performance'], 
                frac_improvement, 
                c='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.65, linestyle='--', color='k')
ax[0].set_ylim((ymin, ymax))
ax[0].set_xlabel(behave_metric)
ax[0].set_ylabel('Normalized dprime change')
ax[0].set_title('active vs. passive, pval: {}'.format(pval), fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# for active vs. big pupil
c, pval = ss.pearsonr(rr_active_bysite['performance'], frac_improvement2)
pval = round(pval, 3)
ax[1].scatter(rr_active_bysite['performance'], 
                frac_improvement2, 
                c='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.65, linestyle='--', color='k')
ax[1].set_ylim((ymin, ymax))
ax[1].set_xlabel(behave_metric)
ax[1].set_ylabel('Normalized dprime change')
ax[1].set_title('active vs. big pupil, pval: {}'.format(pval), fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

# for active vs. big pupil
c, pval = ss.pearsonr(rr_active_bysite['performance'].loc[frac_improvement3.index], frac_improvement3)
pval = round(pval, 3)
ax[2].scatter(rt_active_bysite['performance'].loc[frac_improvement3.index], 
                frac_improvement3, 
                c='k', edgecolor='white')
ax[2].axhline(0, linestyle='--', color='k')
ax[2].axvline(0.65, linestyle='--', color='k')
ax[2].set_ylim((ymin, ymax))
ax[2].set_xlabel(behave_metric)
ax[2].set_ylabel('Normalized dprime change')
ax[2].set_title('big pupil vs. small pupil, pval: {}'.format(pval), fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

f.canvas.set_window_title('Reference vs. Reference')

plt.show()