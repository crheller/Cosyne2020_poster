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

ap_df = ld.load_dprime('refTar_pca4_dprime_act_bigPass_binPreLick')
pbps_df = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_binPreLick')

f, ax = plt.subplots(2, 2, figsize=(8, 6))

axis = 'ssa'
metric = 'dprime'
sd_cut = 3.5
normalize = True

ap_df = ap_df[ap_df['axis']==axis]
pbps_df = pbps_df[pbps_df['axis']==axis]

# compare active vs. big pupil
iv_mask = ((ap_df[metric] > sd_cut * np.nanstd(ap_df[metric].values)) | np.isnan(ap_df[metric])).astype(bool)
ap_df = ap_df[~iv_mask]

rr_idx = [i for i in ap_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rr_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()

vmin = pd.concat([rt_pb_bysite[metric], rt_active_bysite[metric]]).min()
vmax = pd.concat([rt_pb_bysite[metric], rt_active_bysite[metric]]).max()
ax[0, 0].scatter(rt_pb_bysite[metric], rt_active_bysite[metric], color='k', edgecolor='white')
ax[0, 0].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[0, 0].set_xlabel('Big Passive')
ax[0, 0].set_ylabel('Active')
ax[0, 0].set_aspect(cplt.get_square_asp(ax[0, 0]))

vmin = pd.concat([rr_pb_bysite[metric], rr_active_bysite[metric]]).min()
vmax = pd.concat([rr_pb_bysite[metric], rr_active_bysite[metric]]).max()
ax[1, 0].scatter(rr_pb_bysite[metric], rr_active_bysite[metric], color='k', edgecolor='white')
ax[1, 0].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[1, 0].set_xlabel('Big Passive')
ax[1, 0].set_ylabel('Active')
ax[1, 0].set_aspect(cplt.get_square_asp(ax[1, 0]))


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

sites = [s for s in rt_pb2_bysite.index if s in rt_ps_bysite.index]

vmin = pd.concat([rt_pb2_bysite[metric].loc[sites], rt_ps_bysite[metric].loc[sites]]).min()
vmax = pd.concat([rt_pb2_bysite[metric].loc[sites], rt_ps_bysite[metric].loc[sites]]).max()
ax[0, 1].scatter(rt_ps_bysite[metric].loc[sites], rt_pb2_bysite[metric].loc[sites], color='k', edgecolor='white')
ax[0, 1].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[0, 1].set_xlabel('Small Passive')
ax[0, 1].set_ylabel('Big Passive')
ax[0, 1].set_aspect(cplt.get_square_asp(ax[0, 1]))


vmin = pd.concat([rr_pb2_bysite[metric].loc[sites], rr_ps_bysite[metric].loc[sites]]).min()
vmax = pd.concat([rr_pb2_bysite[metric].loc[sites], rr_ps_bysite[metric].loc[sites]]).max()
ax[1, 1].scatter(rr_ps_bysite[metric].loc[sites], rr_pb2_bysite[metric].loc[sites], color='k', edgecolor='white')
ax[1, 1].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[1, 1].set_xlabel('Small Passive')
ax[1, 1].set_ylabel('Big Passive')
ax[1, 1].set_aspect(cplt.get_square_asp(ax[1, 1]))

f.tight_layout()

# bar plots to show the same thing? Caveat / challenge of this is that there's different epoch
# data in these bars because small pupil has less data.

f, ax = plt.subplots(1, 2)

# REF/REF plot (w/in category)
norm = rr_active_bysite[metric].copy()
normS = rr_active_bysite.loc[rr_ps_bysite.index][metric].copy()
if normalize:
    pass
else:
    norm.loc[norm.index] = np.ones(rr_active_bysite.shape[0])
    normS.loc[normS.index]= np.ones(rr_ps_bysite.shape[0])

ax[0].bar([0, 1, 2], [(rr_active_bysite[metric] / norm).mean(),
                      (rr_pb_bysite[metric] / norm).mean(),
                      (rr_ps_bysite[metric] / normS).mean()], 
                      edgecolor='k', color='darkslateblue')

for s in rr_active_bysite.index: 
    try:
        n = norm.loc[s]
        ax[0].plot([0, 1, 2], [rr_active_bysite[metric].loc[s] / n,
                        rr_pb_bysite[metric].loc[s] / n,
                        rr_ps_bysite[metric].loc[s] / n], 
                        'o-', color='k')
    except:
        n = norm.loc[s]
        ax[0].plot([0, 1], [rr_active_bysite[metric].loc[s] / n,
                        rr_pb_bysite[metric].loc[s] / n], 
                        'o-', color='k')

ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# TAR/REF plot (across category)
norm = rt_active_bysite[metric].copy()
normS = rt_active_bysite.loc[rt_ps_bysite.index][metric].copy()
if normalize:
    pass
else:
    norm.loc[norm.index] = np.ones(rt_active_bysite.shape[0])
    normS.loc[normS.index]= np.ones(rt_ps_bysite.shape[0])

ax[1].bar([0, 1, 2], [(rt_active_bysite[metric] / norm).mean(),
                      (rt_pb_bysite[metric] / norm).mean(),
                      (rt_ps_bysite[metric] / normS).mean()], 
                      edgecolor='k', color='violet')

for s in rt_active_bysite.index:
    n = norm.loc[s]
    try:
        ax[1].plot([0, 1, 2], [rt_active_bysite[metric].loc[s] / n,
                        rt_pb_bysite[metric].loc[s] / n,
                        rt_ps_bysite[metric].loc[s] / n], 
                        'o-', color='k')
    except:
        ax[1].plot([0, 1], [rt_active_bysite[metric].loc[s] / n,
                        rt_pb_bysite[metric].loc[s] / n], 
                        'o-', color='k')

ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()