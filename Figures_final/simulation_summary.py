"""
Sumamry ref/tar and ref/ref delta dprime for 
    raw data
    1st order simulation
    2nd order simulation
    Full simulation
Focus only on the big passive vs. active condtion
"""

import loading as ld
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import charlieTools.plotting as cplt
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/dprime_simulation_summary.svg'

ap_df = ld.load_dprime('refTar_pca4_dprime_act_bigPass_LDA_binPreLick')
pbps_df = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_LDA_binPreLick')
ap_df_sim1 = ld.load_dprime('refTar_pca4_dprime_act_bigPass_uMatch_sim1_LDA_binPreLick')
pbps_df_sim1 = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_uMatch_sim1_LDA_binPreLick')
ap_df_sim2 = ld.load_dprime('refTar_pca4_dprime_act_bigPass_uMatch_sim2_LDA_binPreLick')
pbps_df_sim2 = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_uMatch_sim2_LDA_binPreLick')
ap_df_sim12 = ld.load_dprime('refTar_pca4_dprime_act_bigPass_uMatch_sim12_LDA_binPreLick')
pbps_df_sim12 = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_uMatch_sim12_LDA_binPreLick')

axis = 'ssa'
metric = 'dprime'
sd_cut = 3.5
ylim = 10
xlim = 10
vmin = 0
vmax = ylim

ap_df = ap_df[ap_df['axis']==axis]
pbps_df = pbps_df[pbps_df['axis']==axis]

# Raw results
iv_mask = ((ap_df[metric] > sd_cut * np.nanstd(ap_df[metric].values)) | np.isnan(ap_df[metric])).astype(bool)
ap_df = ap_df[~iv_mask]

rr_idx = [i for i in ap_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
rr_active = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()

# Sim1 results
iv_mask = ((ap_df_sim1[metric] > sd_cut * np.nanstd(ap_df_sim1[metric].values)) | np.isnan(ap_df_sim1[metric])).astype(bool)
ap_df_sim1 = ap_df_sim1[~iv_mask]

rr_idx = [i for i in ap_df_sim1.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df_sim1.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df_sim1.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active_sim1 = ap_df_sim1[(ap_df_sim1['state']=='active') & (ap_df_sim1.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb_sim1= ap_df_sim1[(ap_df_sim1['state']=='big_passive') & (ap_df_sim1.index.isin(rt_idx))].groupby(by='site').mean()
rr_active_sim1 = ap_df_sim1[(ap_df_sim1['state']=='active') & (ap_df_sim1.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb_sim1 = ap_df_sim1[(ap_df_sim1['state']=='big_passive') & (ap_df_sim1.index.isin(rr_idx))].groupby(by='site').mean()

# Sim2 results
iv_mask = ((ap_df_sim2[metric] > sd_cut * np.nanstd(ap_df_sim2[metric].values)) | np.isnan(ap_df_sim2[metric])).astype(bool)
ap_df_sim2 = ap_df_sim2[~iv_mask]

rr_idx = [i for i in ap_df_sim2.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df_sim2.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df_sim2.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active_sim2 = ap_df_sim2[(ap_df_sim2['state']=='active') & (ap_df_sim2.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb_sim2= ap_df_sim2[(ap_df_sim2['state']=='big_passive') & (ap_df_sim2.index.isin(rt_idx))].groupby(by='site').mean()
rr_active_sim2 = ap_df_sim2[(ap_df_sim2['state']=='active') & (ap_df_sim2.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb_sim2 = ap_df_sim2[(ap_df_sim2['state']=='big_passive') & (ap_df_sim2.index.isin(rr_idx))].groupby(by='site').mean()

# Full simulation results
iv_mask = ((ap_df_sim12[metric] > sd_cut * np.nanstd(ap_df_sim12[metric].values)) | np.isnan(ap_df_sim12[metric])).astype(bool)
ap_df_sim12 = ap_df_sim12[~iv_mask]

rr_idx = [i for i in ap_df_sim12.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df_sim12.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df_sim12.index if ('TARGET' in i) & ('REFERENCE' in i)]

rt_active_sim12 = ap_df_sim12[(ap_df_sim12['state']=='active') & (ap_df_sim12.index.isin(rt_idx))].groupby(by='site').mean()
rt_pb_sim12= ap_df_sim12[(ap_df_sim12['state']=='big_passive') & (ap_df_sim12.index.isin(rt_idx))].groupby(by='site').mean()
rr_active_sim12 = ap_df_sim12[(ap_df_sim12['state']=='active') & (ap_df_sim12.index.isin(rr_idx))].groupby(by='site').mean()
rr_pb_sim12 = ap_df_sim12[(ap_df_sim12['state']=='big_passive') & (ap_df_sim12.index.isin(rr_idx))].groupby(by='site').mean()


# plot summary
f, ax = plt.subplots(2, 1, figsize=(4, 8))

ax[0].bar([0, 1, 2, 3], 
            [((rt_active['dprime'] - rt_pb['dprime']) / (rt_active['dprime'] + rt_pb['dprime'])).mean(),
             ((rt_active_sim1['dprime'] - rt_pb_sim1['dprime']) / (rt_active_sim1['dprime'] + rt_pb_sim1['dprime'])).mean(),
             ((rt_active_sim2['dprime'] - rt_pb_sim2['dprime']) / (rt_active_sim2['dprime'] + rt_pb_sim2['dprime'])).mean(),
             ((rt_active_sim12['dprime'] - rt_pb_sim12['dprime']) / (rt_active_sim12['dprime'] + rt_pb_sim12['dprime'])).mean()],
             yerr=[((rt_active['dprime'] - rt_pb['dprime']) / (rt_active['dprime'] + rt_pb['dprime'])).sem(),
             ((rt_active_sim1['dprime'] - rt_pb_sim1['dprime']) / (rt_active_sim1['dprime'] + rt_pb_sim1['dprime'])).sem(),
             ((rt_active_sim2['dprime'] - rt_pb_sim2['dprime']) / (rt_active_sim2['dprime'] + rt_pb_sim2['dprime'])).sem(),
             ((rt_active_sim12['dprime'] - rt_pb_sim12['dprime']) / (rt_active_sim12['dprime'] + rt_pb_sim12['dprime'])).sem()],
             edgecolor='k', color='firebrick', lw=2)
ax[0].set_ylim((-0.02, 0.2))
ax[0].set_ylabel("Delta dprime")
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels(['Raw', '1st-order', '2nd-order', 'Full simulation'])

ax[1].bar([0, 1, 2, 3], 
            [((rr_active['dprime'] - rr_pb['dprime']) / (rr_active['dprime'] + rr_pb['dprime'])).mean(),
             ((rr_active_sim1['dprime'] - rr_pb_sim1['dprime']) / (rr_active_sim1['dprime'] + rr_pb_sim1['dprime'])).mean(),
             ((rr_active_sim2['dprime'] - rr_pb_sim2['dprime']) / (rr_active_sim2['dprime'] + rr_pb_sim2['dprime'])).mean(),
             ((rr_active_sim12['dprime'] - rr_pb_sim12['dprime']) / (rr_active_sim12['dprime'] + rr_pb_sim12['dprime'])).mean()],
             yerr=[((rr_active['dprime'] - rr_pb['dprime']) / (rr_active['dprime'] + rr_pb['dprime'])).sem(),
             ((rr_active_sim1['dprime'] - rr_pb_sim1['dprime']) / (rr_active_sim1['dprime'] + rr_pb_sim1['dprime'])).sem(),
             ((rr_active_sim2['dprime'] - rr_pb_sim2['dprime']) / (rr_active_sim2['dprime'] + rr_pb_sim2['dprime'])).sem(),
             ((rr_active_sim12['dprime'] - rr_pb_sim12['dprime']) / (rr_active_sim12['dprime'] + rr_pb_sim12['dprime'])).sem()],
             edgecolor='k', color='navy', lw=2)

ax[1].set_ylim((-.1, .1))
ax[1].set_ylabel("Delta dprime")
ax[1].set_xticks([0, 1, 2, 3])
ax[1].set_xticklabels(['Raw', '1st-order', '2nd-order', 'Full simulation'])

f.tight_layout()

f.savefig(fn)

plt.show()
