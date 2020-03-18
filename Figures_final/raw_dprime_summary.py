"""
Raw dprime results for a vs. bp, bp vs. sp, ref/tar and ref/ref. 
Scatter plots (2x2), two bar plots (one for ref/tar one for ref/ref), 
two bars each (one for act/big and one for big/small)
"""
import loading as ld
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import charlieTools.plotting as cplt
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/raw_dprime_summary.svg'

ap_df = ld.load_dprime('refTar_pca4_dprime_act_bigPass_binPreLick')
pbps_df = ld.load_dprime('refTar_pca4_dprime_bigPass_smallPass_binPreLick')

f, ax = plt.subplots(2, 3, figsize=(8, 6))

axis = 'ssa'
metric = 'dprime'
bysite = False
sd_cut = 3.5
ylim = 15
xlim = 15
vmin = 0
vmax = ylim

ap_df = ap_df[ap_df['axis']==axis]
pbps_df = pbps_df[pbps_df['axis']==axis]

# compare active vs. big pupil
#iv_mask = ((ap_df[metric] > sd_cut * np.nanstd(ap_df[metric].values)) | np.isnan(ap_df[metric])).astype(bool)
#ap_df = ap_df[~iv_mask]

rr_idx = [i for i in ap_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in ap_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in ap_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

if bysite:
    rt_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
    rt_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rt_idx))].groupby(by='site').mean()
    rr_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
    rr_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rr_idx))].groupby(by='site').mean()
else:
    rt_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rt_idx))]
    rt_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rt_idx))]
    rr_active_bysite = ap_df[(ap_df['state']=='active') & (ap_df.index.isin(rr_idx))]
    rr_pb_bysite = ap_df[(ap_df['state']=='big_passive') & (ap_df.index.isin(rr_idx))]


ax[0, 0].scatter(rt_pb_bysite[metric], rt_active_bysite[metric], color='k', edgecolor='white')
#ax[0, 0].scatter(rt_pb_bysite[metric].loc['TAR010c'], rt_active_bysite[metric].loc['TAR010c'], color='r', edgecolor='white')
ax[0, 0].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[0, 0].set_xlabel('Big Passive')
ax[0, 0].set_ylabel('Active')
ax[0, 0].set_ylim((0, ylim))
ax[0, 0].set_xlim((0, xlim))
ax[0, 0].set_aspect(cplt.get_square_asp(ax[0, 0]))

ax[1, 0].scatter(rr_pb_bysite[metric], rr_active_bysite[metric], color='k', edgecolor='white')
#ax[1, 0].scatter(rr_pb_bysite[metric].loc['TAR010c'], rr_active_bysite[metric].loc['TAR010c'], color='r', edgecolor='white')
ax[1, 0].plot([vmin, vmax], [vmin, vmax], 'k--')
ax[1, 0].set_xlabel('Big Passive')
ax[1, 0].set_ylabel('Active')
ax[1, 0].set_ylim((0, ylim))
ax[1, 0].set_xlim((0, xlim))
ax[1, 0].set_aspect(cplt.get_square_asp(ax[1, 0]))


# compare big pupil vs. small pupil
#iv_mask = ((pbps_df[metric] > sd_cut * np.nanstd(pbps_df[metric].values)) | np.isnan(pbps_df[metric])).astype(bool)
#pbps_df = pbps_df[~iv_mask]

rr_idx = [i for i in pbps_df.index if ('TORC' in i) & (('REFERENCE' not in i) & ('TARGET' not in i) & ('TAR_' not in i))]
rt_idx = [i for i in pbps_df.index if (sum(['TORC' in i for i in i.split('_')]) == 1) & \
                                ('REFERENCE' not in i) & ('TARGET' not in i)]
rt_idx = [i for i in pbps_df.index if ('TARGET' in i) & ('REFERENCE' in i)]

if bysite:
    rt_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rt_idx))].groupby(by='site').mean()
    rt_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rt_idx))].groupby(by='site').mean()
    rr_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rr_idx))].groupby(by='site').mean()
    rr_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rr_idx))].groupby(by='site').mean()
else:
    rt_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rt_idx))]
    rt_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rt_idx))]
    rr_pb2_bysite = pbps_df[(pbps_df['state']=='big_passive') & (pbps_df.index.isin(rr_idx))]
    rr_ps_bysite = pbps_df[(pbps_df['state']=='small_passive') & (pbps_df.index.isin(rr_idx))]

if bysite:
    sites = [s for s in rt_pb2_bysite.index if s in rt_ps_bysite.index]
    ax[0, 1].scatter(rt_ps_bysite[metric].loc[sites], rt_pb2_bysite[metric].loc[sites], color='k', edgecolor='white')
    ax[0, 1].scatter(rt_ps_bysite[metric].loc['TAR010c'], rt_pb2_bysite[metric].loc['TAR010c'], color='r', edgecolor='white')
else:
    ax[0, 1].scatter(rt_ps_bysite[metric], rt_pb2_bysite[metric], color='k', edgecolor='white')

ax[0, 1].plot([vmin, 7], [vmin, 7], 'k--')
ax[0, 1].set_xlabel('Small Passive')
ax[0, 1].set_ylabel('Big Passive')
ax[0, 1].set_ylim((0, 7))
ax[0, 1].set_xlim((0, 7))
ax[0, 1].set_aspect(cplt.get_square_asp(ax[0, 1]))

if bysite:
    ax[1, 1].scatter(rr_ps_bysite[metric].loc[sites], rr_pb2_bysite[metric].loc[sites], color='k', edgecolor='white')
    ax[1, 1].scatter(rr_ps_bysite[metric].loc['TAR010c'], rr_pb2_bysite[metric].loc['TAR010c'], color='r', edgecolor='white')
else:
    ax[1, 1].scatter(rr_ps_bysite[metric], rr_pb2_bysite[metric], color='k', edgecolor='white')
ax[1, 1].plot([vmin, 7], [vmin, 7], 'k--')
ax[1, 1].set_xlabel('Small Passive')
ax[1, 1].set_ylabel('Big Passive')
ax[1, 1].set_ylim((0, 7))
ax[1, 1].set_xlim((0, 7))
ax[1, 1].set_aspect(cplt.get_square_asp(ax[1, 1]))


# bar plot of MI to summarize effects
reftar_ap = (rt_active_bysite[metric] - rt_pb_bysite[metric]) / (rt_active_bysite[metric] + rt_pb_bysite[metric])
reftar_bs = (rt_pb2_bysite[metric] - rt_ps_bysite[metric]) / (rt_pb2_bysite[metric] + rt_ps_bysite[metric])
ax[0, 2].bar([0, 1], [reftar_ap.mean(), reftar_bs.mean()], 
                    yerr=[reftar_ap.sem(), reftar_bs.sem()], color=['firebrick', 'lightcoral'], edgecolor='k', lw=2)
ax[0, 2].set_xticks([0, 1])
ax[0, 2].set_xticklabels(['A vs. Big', 'Big vs. Small'])
ax[0, 2].set_ylabel('Dprime MI')
ax[0, 2].set_ylim((-0.05, 0.25))
ax[0, 2].set_aspect(cplt.get_square_asp(ax[0, 2]))

refref_ap = (rr_active_bysite[metric] - rr_pb_bysite[metric]) / (rr_active_bysite[metric] + rr_pb_bysite[metric])
refref_bs = (rr_pb2_bysite[metric] - rr_ps_bysite[metric]) / (rr_pb2_bysite[metric] + rr_ps_bysite[metric])
ax[1, 2].bar([0, 1], [refref_ap.mean(), refref_bs.mean()], 
                    yerr=[refref_ap.sem(), refref_bs.sem()], color=['navy', 'lightskyblue'], edgecolor='k', lw=2)
ax[1, 2].set_xticks([0, 1])
ax[1, 2].set_xticklabels(['A vs. Big', 'Big vs. Small'])
ax[1, 2].set_ylabel('Dprime MI')
ax[1, 2].set_ylim((-0.05, 0.15))
ax[1, 2].set_aspect(cplt.get_square_asp(ax[1, 2]))

f.tight_layout()

f.savefig(fn)

plt.show()