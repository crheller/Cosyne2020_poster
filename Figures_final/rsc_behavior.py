"""
Magnitude of change in rsc correlated with behavioral performance
"""

import matplotlib.pyplot as plt
import loading as ld
import charlieTools.plotting as cplt
import pandas as pd
import scipy.stats as ss
import numpy as np
import os
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/rsc_behavior.svg'

xlabel = 'Behavioral Performance (DI)'
ylabel = 'Delta Noise Correlation'
normalize = True
beh_metric = 'DI_collapse'

rsc_df = ld.load_rsc('tar_rsc_0_0,1')
rsc_df_site = rsc_df.groupby(by='site').mean()

rsc_df2 = ld.load_rsc('tar_rsc_0,1_0,2')
rsc_df_site2 = rsc_df2.groupby(by='site').mean()

rsc_df3 = ld.load_rsc('tar_rsc_0,2_0,3')
rsc_df_site3 = rsc_df3.groupby(by='site').mean()

rsc_df4 = ld.load_rsc('tar_rsc_0,3_0,4')
rsc_df_site4 = rsc_df4.groupby(by='site').mean()

behavior = pd.read_pickle('/auto/users/hellerc/code/projects/Cosyne2020_poster/behavior/results.pickle')
DI_collapse = [np.mean(di) for di in behavior['all_DI']]
dprime_collapse = [np.mean(di) for di in behavior['all_dprime']]
behavior['DI_collapse'] = DI_collapse
behavior['dprime_collapse'] = dprime_collapse

df = pd.concat([rsc_df_site, behavior[beh_metric]], axis=1)
df2 = pd.concat([rsc_df_site2, behavior[beh_metric]], axis=1)
df3 = pd.concat([rsc_df_site3, behavior[beh_metric]], axis=1)
df4 = pd.concat([rsc_df_site4, behavior[beh_metric]], axis=1)

f, ax = plt.subplots(1, 4, figsize=(12, 4))

if normalize:
    norm = (abs(df['passBig_rsc']) + abs(df['act_rsc']))
else:
    norm = 1
ax[0].scatter(df[beh_metric], (df['passBig_rsc'] - df['act_rsc']) / norm, s=50, color='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df[beh_metric], (df['passBig_rsc'] - df['act_rsc']) / norm)
cc = round(cc, 3)
pval = round(pval, 3)
ax[0].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[0].set_ylabel(ylabel)
ax[0].set_xlabel(xlabel)

if normalize:
    norm = (abs(df2['passBig_rsc']) + abs(df2['act_rsc']))
else:
    norm = 1
ax[1].scatter(df2[beh_metric], (df2['passBig_rsc'] - df2['act_rsc']) / norm, s=50, color='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df2[beh_metric], (df2['passBig_rsc'] - df2['act_rsc']) / norm)
cc = round(cc, 3)
pval = round(pval, 3)
ax[1].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[1].set_ylabel(ylabel)
ax[1].set_xlabel(xlabel)

if normalize:
    norm = (abs(df3['passBig_rsc']) + abs(df3['act_rsc']))
else:
    norm = 1
ax[2].scatter(df3[beh_metric], (df3['passBig_rsc'] - df3['act_rsc']) / norm, s=50, color='k', edgecolor='white')
ax[2].axhline(0, linestyle='--', color='k')
ax[2].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df3[beh_metric], (df3['passBig_rsc'] - df3['act_rsc']) / norm)
cc = round(cc, 3)
pval = round(pval, 3)
ax[2].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[2].set_ylabel(ylabel)
ax[2].set_xlabel(xlabel)

if normalize:
    norm = (abs(df4['passBig_rsc']) + abs(df4['act_rsc']))
else:
    norm = 1
ax[3].scatter(df4[beh_metric], (df4['passBig_rsc'] - df4['act_rsc']) / norm, s=50, color='k', edgecolor='white')
ax[3].axhline(0, linestyle='--', color='k')
ax[3].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df4[beh_metric], (df4['passBig_rsc'] - df4['act_rsc']) / norm)
cc = round(cc, 3)
pval = round(pval, 3)
ax[3].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[3].set_ylabel(ylabel)
ax[3].set_xlabel(xlabel)

ax[0].set_ylim((-1.1, 1.1))
ax[1].set_ylim((-1.1, 1.1))
ax[2].set_ylim((-1.1, 1.1))
ax[3].set_ylim((-1.1, 1.1))

ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))
ax[2].set_aspect(cplt.get_square_asp(ax[2]))
ax[3].set_aspect(cplt.get_square_asp(ax[3]))


f.tight_layout()

f.savefig(fn)

# across time windows:
corr_coef = []
windows = ['0_0,1', '0,1_0,2', '0,2_0,3', '0,3_0,4', '0,4_0,5', '0,5_0,6', '0,6_0,7', '0,7_0,8']    
for win in windows:
    rsc_df = ld.load_rsc('tar_rsc_{}'.format(win))
    rsc_df_site = rsc_df.groupby(by='site').mean()
    df = pd.concat([rsc_df_site, behavior[beh_metric]], axis=1)
    cc, pval = ss.pearsonr(df[beh_metric], df['passBig_rsc'] - df['act_rsc'] / (df['passBig_rsc'] + df['act_rsc']))
    corr_coef.append(cc)

f, ax = plt.subplots(1, 1)

ax.plot(corr_coef, 'o-', color='coral')
ax.set_xticks(range(0, len(windows)))
ax.set_xticklabels(windows)
ax.axhline(0, linestyle='--', color='k')

plt.show()