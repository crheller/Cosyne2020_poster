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

xlabel = 'Behavioral Performance (DI)'
ylabel = 'Delta Noise Correlation'

beh_metric = 'DI_collapse'

rsc_df = ld.load_rsc('tar_rsc_0_0,2')
rsc_df_site = rsc_df.groupby(by='site').mean()

behavior = pd.read_pickle('/auto/users/hellerc/code/projects/Cosyne2020_poster/behavior/results.pickle')
DI_collapse = [np.mean(di) for di in behavior['all_DI']]
dprime_collapse = [np.mean(di) for di in behavior['all_dprime']]
behavior['DI_collapse'] = DI_collapse
behavior['dprime_collapse'] = dprime_collapse

df = pd.concat([rsc_df_site, behavior['DI_collapse']], axis=1)

f, ax = plt.subplots(1, 3, figsize=(8, 4))

ax[0].scatter(df[beh_metric], df['pass_rsc'] - df['act_rsc'], s=50, color='k', edgecolor='white')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df[beh_metric], df['pass_rsc'] - df['act_rsc'])
cc = round(cc, 3)
pval = round(pval, 3)
ax[0].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[0].set_ylabel(ylabel)
ax[0].set_xlabel(xlabel)

ax[1].scatter(df[beh_metric], df['passBig_rsc'] - df['act_rsc'], s=50, color='k', edgecolor='white')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.6, linestyle='--', color='k')
cc, pval = ss.pearsonr(df[beh_metric], df['passBig_rsc'] - df['act_rsc'])
cc = round(cc, 3)
pval = round(pval, 3)
ax[1].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[1].set_ylabel(ylabel)
ax[1].set_xlabel(xlabel)

ax[2].scatter(df[beh_metric], df['passSmall_rsc'] - df['passBig_rsc'], s=50, color='k', edgecolor='white')
ax[2].axhline(0, linestyle='--', color='k')
ax[2].axvline(0.6, linestyle='--', color='k')
idx = ~np.isnan(df['passSmall_rsc'] - df['passBig_rsc'])
cc, pval = ss.pearsonr(df[beh_metric][idx], (df['passSmall_rsc'] - df['passBig_rsc'])[idx])
cc = round(cc, 3)
pval = round(pval, 3)
ax[2].set_title('r: {0}, p: {1}'.format(cc, pval))
ax[2].set_ylabel(ylabel)
ax[2].set_xlabel(xlabel)


xlims = (np.min([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()]), np.max([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()]))
ylims = (np.min([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()]), np.max([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()]))
ax[0].set_xlim(xlims)
ax[0].set_ylim(ylims)
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)
ax[2].set_xlim(xlims)
ax[2].set_ylim(ylims)

ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

plt.show()