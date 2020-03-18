"""
Small Pupil, Big Pupil, Active PSTH heatmap (inclue big vs. active diff?)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/heatmap.svg'

results_path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/single_cell_analyses/results/'

flip_sign = 'cm_sign'
latency = 'cm_latency'
sort_cols = [flip_sign, latency]
cmap = 'jet'
fs = 20
sp_bins = 7
sd = 3
balanced_only = True

ref_active = pd.read_pickle(results_path+'ref_active_norm.pickle')
ref_passiveBig = pd.read_pickle(results_path+'ref_bigPassive_norm.pickle')
ref_passiveSmall = pd.read_pickle(results_path+'ref_smallPassive_norm.pickle')

tar_active = pd.read_pickle(results_path+'tar_active_norm.pickle')
tar_passiveBig = pd.read_pickle(results_path+'tar_bigPassive_norm.pickle')
tar_passiveSmall = pd.read_pickle(results_path+'tar_smallPassive_norm.pickle')

f, ax = plt.subplots(2, 3, figsize=(12, 8))

tcols = [i for i in ref_active.columns if type(i)==int]

# ================================ TARGETS ============================================
sorted_cellids = tar_active.sort_values(by=sort_cols).index
if balanced_only:
    small_cellids = [s for s in sorted_cellids if s in tar_passiveSmall.index]
    sorted_cellids = small_cellids

tps = tar_passiveSmall[tcols].loc[sorted_cellids].values
tps = tps / tar_passiveSmall.loc[sorted_cellids][flip_sign].values[:, np.newaxis]
tpb = tar_passiveBig[tcols].loc[sorted_cellids].values
tpb = tpb / tar_passiveBig.loc[sorted_cellids][flip_sign].values[:, np.newaxis]
ta = tar_active[tcols].loc[sorted_cellids].values
ta = ta / tar_active.loc[sorted_cellids][flip_sign].values[:, np.newaxis]

# =============================== REFERENCES ==========================================
sorted_cellids = ref_active.sort_values(by=sort_cols).index
if balanced_only:
    small_cellids = [s for s in sorted_cellids if s in ref_passiveSmall.index]
    sorted_cellids = small_cellids

rps = ref_passiveSmall[tcols].loc[sorted_cellids].values
rps = rps / ref_passiveSmall.loc[sorted_cellids][flip_sign].values[:, np.newaxis]
rpb = ref_passiveBig[tcols].loc[sorted_cellids].values
rpb = rpb / ref_passiveBig.loc[sorted_cellids][flip_sign].values[:, np.newaxis]
ra = ref_active[tcols].loc[sorted_cellids].values
ra = ra / ref_active.loc[sorted_cellids][flip_sign].values[:, np.newaxis]

vmax = np.nanstd(np.concatenate((ta, tpb, tps, rps, rpb, ra))) * sd
vmin = -vmax

ax[0, 0].set_title('Passive small pupil')

ax[0, 0].imshow(tps, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[0, 0].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[0, 0].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[0, 0].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[0, 0].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')

ax[0, 1].set_title('Passive big pupil')

ax[0, 1].imshow(tpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[0, 1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[0, 1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[0, 1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[0, 1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')

ax[0, 2].set_title('Active')

ax[0, 2].imshow(ta, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[0, 2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[0, 2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[0, 2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[0, 2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')


ax[1, 0].imshow(rps, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1, 0].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1, 0].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1, 0].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1, 0].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')

ax[1, 1].imshow(rpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1, 1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1, 1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1, 1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1, 1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')

ax[1, 2].imshow(ra, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1, 2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1, 2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1, 2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1, 2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')

f.tight_layout()

f.savefig(fn)

plt.show()