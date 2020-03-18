"""
Population heatmap in the style Fritz 2010.
Plot for big / small comparison and big / active comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/single_cell_analyses/results/'

flip_sign = 'peak_sign'
latency = 'peak_latency'
sort_cols = [flip_sign, latency]
cmap = 'plasma'
fs = 20
sp_bins = 7

ref_active = pd.read_pickle(results_path+'ref_active_norm.pickle')
ref_passive = pd.read_pickle(results_path+'ref_passive_norm.pickle')
ref_passiveBig = pd.read_pickle(results_path+'ref_bigPassive_norm.pickle')
ref_passiveSmall = pd.read_pickle(results_path+'ref_smallPassive_norm.pickle')

tar_active = pd.read_pickle(results_path+'tar_active_norm.pickle')
tar_passive = pd.read_pickle(results_path+'tar_passive_norm.pickle')
tar_passiveBig = pd.read_pickle(results_path+'tar_bigPassive_norm.pickle')
tar_passiveSmall = pd.read_pickle(results_path+'tar_smallPassive_norm.pickle')

tcols = [i for i in ref_active.columns if type(i)==int]

mask = ref_active['sig'] | tar_active['sig'] # keep all cells with sig response for either REF or TAR

#mask = mask & [i[:7] not in bad_sites for i in ref_active.index]

# ==================================== TARGETS =================================================
f, ax = plt.subplots(1, 3, figsize=(12, 8))

sorted_cellids = tar_active[mask].sort_values(by=sort_cols).index
groupdiv = np.argwhere(np.diff(tar_active[mask].loc[sorted_cellids][sort_cols[0]])>0)[0][0]

tps = tar_passiveSmall[mask][tcols].loc[sorted_cellids].values
tps = tps / tar_passiveSmall[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]
tpb = tar_passiveBig[mask][tcols].loc[sorted_cellids].values
tpb = tpb / tar_passiveBig[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]
ta = tar_active[mask][tcols].loc[sorted_cellids].values
ta = ta / tar_active[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]

vmax = np.nanstd(np.concatenate((ta, tpb, tps))) * 2
vmin = -vmax

ax[0].set_title('Passive small pupil', fontsize=8)

ax[0].imshow(tps, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[0].axhline(groupdiv, color='white', lw=2)
ax[0].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[0].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[0].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[0].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[0].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[1].set_title('Passive big pupil', fontsize=8)

ax[1].imshow(tpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1].axhline(groupdiv, color='white', lw=2)
ax[1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[1].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[2].set_title('Active', fontsize=8)

ax[2].imshow(ta, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[2].axhline(groupdiv, color='white', lw=2)
ax[2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[2].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

f.tight_layout()

f.canvas.set_window_title('Target PSTH')

# plot difference in psth between active and big passive
f, ax = plt.subplots(1, 1, figsize=(6, 8))

# sort by sign of difference and latency to peak difference
diff = ta - tpb
lat = np.argmax(abs(diff[:, sp_bins:]), axis=-1) + 7
#sign = np.sign(diff[range(0, diff.shape[0]), lat])
sign = diff[range(0, diff.shape[0]), lat]

sort_idx = np.lexsort(np.concatenate((lat[np.newaxis,:], sign[np.newaxis, :]), axis=0))
diff = diff[sort_idx, :]

vmax = np.nanstd(diff) * 2
vmin = -vmax

ax.set_title('Active - Big passive', fontsize=8)

ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
#ax.axhline(groupdiv, color='white', lw=2)
ax.set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax.set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax.axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax.axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax.plot(lat[sort_idx], np.arange(0, len(sort_idx)), 'k') 

f.canvas.set_window_title('TARGET A vs. BP diff')

f.tight_layout()

# ================================== REFERENCES ==================================================
f, ax = plt.subplots(1, 3, figsize=(12, 8))

sorted_cellids = ref_active[mask].sort_values(by=sort_cols).index
groupdiv = np.argwhere(np.diff(ref_active[mask].loc[sorted_cellids][sort_cols[0]])>0)[0][0]

rps = ref_passiveSmall[mask][tcols].loc[sorted_cellids].values
rps = rps / ref_passiveSmall[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]
rpb = ref_passiveBig[mask][tcols].loc[sorted_cellids].values
rpb = rpb / ref_passiveBig[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]
ra = ref_active[mask][tcols].loc[sorted_cellids].values
ra = ra / ref_active[mask].loc[sorted_cellids][flip_sign].values[:, np.newaxis]

vmax = np.nanstd(np.concatenate((ra, rpb, rps))) * 2
vmin = -vmax

ax[0].set_title('Passive small pupil', fontsize=8)

ax[0].imshow(rps, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[0].axhline(groupdiv, color='white', lw=2)
ax[0].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[0].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[0].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[0].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[0].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[1].set_title('Passive big pupil', fontsize=8)

ax[1].imshow(rpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1].axhline(groupdiv, color='white', lw=2)
ax[1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[1].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[2].set_title('Active', fontsize=8)

ax[2].imshow(ra, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[2].axhline(groupdiv, color='white', lw=2)
ax[2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax[2].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

f.tight_layout()

f.canvas.set_window_title('Reference PSTH')

# plot difference in psth between active and big passive
f, ax = plt.subplots(1, 1, figsize=(6, 8))

# sort by sign of difference and latency to peak difference
diff = ra - rpb
lat = np.argmax(abs(diff[:, sp_bins:]), axis=-1) + 7
sign = np.sign(diff[range(0, diff.shape[0]), lat])

sort_idx = np.lexsort(np.concatenate((lat[np.newaxis,:], sign[np.newaxis, :]), axis=0))
diff = diff[sort_idx, :]

vmax = np.nanstd(diff) * 2
vmin = -vmax

ax.set_title('Active - Big passive', fontsize=8)

ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
#ax.axhline(groupdiv, color='white', lw=2)
ax.set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax.set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax.axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax.axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
#ax.plot(lat[sort_idx], np.arange(0, len(sort_idx)), 'k') 

f.canvas.set_window_title('REFERENCE A vs. BP diff')

f.tight_layout()

plt.show()