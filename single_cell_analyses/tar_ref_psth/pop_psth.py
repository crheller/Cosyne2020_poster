import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import charlieTools.plotting as cplt

results_path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/single_cell_analyses/results/'
longLat = False
shortLat = False
midLat = False
onBF = False
offBF = False
singleTar = True
site = None
cmap = 'jet'
fs = 20
latency = 'peak_latency'  # cm_latency, peak_latency
flip_sign = 'peak_sign'   # cm_sign, cm_latency
sort_cols = [flip_sign, latency]
latency_cutoff = 7 + 8 # used if short/longlat is true (bins from starts of trial, there are 7 spont bins)

# Load psths
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

if site is not None:
    mask = ref_active['site'] == site

if longLat:
    mask = mask & (tar_active[latency] > latency_cutoff) # keep only long latency target responses

if shortLat:
    mask = mask & (tar_active[latency] < latency_cutoff) # keep only short latency target responses

if midLat:
    mask = mask & (tar_active[latency] < 16) & (tar_active[latency] > 10)

# masks to deal with BF / OFF BF cells... For this, for now, 
# only look at sites with a single target frequency
unique_freq = [[int(t.split('_')[1]) for t in k] for k in ref_active['targets']]
unique_freq = np.array([x[0] if (len(x)==1) else x[0] if (abs(np.diff(x).max())<=1) else False for x in unique_freq])

if onBF | offBF | singleTar:
    mask = mask & (unique_freq != False)

# on BF
if onBF:
    mask = mask & ((ref_active['BF'] < (unique_freq + (0.3 * unique_freq))) & (ref_active['BF'] > (unique_freq - (0.3 * unique_freq))))

# OFF BF
if offBF:
    mask = mask & ((ref_active['BF'] > (unique_freq + (0.3 * unique_freq))) | (ref_active['BF'] < (unique_freq - (0.3 * unique_freq))))

# ======================= Active vs. Big pupil ===============================
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

f.suptitle('n = {}'.format(mask.sum()), fontsize=8)

# Reference subplot
u = ref_active[tcols][mask].mean(axis=0)
sem = ref_active[tcols][mask].sem(axis=0)
ax[0].plot(tcols, u, color='blue', lw=2, label='active')
ax[0].fill_between(tcols, u-sem, u+sem, color='blue', alpha=0.5, lw=0)

u = ref_passiveBig[tcols][mask].mean(axis=0)
sem = ref_passiveBig[tcols][mask].sem(axis=0)
ax[0].plot(tcols, u, linestyle='--', color='blue', lw=2, label='passive, big pupil')
ax[0].fill_between(tcols, u-sem, u+sem, color='lightblue', alpha=0.5, lw=0)

u = ref_passiveSmall[tcols][mask].mean(axis=0)
sem = ref_passiveSmall[tcols][mask].sem(axis=0)
ax[0].plot(tcols, u, linestyle='dotted', color='purple', lw=2, label='passive, small pupil')
ax[0].fill_between(tcols, u-sem, u+sem, color='orchid', alpha=0.5, lw=0)

ax[0].legend(frameon=False, fontsize=8)
ax[0].set_xlabel('Time from sound onset', fontsize=8)
ax[0].set_ylabel('Normalized response', fontsize=8)
ax[0].set_title('Reference')

# Target subplot
u = tar_active[tcols][mask].mean(axis=0)
sem = tar_active[tcols][mask].sem(axis=0)
ax[1].plot(tcols, u, color='red', lw=2, label='active')
ax[1].fill_between(tcols, u-sem, u+sem, color='red', alpha=0.5, lw=0)

u = tar_passiveBig[tcols][mask].mean(axis=0)
sem = tar_passiveBig[tcols][mask].sem(axis=0)
ax[1].plot(tcols, u, linestyle='--', color='red', lw=2, label='passive, big pupil')
ax[1].fill_between(tcols, u-sem, u+sem, color='coral', alpha=0.5, lw=0)

u = tar_passiveSmall[tcols][mask].mean(axis=0)
sem = tar_passiveSmall[tcols][mask].sem(axis=0)
ax[1].plot(tcols, u, linestyle='dotted', color='darkorange', lw=2, label='passive, small pupil')
ax[1].fill_between(tcols, u-sem, u+sem, color='bisque', alpha=0.5, lw=0)

ax[1].legend(frameon=False, fontsize=8)
ax[1].set_xlabel('Time from sound onset', fontsize=8)
ax[1].set_title('Target')

f.tight_layout()

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
ax[0].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[1].set_title('Passive big pupil', fontsize=8)

ax[1].imshow(tpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1].axhline(groupdiv, color='white', lw=2)
ax[1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax[1].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[2].set_title('Active', fontsize=8)

ax[2].imshow(ta, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[2].axhline(groupdiv, color='white', lw=2)
ax[2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax[2].plot(tar_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

f.tight_layout()

f.canvas.set_window_title('Target PSTH')

# plot difference in psth between active and big passive
f, ax = plt.subplots(1, 1, figsize=(6, 8))

# sort by sign of difference and latency to peak difference
diff = ta - tpb
lat = np.argmax(abs(diff[:, 7:]), axis=-1) + 7
sign = np.sign(diff[range(0, diff.shape[0]), lat])

sort_idx = np.lexsort(np.concatenate((lat[np.newaxis,:], sign[np.newaxis, :]), axis=0))
diff = diff[sort_idx, :]

vmax = np.nanstd(diff) * 2
vmin = -vmax

ax.set_title('Active - Big passive', fontsize=8)

ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax.axhline(groupdiv, color='white', lw=2)
ax.set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax.set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax.axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax.axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax.plot(lat[sort_idx], np.arange(0, len(sort_idx)), 'k') 

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
ax[0].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[1].set_title('Passive big pupil', fontsize=8)

ax[1].imshow(rpb, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[1].axhline(groupdiv, color='white', lw=2)
ax[1].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[1].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[1].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[1].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax[1].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

ax[2].set_title('Active', fontsize=8)

ax[2].imshow(ra, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax[2].axhline(groupdiv, color='white', lw=2)
ax[2].set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax[2].set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax[2].axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax[2].axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax[2].plot(ref_active[mask].loc[sorted_cellids][latency], np.arange(0, len(sorted_cellids)), 'k') 

f.tight_layout()

f.canvas.set_window_title('Reference PSTH')

# plot difference in psth between active and big passive
f, ax = plt.subplots(1, 1, figsize=(6, 8))

# sort by sign of difference and latency to peak difference
diff = ra - rpb
lat = np.argmax(abs(diff[:, 7:]), axis=-1) + 7
sign = np.sign(diff[range(0, diff.shape[0]), lat])

sort_idx = np.lexsort(np.concatenate((lat[np.newaxis,:], sign[np.newaxis, :]), axis=0))
diff = diff[sort_idx, :]

vmax = np.nanstd(diff) * 2
vmin = -vmax

ax.set_title('Active - Big passive', fontsize=8)

ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax.axhline(groupdiv, color='white', lw=2)
ax.set_xticks(np.arange(0, len(tcols))[1:-1:3]-0.5)
ax.set_xticklabels(np.array(tcols)[1:-1:3] / fs, fontsize=8)
ax.axvline(np.argwhere(np.array(tcols)==0)-0.5, color='k', linestyle='--')
ax.axvline(np.argwhere(np.array(tcols)==int(0.75 * 20))-0.5, color='k', linestyle='--')
ax.plot(lat[sort_idx], np.arange(0, len(sort_idx)), 'k') 

f.canvas.set_window_title('REFERENCE A vs. BP diff')

f.tight_layout()

plt.show()