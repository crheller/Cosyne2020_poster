"""
Plot long latency / short latency population PSTHs for
all three states (Like Atiani & David plot)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/pop_psth.svg'

lat_div = 17
lat_metric = 'cm_latency'
balanced = True

results_path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/single_cell_analyses/results/'

ref_active = pd.read_pickle(results_path+'ref_active_norm.pickle')
ref_passive = pd.read_pickle(results_path+'ref_passive_norm.pickle')
ref_passiveBig = pd.read_pickle(results_path+'ref_bigPassive_norm.pickle')
ref_passiveSmall = pd.read_pickle(results_path+'ref_smallPassive_norm.pickle')

tar_active = pd.read_pickle(results_path+'tar_active_norm.pickle')
tar_passive = pd.read_pickle(results_path+'tar_passive_norm.pickle')
tar_passiveBig = pd.read_pickle(results_path+'tar_bigPassive_norm.pickle')
tar_passiveSmall = pd.read_pickle(results_path+'tar_smallPassive_norm.pickle')

tcols = [i for i in ref_active.columns if type(i)==int]

mask = ref_active['sig'] | tar_active['sig']

if balanced:
    mask = mask & [True if (ref_active.index[i] in ref_passiveSmall.index) else False for i in range(0, ref_active.shape[0])]

short_mask = mask & (tar_active[lat_metric] <= lat_div)
long_mask = mask & (tar_active[lat_metric] >= lat_div)

f, ax = plt.subplots(2, 2, figsize=(6, 6), sharey=True)


# Target subplots
u = tar_active[tcols][short_mask].mean(axis=0)
sem = tar_active[tcols][short_mask].sem(axis=0)
ax[0, 0].plot(tcols, u, color='red', lw=2, label='active')
ax[0, 0].fill_between(tcols, u-sem, u+sem, color='red', alpha=0.5, lw=0)

u = tar_passiveBig[tcols][short_mask].mean(axis=0)
sem = tar_passiveBig[tcols][short_mask].sem(axis=0)
ax[0, 0].plot(tcols, u, linestyle='--', color='red', lw=2, label='passive, big pupil')
ax[0, 0].fill_between(tcols, u-sem, u+sem, color='coral', alpha=0.5, lw=0)

u = tar_passiveSmall[tcols][short_mask].mean(axis=0)
sem = tar_passiveSmall[tcols][short_mask].sem(axis=0)
ax[0, 0].plot(tcols, u, linestyle='dotted', color='darkorange', lw=2, label='passive, small pupil')
ax[0, 0].fill_between(tcols, u-sem, u+sem, color='bisque', alpha=0.5, lw=0)

ax[0, 0].legend(frameon=False, fontsize=8)
ax[0, 0].set_xlabel('Time from sound onset', fontsize=8)
ax[0, 0].set_title('Target')

u = tar_active[tcols][long_mask].mean(axis=0)
sem = tar_active[tcols][long_mask].sem(axis=0)
ax[0, 1].plot(tcols, u, color='red', lw=2, label='active')
ax[0, 1].fill_between(tcols, u-sem, u+sem, color='red', alpha=0.5, lw=0)

u = tar_passiveBig[tcols][long_mask].mean(axis=0)
sem = tar_passiveBig[tcols][long_mask].sem(axis=0)
ax[0, 1].plot(tcols, u, linestyle='--', color='red', lw=2, label='passive, big pupil')
ax[0, 1].fill_between(tcols, u-sem, u+sem, color='coral', alpha=0.5, lw=0)

u = tar_passiveSmall[tcols][long_mask].mean(axis=0)
sem = tar_passiveSmall[tcols][long_mask].sem(axis=0)
ax[0, 1].plot(tcols, u, linestyle='dotted', color='darkorange', lw=2, label='passive, small pupil')
ax[0, 1].fill_between(tcols, u-sem, u+sem, color='bisque', alpha=0.5, lw=0)

ax[0, 1].legend(frameon=False, fontsize=8)
ax[0, 1].set_xlabel('Time from sound onset', fontsize=8)
ax[0, 1].set_title('Target')


# Reference subplot
u = ref_active[tcols][short_mask].mean(axis=0)
sem = ref_active[tcols][short_mask].sem(axis=0)
ax[1, 0].plot(tcols, u, color='blue', lw=2, label='active')
ax[1, 0].fill_between(tcols, u-sem, u+sem, color='blue', alpha=0.5, lw=0)

u = ref_passiveBig[tcols][short_mask].mean(axis=0)
sem = ref_passiveBig[tcols][short_mask].sem(axis=0)
ax[1, 0].plot(tcols, u, linestyle='--', color='blue', lw=2, label='passive, big pupil')
ax[1, 0].fill_between(tcols, u-sem, u+sem, color='lightblue', alpha=0.5, lw=0)

u = ref_passiveSmall[tcols][short_mask].mean(axis=0)
sem = ref_passiveSmall[tcols][short_mask].sem(axis=0)
ax[1, 0].plot(tcols, u, linestyle='dotted', color='purple', lw=2, label='passive, small pupil')
ax[1, 0].fill_between(tcols, u-sem, u+sem, color='orchid', alpha=0.5, lw=0)

ax[1, 0].legend(frameon=False, fontsize=8)
ax[1, 0].set_xlabel('Time from sound onset', fontsize=8)
ax[1, 0].set_ylabel('Normalized response', fontsize=8)
ax[1, 0].set_title('Reference')

u = ref_active[tcols][long_mask].mean(axis=0)
sem = ref_active[tcols][long_mask].sem(axis=0)
ax[1, 1].plot(tcols, u, color='blue', lw=2, label='active')
ax[1, 1].fill_between(tcols, u-sem, u+sem, color='blue', alpha=0.5, lw=0)

u = ref_passiveBig[tcols][long_mask].mean(axis=0)
sem = ref_passiveBig[tcols][long_mask].sem(axis=0)
ax[1, 1].plot(tcols, u, linestyle='--', color='blue', lw=2, label='passive, big pupil')
ax[1, 1].fill_between(tcols, u-sem, u+sem, color='lightblue', alpha=0.5, lw=0)

u = ref_passiveSmall[tcols][long_mask].mean(axis=0)
sem = ref_passiveSmall[tcols][long_mask].sem(axis=0)
ax[1, 1].plot(tcols, u, linestyle='dotted', color='purple', lw=2, label='passive, small pupil')
ax[1, 1].fill_between(tcols, u-sem, u+sem, color='orchid', alpha=0.5, lw=0)

ax[1, 1].legend(frameon=False, fontsize=8)
ax[1, 1].set_xlabel('Time from sound onset', fontsize=8)
ax[1, 1].set_ylabel('Normalized response', fontsize=8)
ax[1, 1].set_title('Reference')

f.tight_layout()

f.savefig(fn)

plt.show()