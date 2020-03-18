"""
Population psth in the style of Atiani and David 2014.
Plot for big / small comparison and big / active comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fig_fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/test.svg'

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

mask = ref_active['sig'] | tar_active['sig'] # keep all cells with sig response for either REF or TAR

#mask = mask & (tar_active['cm_latency'] < 17)
mask = mask & (tar_active['cm_latency'] >= 21)
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

f.savefig(fig_fn)

plt.show()