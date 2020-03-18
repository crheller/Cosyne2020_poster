# project data onto decoding axis and look at tar/ref separation over time
# in the trial (up through first 200ms, when no licking happening)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
import nems_lbhb.baphy as nb
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import charlieTools.discrimination_tools as di
import charlieTools.preprocessing as preproc
import ptd_utils as pu
import charlieTools.plotting as cplt
import seaborn as sns
import charlieTools.simulate_data as sim
import os
import nems
import nems.db as nd
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/example_decoding_trial.svg'

lda = False
fix_decoder = False

# load example site
site = 'TAR010c'
batch = 307
fs = 20
rawid = pu.which_rawids(site)
ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
        'rawid': rawid}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])
rec = rec.and_mask(['PostStimSilence'], invert=True)

rec = rec.apply_mask(reset_epochs=True)

rec = preproc.create_ptd_masks(rec)
epochs = ['TARGET', 'REFERENCE', 'PreStimSilence']
d_active = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
d_passSmall = rec['resp'].extract_epochs(epochs, mask=rec['ps_mask'])
d_passbig = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])

# remove lick periods from all dictionaries of spike counts
# (STIM epochs contain prestim silence)
nbins = int(1.1 * fs)
for k in d_passSmall.keys():
    if 'Silence' not in k:
        spontbins = d_passSmall['PreStimSilence'].shape[-1]
        startbin = 0
        d_passSmall[k] = d_passSmall[k][:, :, startbin:(spontbins+nbins)]
        d_passbig[k] = d_passbig[k][:, :, startbin:(spontbins+nbins)]
        d_active[k] = d_active[k][:, :, startbin:(spontbins+nbins)]
    else:
        pass

timebins = d_active['TARGET'].shape[-1]
spontend = d_active['PreStimSilence'].shape[-1] - startbin

dws = 7
dwe = 11
if lda:
    da = di.get_LDA_axis(d_active['TARGET'][:, :, dws:dwe].mean(axis=-1, keepdims=True), d_active['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
    dp = di.get_LDA_axis(d_passSmall['TARGET'][:, :, dws:dwe].mean(axis=-1, keepdims=True), d_passSmall['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
    dpb = di.get_LDA_axis(d_passbig['TARGET'][train_tarpb, :, dws:dwe].mean(axis=-1, keepdims=True), d_passbig['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
else:
    da = di.get_null_axis(d_active['TARGET'][:, :, dws:dwe].mean(axis=-1, keepdims=True), d_active['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
    dp = di.get_null_axis(d_passSmall['TARGET'][:, :, dws:dwe].mean(axis=-1, keepdims=True), d_passSmall['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
    dpb = di.get_null_axis(d_passbig['TARGET'][:, :, dws:dwe].mean(axis=-1, keepdims=True), d_passbig['REFERENCE'][:, :, dws:dwe].mean(axis=-1, keepdims=True))
if fix_decoder:
    dp = da
    dpb = da

tara = np.zeros((timebins, d_active['TARGET'].shape[0]))
refa = np.zeros((timebins, d_active['REFERENCE'].shape[0]))
tarp = np.zeros((timebins, d_passSmall['TARGET'].shape[0]))
refp = np.zeros((timebins, d_passSmall['REFERENCE'].shape[0]))
tarpb = np.zeros((timebins, d_passbig['TARGET'].shape[0]))
refpb = np.zeros((timebins, d_passbig['REFERENCE'].shape[0]))
for j in range(timebins):
    tara[j, :] = np.matmul(d_active['TARGET'][:, :, j], da).squeeze()
    refa[j, :] = np.matmul(d_active['REFERENCE'][:, :, j], da).squeeze()
    tarp[j, :] = np.matmul(d_passSmall['TARGET'][:, :, j], dp).squeeze()
    refp[j, :] = np.matmul(d_passSmall['REFERENCE'][:, :, j], dp).squeeze()
    tarpb[j, :] = np.matmul(d_passbig['TARGET'][:, :, j], dpb).squeeze()
    refpb[j, :] = np.matmul(d_passbig['REFERENCE'][:, :, j], dpb).squeeze()

f, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

ax[0].plot(tara.mean(axis=-1), color='red', lw=2, label='Target')
sem = tara.std(axis=-1) / np.sqrt(tara.shape[-1])
ax[0].fill_between(range(0, tara.shape[0]), tara.mean(axis=-1) - sem, tara.mean(axis=-1)+sem, alpha=0.5, color=ax[0].get_lines()[-1].get_color(), lw=0)
ax[0].plot(refa.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refa.std(axis=-1) / np.sqrt(refa.shape[-1])
ax[0].fill_between(range(0, refa.shape[0]), refa.mean(axis=-1) - sem, refa.mean(axis=-1)+sem, alpha=0.5, color=ax[0].get_lines()[-1].get_color(), lw=0)
ax[0].axhline(0, color='k', linestyle='--')
yspan = np.arange(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
ax[0].fill_betweenx(yspan, 
                    dws * np.ones(len(yspan)), dwe * np.ones(len(yspan)), 
                    color='lightgrey', zorder=-1)

ax[0].set_xticks(np.arange(1, tarp.shape[0], 2))
ax[0].set_xticklabels(np.round(np.arange(-6 * (1 / fs), 15 * (1/ fs), 2 / fs), 2))
ax[0].set_ylabel('Projected response')
ax[0].set_xlabel('Time (s)')
ax[0].set_title('Active', fontsize=8)

ax[0].legend(frameon=False, fontsize=8)

ax[1].plot(tarpb.mean(axis=-1), color='red', lw=2, label='Target')
sem = tarpb.std(axis=-1) / np.sqrt(tarpb.shape[-1])
ax[1].fill_between(range(0, tarpb.shape[0]), tarpb.mean(axis=-1) - sem, tarpb.mean(axis=-1)+sem, alpha=0.5, color=ax[1].get_lines()[-1].get_color(), lw=0)
ax[1].plot(refpb.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refpb.std(axis=-1) / np.sqrt(refpb.shape[-1])
ax[1].fill_between(range(0, refpb.shape[0]), refpb.mean(axis=-1) - sem, refpb.mean(axis=-1)+sem, alpha=0.5, color=ax[1].get_lines()[-1].get_color(), lw=0)
ax[1].axhline(0, color='k', linestyle='--')
yspan = np.arange(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
ax[1].fill_betweenx(yspan, 
                    dws * np.ones(len(yspan)), dwe * np.ones(len(yspan)), 
                    color='lightgrey', zorder=-1)

ax[1].set_xticks(np.arange(1, tarp.shape[0], 2))
ax[1].set_xticklabels(np.round(np.arange(-6 * (1 / fs), 15 * (1/ fs), 2 / fs), 2))
ax[1].set_xlabel('Time (s)')
ax[1].set_title('Passive, big pupil', fontsize=8)


ax[2].plot(tarp.mean(axis=-1), color='red', lw=2, label='Target')
sem = tarp.std(axis=-1) / np.sqrt(tarp.shape[-1])
ax[2].fill_between(range(0, tarp.shape[0]), tarp.mean(axis=-1) - sem, tarp.mean(axis=-1)+sem, alpha=0.5, color=ax[2].get_lines()[-1].get_color(), lw=0)
ax[2].plot(refp.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refp.std(axis=-1) / np.sqrt(refp.shape[-1])
ax[2].fill_between(range(0, refp.shape[0]), refp.mean(axis=-1) - sem, refp.mean(axis=-1)+sem, alpha=0.5, color=ax[2].get_lines()[-1].get_color(), lw=0)
ax[2].axhline(0, color='k', linestyle='--')
yspan = np.arange(ax[0].get_ylim()[0], ax[0].get_ylim()[1])
ax[2].fill_betweenx(yspan, 
                    dws * np.ones(len(yspan)), dwe * np.ones(len(yspan)), 
                    color='lightgrey', zorder=-1)

ax[2].set_xticks(np.arange(1, tarp.shape[0], 2))
ax[2].set_xticklabels(np.round(np.arange(-6 * (1 / fs), 15 * (1/ fs), 2 / fs), 2))
ax[2].set_xlabel('Time (s)')
ax[2].set_title('Passive, small pupil', fontsize=8)

f.tight_layout()

f.savefig(fn)


plt.show()