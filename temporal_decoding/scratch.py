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

lda = False
fix_decoder = False
train_factor = 2

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
epochs = epochs + ['TARGET', 'REFERENCE', 'PreStimSilence']
epochs = ['TARGET', 'REFERENCE', 'PreStimSilence']
d_active = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
d_passSmall = rec['resp'].extract_epochs(epochs, mask=rec['ps_mask'])
d_passbig = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])

# remove lick periods from all dictionaries of spike counts
# (STIM epochs contain prestim silence)
nbins = int(1.1 * fs)
for k in d_passive.keys():
    if 'Silence' not in k:
        spontbins = d_passive['PreStimSilence'].shape[-1]
        startbin = 0
        d_passSmall[k] = d_passSmall[k][:, :, startbin:(spontbins+nbins)]
        d_passbig[k] = d_passbig[k][:, :, startbin:(spontbins+nbins)]
        d_active[k] = d_active[k][:, :, startbin:(spontbins+nbins)]
    else:
        pass

# define decoding axis on random subset of data, then project the remaining trials 
# define a different decoding axis for each time bin

# active data
nrefa = d_active['REFERENCE'].shape[0]
train_refa = np.random.choice(np.arange(0, nrefa), int(nrefa / train_factor))
test_refa = np.array(list(set(np.arange(0, nrefa)) - set(train_refa)))
ntara = d_active['TARGET'].shape[0]
train_tara = np.random.choice(np.arange(0, ntara), int(ntara / train_factor)) 
test_tara = np.array(list(set(np.arange(0, ntara)) - set(train_tara)))

# small passive data
nrefp = d_passSmall['REFERENCE'].shape[0]
train_refp = np.random.choice(np.arange(0, nrefp), int(nrefp / train_factor))
test_refp = np.array(list(set(np.arange(0, nrefp)) - set(train_refp)))
ntarp = d_passSmall['TARGET'].shape[0]
train_tarp = np.random.choice(np.arange(0, ntarp), int(ntarp / train_factor)) 
test_tarp = np.array(list(set(np.arange(0, ntarp)) - set(train_tarp)))

# passive big pupil data
nrefpb = d_passbig['REFERENCE'].shape[0]
train_refpb = np.random.choice(np.arange(0, nrefpb), int(nrefpb / train_factor))
test_refpb = np.array(list(set(np.arange(0, nrefpb)) - set(train_refpb)))
ntarpb = d_passbig['TARGET'].shape[0]
train_tarpb = np.random.choice(np.arange(0, ntarpb), int(ntarpb / train_factor)) 
test_tarpb = np.array(list(set(np.arange(0, ntarpb)) - set(train_tarpb)))

timebins = d_active['TARGET'].shape[-1]
spontend = d_active['PreStimSilence'].shape[-1] - startbin

dws = 7
dwe = 11
if lda:
    da = di.get_LDA_axis(d_active['TARGET'][train_tara, :, dws:dwe].mean(axis=-1, keepdims=True), d_active['REFERENCE'][train_refa, :, dws:dwe].mean(axis=-1, keepdims=True))
    dp = di.get_LDA_axis(d_passSmall['TARGET'][train_tarp, :, dws:dwe].mean(axis=-1, keepdims=True), d_passSmall['REFERENCE'][train_refp, :, dws:dwe].mean(axis=-1, keepdims=True))
    dpb = di.get_LDA_axis(d_passbig['TARGET'][train_tarpb, :, dws:dwe].mean(axis=-1, keepdims=True), d_passbig['REFERENCE'][train_refpb, :, dws:dwe].mean(axis=-1, keepdims=True))
else:
    da = di.get_null_axis(d_active['TARGET'][train_tara, :, dws:dwe].mean(axis=-1, keepdims=True), d_active['REFERENCE'][train_refa, :, dws:dwe].mean(axis=-1, keepdims=True))
    dp = di.get_null_axis(d_passSmall['TARGET'][train_tarp, :, dws:dwe].mean(axis=-1, keepdims=True), d_passSmall['REFERENCE'][train_refp, :, dws:dwe].mean(axis=-1, keepdims=True))
    dpb = di.get_null_axis(d_passbig['TARGET'][train_tarpb, :, dws:dwe].mean(axis=-1, keepdims=True), d_passbig['REFERENCE'][train_refpb, :, dws:dwe].mean(axis=-1, keepdims=True))
if fix_decoder:
    dp = da
    dpb = da

tara = np.zeros((timebins, len(test_tara)))
refa = np.zeros((timebins, len(test_refa)))
tarp = np.zeros((timebins, len(test_tarp)))
refp = np.zeros((timebins, len(test_refp)))
tarpb = np.zeros((timebins, len(test_tarpb)))
refpb = np.zeros((timebins, len(test_refpb)))
for j in range(timebins):
    tara[j, :] = np.matmul(d_active['TARGET'][test_tara, :, j], da).squeeze()
    refa[j, :] = np.matmul(d_active['REFERENCE'][test_refa, :, j], da).squeeze()
    tarp[j, :] = np.matmul(d_passSmall['TARGET'][test_tarp, :, j], dp).squeeze()
    refp[j, :] = np.matmul(d_passSmall['REFERENCE'][test_refp, :, j], dp).squeeze()
    tarpb[j, :] = np.matmul(d_passbig['TARGET'][test_tarpb, :, j], dpb).squeeze()
    refpb[j, :] = np.matmul(d_passbig['REFERENCE'][test_refpb, :, j], dpb).squeeze()

f, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 4))

ax[0].plot(tara.mean(axis=-1), color='red', lw=2, label='Target')
sem = tara.std(axis=-1) / np.sqrt(tara.shape[-1])
ax[0].fill_between(range(0, tara.shape[0]), tara.mean(axis=-1) - sem, tara.mean(axis=-1)+sem, alpha=0.5, color=ax[0].get_lines()[-1].get_color(), lw=0)
ax[0].plot(refa.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refa.std(axis=-1) / np.sqrt(refa.shape[-1])
ax[0].fill_between(range(0, refa.shape[0]), refa.mean(axis=-1) - sem, refa.mean(axis=-1)+sem, alpha=0.5, color=ax[0].get_lines()[-1].get_color(), lw=0)
ax[0].axhline(0, color='k', linestyle='--')
ax[0].axvline(spontend+int(0.2 * fs), label='Early Win', color='grey')
ax[0].axvline(spontend, label='onset', linestyle='--', color='k')

ax[0].set_ylabel('Projected response', fontsize=8)
ax[0].set_xlabel('Time bins', fontsize=8)
ax[0].set_title('Active', fontsize=8)

ax[0].legend(frameon=False, fontsize=8)

ax[1].plot(tarpb.mean(axis=-1), color='red', lw=2, label='Target')
sem = tarpb.std(axis=-1) / np.sqrt(tarpb.shape[-1])
ax[1].fill_between(range(0, tarpb.shape[0]), tarpb.mean(axis=-1) - sem, tarpb.mean(axis=-1)+sem, alpha=0.5, color=ax[1].get_lines()[-1].get_color(), lw=0)
ax[1].plot(refpb.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refpb.std(axis=-1) / np.sqrt(refpb.shape[-1])
ax[1].fill_between(range(0, refpb.shape[0]), refpb.mean(axis=-1) - sem, refpb.mean(axis=-1)+sem, alpha=0.5, color=ax[1].get_lines()[-1].get_color(), lw=0)
ax[1].axhline(0, color='k', linestyle='--')
ax[1].axvline(spontend+int(0.2 * fs), label='Early Win', color='grey')
ax[1].axvline(spontend, label='onset', linestyle='--', color='k')

ax[1].set_ylabel('Projected response', fontsize=8)
ax[1].set_xlabel('Time bins', fontsize=8)
ax[1].set_title('Passive, big pupil', fontsize=8)


ax[2].plot(tarp.mean(axis=-1), color='red', lw=2, label='Target')
sem = tarp.std(axis=-1) / np.sqrt(tarp.shape[-1])
ax[2].fill_between(range(0, tarp.shape[0]), tarp.mean(axis=-1) - sem, tarp.mean(axis=-1)+sem, alpha=0.5, color=ax[2].get_lines()[-1].get_color(), lw=0)
ax[2].plot(refp.mean(axis=-1), color='blue', lw=2, label='Reference')
sem = refp.std(axis=-1) / np.sqrt(refp.shape[-1])
ax[2].fill_between(range(0, refp.shape[0]), refp.mean(axis=-1) - sem, refp.mean(axis=-1)+sem, alpha=0.5, color=ax[2].get_lines()[-1].get_color(), lw=0)
ax[2].axhline(0, color='k', linestyle='--')
ax[2].axvline(spontend+int(0.2 * fs), label='Early Win', color='grey')
ax[2].axvline(spontend, label='onset', linestyle='--', color='k')

ax[2].set_ylabel('Projected response', fontsize=8)
ax[2].set_xlabel('Time bins', fontsize=8)
ax[2].set_title('Passive, small pupil', fontsize=8)


f.tight_layout()


plt.show()