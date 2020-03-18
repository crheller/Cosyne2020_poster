"""
Plot ref / tar projections on decoding axis over the entire expriment. Compare separation 
to pupil size / task status. Presumeably should see both sharp change with task and slow
change with pupil.
"""
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
rec = rec.and_mask(['REFERENCE', 'TARGET'])
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

rec = rec.apply_mask(reset_epochs=True)

rec = preproc.create_ptd_masks(rec)
epochs = ['TARGET', 'REFERENCE']
d_active = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
d_passive = rec['resp'].extract_epochs(epochs, mask=rec['p_mask'])
all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['p_mask']._data)
all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

trial_len = all_dict['TARGET'].shape[-1]
nbins = int(fs * 0.2)
# extract only the first 200ms
for k in d_active.keys():
    all_dict[k] = all_dict[k][:, :, :nbins].mean(axis=-1, keepdims=True)
    d_active[k] = d_active[k][:, :, :nbins].mean(axis=-1, keepdims=True)
    d_passive[k] = d_passive[k][:, :, :nbins].mean(axis=-1, keepdims=True)

# compute decoding axis
axis = di.get_LDA_axis(d_active['REFERENCE'].squeeze(), d_active['TARGET'].squeeze())

# convert back to trial time and project each single trial onto the decoding axis
# to do this, pad dicts with nans so that can use "replace" epochs function
for e in epochs:
    nreps = all_dict[e].shape[0]
    ncells = all_dict[e].shape[1]
    all_dict[e] = np.concatenate((all_dict[e], np.nan * np.ones((nreps, ncells, trial_len-1))), axis=-1)

# replace epochs
r_new = rec.copy()
r_new['mask'] = all_mask
r_new = r_new.apply_mask(reset_epochs=True)
resp = r_new['resp'].replace_epochs(all_dict)

# project data
tar_mask = r_new.and_mask('TARGET')['mask']
target = np.matmul(resp._data.T, axis)
target[~tar_mask._data.squeeze()] = np.nan
ref_mask = r_new.and_mask('REFERENCE')['mask']
reference = np.matmul(resp._data.T, axis)
reference[~ref_mask._data.squeeze()] = np.nan

f, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

ax[0].plot(target, '.', color='r', label='Target')
ax[0].plot(reference, '.', color='blue', label='Reference')
ax[0].set_ylabel('Response Projection')
ax[0].legend(frameon=False)

ax[1].plot(r_new['pupil']._data.squeeze() / r_new['pupil']._data.max(), color='purple', label='pupil size')
ax[1].plot(r_new['pupil'].epoch_to_signal('ACTIVE_EXPERIMENT')._data.T, color='grey', label='task-status')
ax[1].set_xticks(np.arange(0, r_new['pupil'].shape[-1], 1200))
ax[1].set_xticklabels(np.arange(0, r_new['pupil'].shape[-1] / 20, 60))
ax[1].set_xlabel('Experiment Time (s)')
ax[1].legend(frameon=False)

f.tight_layout()

plt.show()