import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
import nems_lbhb.baphy as nb
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import charlieTools.preprocessing as preproc

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

f, axes = plt.subplots(3, 4, figsize=(12, 8))

for site, ax in zip(sites, axes.flatten()):
    batch = 307
    fs = 20
    rawid = pu.which_rawids(site)
    ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
            'rawid': rawid}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])
    rec = rec.and_mask(['REFERENCE', 'TARGET']) # to remove long post stim silence in active

    rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    rec = rec.apply_mask(reset_epochs=True)

    rec = preproc.create_ptd_masks(rec)

    active = rec['pupil']._data[rec['a_mask']._data]
    big_passive = rec['pupil']._data[rec['pb_mask']._data]
    small_passive = rec['pupil']._data[rec['ps_mask']._data]

    bins = np.arange(rec['pupil']._data.min(), rec['pupil']._data.max(),  
                            (rec['pupil']._data.max() - rec['pupil']._data.min()) / 20)
    ax.hist(active, color='red', edgecolor='red', lw=3, bins=bins, density=False, histtype='step', fill=False, label='active')
    ax.hist(big_passive, color='blue', edgecolor='blue', lw=3, bins=bins, density=False, histtype='step', fill=False, label='big passive')
    ax.hist(small_passive, color='lightblue', edgecolor='lightblue', lw=3, bins=bins, density=False, histtype='step', fill=False, label='small passive')
    cutoff = active.mean() - (2 * active.std())
    ax.axvline(cutoff, linestyle='--', color='k', lw=2)

    if site == sites[0]:
        ax.legend(frameon=False, fontsize=8)


    ax.set_title(site, fontsize=8)
    

f.tight_layout()

plt.show()