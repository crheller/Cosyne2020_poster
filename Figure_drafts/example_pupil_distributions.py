"""
Two example pupil distributions
"""

import charlieTools.plotting as cplt
import charlieTools.preprocessing as preproc
import matplotlib.pyplot as plt
import numpy as np
from nems.recording import Recording
import nems_lbhb.baphy as nb
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu

site = 'AMT026a'
site2 = 'BRT033b'

f, ax = plt.subplots(1, 2, figsize=(8, 4))

fs = 20
batch = 307
rawid = pu.which_rawids(site)
ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
        'rawid': rawid}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])

rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
rec = rec.apply_mask(reset_epochs=True)

rec = preproc.create_ptd_masks(rec)

active = rec['pupil']._data[rec['a_mask']._data]
big_passive = rec['pupil']._data[rec['pb_mask']._data]
small_passive = rec['pupil']._data[rec['ps_mask']._data]

bins = np.arange(rec['pupil']._data.min(), rec['pupil']._data.max(),  
                        (rec['pupil']._data.max() - rec['pupil']._data.min()) / 20)
ax[1].hist(active, color='red', edgecolor='red', lw=3, bins=bins, density=True, histtype='step', fill=False, label='active')
ax[1].hist(big_passive, color='blue', edgecolor='blue', lw=3, bins=bins, density=True, histtype='step', fill=False, label='big passive')
ax[1].hist(small_passive, color='lightblue', edgecolor='orchid', lw=3, bins=bins, density=True, histtype='step', fill=False, label='small passive')
ax[1].set_title(site)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

site = site2

rawid = pu.which_rawids(site)
ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
        'rawid': rawid}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])

rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
rec = rec.apply_mask(reset_epochs=True)

rec = preproc.create_ptd_masks(rec)

active = rec['pupil']._data[rec['a_mask']._data]
big_passive = rec['pupil']._data[rec['pb_mask']._data]
small_passive = rec['pupil']._data[rec['ps_mask']._data]

bins = np.arange(rec['pupil']._data.min(), rec['pupil']._data.max(),  
                        (rec['pupil']._data.max() - rec['pupil']._data.min()) / 20)
ax[0].hist(active, color='red', edgecolor='red', lw=3, bins=bins, density=True, histtype='step', fill=False, label='active')
ax[0].hist(big_passive, color='blue', edgecolor='blue', lw=3, bins=bins, density=True, histtype='step', fill=False, label='big passive')
ax[0].hist(small_passive, color='orchid', edgecolor='orchid', lw=3, bins=bins, density=True, histtype='step', fill=False, label='small passive')
ax[0].legend()
ax[0].set_title(site)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

f.tight_layout()

plt.show()