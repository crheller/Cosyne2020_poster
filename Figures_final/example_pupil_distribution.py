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
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/example_pupil.svg'

site = 'AMT026a'

fs = 20
batch = 307
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

active = rec['pupil']._data[rec['a_mask']._data]
big_passive = rec['pupil']._data[rec['pb_mask']._data]
small_passive = rec['pupil']._data[rec['ps_mask']._data]


f, ax = plt.subplots(1, 1, figsize=(8, 4))
bins = np.arange(rec['pupil']._data.min(), rec['pupil']._data.max(),  
                        (rec['pupil']._data.max() - rec['pupil']._data.min()) / 20)
ax.hist(active, color='red', edgecolor='k', lw=2, bins=bins, density=False, histtype='step', fill=False, label='Active')
ax.hist(big_passive, color='blue', edgecolor='grey', lw=2, bins=bins, density=False, histtype='step', fill=False, label='Passive, pupil matched')
ax.hist(small_passive, color='lightblue', edgecolor='grey', linestyle='--', lw=2, bins=bins, density=False, histtype='step', fill=False, label='Passive, small pupil')
ax.set_title(site)
ax.legend(frameon=False)
ax.set_xlabel("Pupil Size")
ax.set_ylabel("Frequency")
ax.set_aspect(cplt.get_square_asp(ax))

f.tight_layout()

f.savefig(fn)

plt.show()