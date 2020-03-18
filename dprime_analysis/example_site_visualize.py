'''
4 pc, 2nd order simulation
ref vs. ref
           dprime A    dprime PB
site                       
AMT018a  4.134714  4.695714
AMT020a  2.255072  2.887744
AMT022c  2.750236  3.394905
AMT026a  1.850283  2.482486
BRT026c  3.529248  4.052788
BRT033b  3.108554  3.779062
BRT034f  1.775622  2.031646
BRT036b  3.059310  3.143183
BRT037b  4.666897  4.442588
BRT039c  1.454208  1.624729
TAR010c  1.762157  2.345086
bbl102d  1.940715  1.628190

4 pc, 2nd order simulation
ref vs. tar
           dprime A    dprime PB
site                       
AMT018a  5.660368  5.391342
AMT020a  4.257130  4.826725
AMT022c  3.489268  3.372526
AMT026a  1.391350  1.607571
BRT026c  2.246433  1.935641
BRT033b  6.103709  4.664837
BRT034f  1.385945  1.209233
BRT036b  3.661383  3.222740
BRT037b  5.139270  4.213422
BRT039c  0.980407  0.825662
TAR010c  4.556149  3.474619
bbl102d  2.419572  2.037359
'''
# code copied out of cache_dprime script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
import nems_lbhb.baphy as nb
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import charlieTools.preprocessing as preproc
import charlieTools.discrimination_tools as di
import charlieTools.plotting as cplt
import charlieTools.simulate_data as sim

simulate = True
site = 'TAR010c'
nPCs = 2
batch = 307
fs = 20
rawid = pu.which_rawids(site)
ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
        'rawid': rawid}
uri = nb.baphy_load_recording_uri(**ops)
rec = Recording.load(uri)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])

rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
rec = rec.apply_mask(reset_epochs=True)

# create four state masks
rec = preproc.create_ptd_masks(rec)

# create the appropriate dictionaries of responses
# don't *think* it's so important to balanced reps for 
# this analysis since it's parametric (as opposed to the
# discrimination calc which depends greatly on being balanced)

# get list of unique targets present in both a/p. This is for 
# variable snr and/or on/off BF so they aren't lumped together for dprime
ra = rec.copy()
ra['mask'] = rec['a_mask']
ra = ra.apply_mask(reset_epochs=True)
rp = rec.copy()
rp['mask'] = rec['p_mask']
rp = rp.apply_mask(reset_epochs=True)
rpb = rec.copy()
rpb['mask'] = rec['pb_mask']
rpb = rpb.apply_mask(reset_epochs=True)
rps = rec.copy()
rps['mask'] = rec['ps_mask']
rps = rps.apply_mask(reset_epochs=True)
a_epochs = np.unique([t for t in ra.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
pb_epochs = np.unique([t for t in rpb.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
all_epochs = np.unique([t for t in rec.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
epochs = [t for t in all_epochs if (t in a_epochs) & (t in pb_epochs)]

epochs += ['REFERENCE', 'TARGET']

d1 = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
d2 = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])
all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['pb_mask']._data)
all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

epochs = list(d1.keys())
for e in epochs:
    if (d1[e].shape[0] < 3) | (d2[e].shape[0] < 3):
        del d1[e]
        del d2[e]
        print("removing epoch {} with insufficient rep count".format(e))

nbins = int(fs * 0.2)
# extract only the first 200ms
for k in d1.keys():
    all_dict[k] = all_dict[k][:, :, :nbins].mean(axis=-1, keepdims=True)
    d1[k] = d1[k][:, :, :nbins].mean(axis=-1, keepdims=True)
    d2[k] = d2[k][:, :, :nbins].mean(axis=-1, keepdims=True)

pca_dict = preproc.get_balanced_rep_counts(d1, d2)
d1, var = preproc.pca_reduce_dimensionality(d1, npcs=nPCs, dict_fit=pca_dict)
d2, var = preproc.pca_reduce_dimensionality(d2, npcs=nPCs, dict_fit=pca_dict)
all_dict, var = preproc.pca_reduce_dimensionality(all_dict, npcs=nPCs, dict_fit=all_dict)

if simulate:
    d1 = sim.generate_simulated_trials(d1, 
                                    r2=all_dict,
                                    keep_stats=[2], N=5000)
    d2 = sim.generate_simulated_trials(d2, 
                                    r2=all_dict,
                                    keep_stats=[2], N=5000)

# compute dprime
r1_ssa = di.compute_dprime_from_dicts(d1, LDA=False, norm=True, equal_tbin=True)
r2_ssa = di.compute_dprime_from_dicts(d2, LDA=False, norm=True, equal_tbin=True)

# plot data. d1 is active, d2 is big pupil

# Don't plot REF or TARGET ID alone
plot_epochs = [e for e in d1.keys() if (e != 'TARGET') & (e != 'REFERENCE')]

# get ref / tar decoding axis for active / passive
di.get_null_axis(d1['REFERENCE'].squeeze(), d1['TARGET'].squeeze())

f, ax = plt.subplots(1, 2)

for ep in plot_epochs:
    
    active_el = cplt.compute_ellipse(d1[ep][:, 0], d1[ep][:, 1])
    passive_el = cplt.compute_ellipse(d2[ep][:, 0], d2[ep][:, 1])

    if 'TORC' in ep:
        ax[0].plot(active_el[0, :], active_el[1, :], color='navy')
        ax[1].plot(passive_el[0, :], passive_el[1, :], color='navy')
    
    else:
        ax[0].plot(active_el[0, :], active_el[1, :], color='firebrick')
        ax[1].plot(passive_el[0, :], passive_el[1, :], color='firebrick')

ax[0].axhline(0, linestyle='--', color='k')
ax[0].axvline(0, linestyle='--', color='k')

ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0, linestyle='--', color='k')

xmax = np.max([ax[0].get_xlim(), ax[1].get_xlim()], axis=1).max()
xmin = np.min([ax[0].get_xlim(), ax[1].get_xlim()], axis=1).min()

ymax = np.max([ax[0].get_ylim(), ax[1].get_ylim()], axis=0).max()
ymin = np.min([ax[0].get_ylim(), ax[1].get_ylim()], axis=0).min()

ax[0].set_xlim((xmin, xmax))
ax[0].set_ylim((ymin, ymax))
ax[1].set_xlim((xmin, xmax))
ax[1].set_ylim((ymin, ymax))

ax[0].set_title('Active')
ax[1].set_title('Big pupil')

ax[0].set_xlabel('PC1')
ax[1].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylabel('PC2')

ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

plt.show()