
# code copied out of cache_dprime script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
import nems_lbhb.baphy as nb
from sklearn.decomposition import PCA
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import charlieTools.preprocessing as preproc
import charlieTools.discrimination_tools as di
import charlieTools.plotting as cplt
import charlieTools.simulate_data as sim
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/dec_cv_example.svg'

site = 'TAR010c'
nPCs = 4
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
dm = rec['resp'].extract_epochs(epochs, mask=rec['miss_mask'])
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

dm['TARGET'] = dm['TARGET'][:, :, :nbins].mean(axis=-1, keepdims=True)

#pca_dict = preproc.get_balanced_rep_counts(d1, d2)
#d1, var = preproc.pca_reduce_dimensionality(d1, npcs=4, dict_fit=pca_dict)
#d2, var = preproc.pca_reduce_dimensionality(d2, npcs=4, dict_fit=pca_dict)
# also, project all pca_dict data into this space for defining state-independent decoding axis
# this way it has the same dimensions, and the data are balanced
#all_dict, var = preproc.pca_reduce_dimensionality(pca_dict, npcs=4, dict_fit=pca_dict)

# project into decoding axis vs. maximum correlated variability axis space
# for active and passive
dec_axis = di.get_LDA_axis(d1['TARGET'].squeeze(), d1['REFERENCE'].squeeze())
pca = PCA(n_components=1)
noise_axis = pca.fit(d2['TARGET'].squeeze()).components_
#diff = np.cov(d1['TARGET'].squeeze().T) - np.cov(d2['TARGET'].squeeze().T) 
#eig, evec = np.linalg.eig(diff)
#noise_axis = evec[:, 0]
#noise_axis = (d1['TARGET'].mean(axis=0) - dm['TARGET'].mean(axis=0)).squeeze()
noise_on_dec = (np.dot(noise_axis, dec_axis[:, np.newaxis])) * dec_axis
orth_ax = noise_axis - noise_on_dec
orth_ax = (orth_ax / np.linalg.norm(orth_ax)).squeeze()

rot_mat = np.concatenate((orth_ax[:, np.newaxis], dec_axis[:, np.newaxis]), axis=-1)[:, ::-1]

f, ax = plt.subplots(1, 2)

t_active = np.matmul(d1['TARGET'].squeeze(), rot_mat)
t_passive = np.matmul(d2['TARGET'].squeeze(), rot_mat)
r_active = np.matmul(d1['REFERENCE'].squeeze(), rot_mat)
r_passive = np.matmul(d2['REFERENCE'].squeeze(), rot_mat)
miss = np.matmul(dm['TARGET'].squeeze(), rot_mat)

t_active_el = cplt.compute_ellipse(t_active[:, 0], t_active[:, 1])
t_passive_el = cplt.compute_ellipse(t_passive[:, 0], t_passive[:, 1])
r_active_el = cplt.compute_ellipse(r_active[:, 0], r_active[:, 1])
r_passive_el = cplt.compute_ellipse(r_passive[:, 0], r_passive[:, 1])
miss_el = cplt.compute_ellipse(miss[:, 0], miss[:, 1])

ax[0].scatter(t_passive[:, 0], t_passive[:, 1], color='red', alpha=0.3, s=15, edgecolor='none')
ax[0].scatter(r_passive[:, 0], r_passive[:, 1], color='blue', alpha=0.3, s=15, edgecolor='none')
ax[0].plot(t_passive_el[0, :], t_passive_el[1, :], color='red')
ax[0].plot(r_passive_el[0, :], r_passive_el[1, :], color='blue')

ax[1].scatter(t_active[:, 0], t_active[:, 1], color='red', alpha=0.3, s=15, edgecolor='none')
ax[1].scatter(r_active[:, 0], r_active[:, 1], color='blue', alpha=0.3, s=15, edgecolor='none')
ax[1].scatter(miss[:, 0], miss[:, 1], color='k', alpha=0.3, s=15, edgecolor='none')
ax[1].plot(t_active_el[0, :], t_active_el[1, :], color='red')
ax[1].plot(r_active_el[0, :], r_active_el[1, :], color='blue')
ax[1].plot(miss_el[0, :], miss_el[1, :], color='k')

ax_min = np.min(ax[0].get_xlim() + ax[0].get_ylim())
ax_max = np.max(ax[0].get_xlim() + ax[0].get_ylim())

ax[0].set_xlim((ax_min, ax_max))
ax[0].set_ylim((ax_min, ax_max))
ax[1].set_xlim((ax_min, ax_max))
ax[1].set_ylim((ax_min, ax_max))

#ax[0].set_xlim((-5, 7))
#ax[0].set_ylim((4, 14))
#ax[1].set_xlim((-5, 7))
#ax[1].set_ylim((4, 14))

ax[0].set_xlabel('Decoding Axis')
ax[1].set_xlabel('Decoding Axis')
ax[0].set_ylabel('Correlated Variability Axis')
ax[1].set_ylabel('Correlated Variability Axis')
ax[0].set_title("Passive")
ax[1].set_title("Active")

ax[0].set_aspect(cplt.get_square_asp(ax[0]))
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

f.tight_layout()

f.savefig(fn)

plt.show()