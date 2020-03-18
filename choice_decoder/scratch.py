"""
Project hit / miss trials onto delta noise correlation axis. How well can decode choice on this axis?
In the spirit of Ni et. al 2018
"""

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

site = 'TAR010c'
ts = 0.1
te = 0.2
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

rec_act = rec.copy()
rec_act['mask'] = rec['a_mask']
rec_act = rec_act.apply_mask(reset_epochs=True)

rec_passBig = rec.copy()
rec_passBig['mask'] = rec['pb_mask']
rec_passBig = rec_passBig.apply_mask(reset_epochs=True)

rec_miss = rec.copy()
rec_miss['mask'] = rec['miss_mask']
rec_miss = rec_miss.apply_mask(reset_epochs=True)

all_tar_epochs = np.unique([e for e in rec.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
t_act_epochs = np.unique([e for e in rec_act.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
t_pb_epochs = np.unique([e for e in rec_passBig.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()
t_miss_epochs = np.unique([e for e in rec_miss.epochs.name if (('STIM_' in e) | ('TAR_' in e)) & ('TORC' not in e)]).tolist()

# make sure targets exists in all state epochs at least 3 times
tar_epochs = [t for t in all_tar_epochs if ((rec_act.epochs.name==t).sum() > 2) & \
                                        ((rec_passBig.epochs.name==t).sum() > 2) & \
                                        ((rec_miss.epochs.name==t).sum() > 2)]

print('number of targets: {}'.format(len(tar_epochs)))

act = rec_act['resp'].extract_epochs(tar_epochs)
pb = rec_passBig['resp'].extract_epochs(tar_epochs)
miss = rec_miss['resp'].extract_epochs(tar_epochs)

# extract time window
s = int(ts * fs)
e = int(te * fs)
for k in tar_epochs:
        act[k] = act[k][:, :, s:e].mean(axis=-1)
        pb[k] = pb[k][:, :, s:e].mean(axis=-1)
        miss[k] = miss[k][:, :, s:e].mean(axis=-1)

# get the axis on which correlated variability changes 
# between passive and active
for i, k in enumerate(tar_epochs):
        a_resp = act[k]
        print('{0} active reps of target: {1}'.format(a_resp.shape[0], k))
        a_mean = a_resp.mean(axis=0)
        a_std = a_resp.std(axis=0)
        a_residual = a_resp - a_mean
        a_std[a_std==0] = 1
        a_residual /= a_std

        pb_resp = pb[k]
        print('{0} passive big pupil reps of target: {1}'.format(pb_resp.shape[0], k))
        pb_mean = pb_resp.mean(axis=0)
        pb_std = pb_resp.std(axis=0)
        pb_residual = pb_resp - pb_mean
        pb_std[pb_std==0] = 1
        pb_residual /= pb_std

        miss_resp = miss[k]
        print('{0} miss reps of target: {1}'.format(miss_resp.shape[0], k))
        miss_mean = miss_resp.mean(axis=0)
        miss_std = miss_resp.std(axis=0)
        miss_residual = miss_resp - miss_mean
        miss_std[miss_std==0] = 1
        miss_residual /= miss_std

        if i == 0:
            a_residuals = a_residual
            pb_residuals = pb_residual
            a_R = a_resp
            pb_R = pb_resp
            miss_R = miss_resp

        else:
            a_residuals = np.concatenate((a_residuals, a_residual), axis=0)
            pb_residuals = np.concatenate((pb_residuals, pb_residual), axis=0)
            a_R = np.concatenate((a_R, a_resp), axis=0)
            pb_R = np.concatenate((pb_R, pb_resp), axis=0) 
            miss_R = np.concatenate((miss_R, miss_resp), axis=0) 

diff = np.cov(a_R.T) - np.cov(pb_R.T)
eig, evec = np.linalg.eig(diff)
axis = evec[:, 0]

# project passive, miss, and hit onto this axis
hit_proj = np.matmul(a_R, axis)
miss_proj = np.matmul(miss_R, axis)
pass_proj = np.matmul(pb_R, axis)


f, ax = plt.subplots(1, 1)

ax.hist(hit_proj)
ax.hist(pass_proj)
ax.hist(miss_proj)

plt.show()