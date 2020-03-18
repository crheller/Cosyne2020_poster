"""
Lump all REFs and TARs together and compute dprime between them for active / passive.

Just compute in the first 200 ms, as this is before the early window (licking)

Do for 3 cases:
    decoder trained on active
    decoder trained on passive
    decoder trained over all data
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
import charlieTools.simulate_data as sim
import os
import nems
import nems.db as nd
import logging

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# first system argument is the cellid
site = sys.argv[1]  # very first (index 0) is the script to be run
# second systems argument is the batch
batch = sys.argv[2]
# third system argument in the modelname
modelname = sys.argv[3]

# parse model options
simulate = '_sim' in modelname
regress_pupil = '_pr' in modelname
regress_behavior = '_br' in modelname
active = '_act' in modelname
passive = '_pass' in modelname
big_passive = '_bigPass' in modelname
small_passive = '_smallPass' in modelname
miss_trial = '_missTrial' in modelname
lda = '_LDA' in modelname
bin_pre_lick = '_binPreLick' in modelname
pca = '_pca' in modelname
uMatch = '_uMatch' in modelname

nPCs = None
if pca:
    nPCs = int(modelname.split('_pca')[1].split('_')[0])

if (active & passive):
    d1_string = 'active'
    d2_string = 'passive'
if (active & big_passive):
    d1_string = 'active'
    d2_string = 'big_passive'
if (active & small_passive):
    d1_string = 'active'
    d2_string = 'small_passive'
if (small_passive & big_passive):
    d1_string = 'big_passive'
    d2_string = 'small_passive'
if (miss_trial & big_passive):
    d1_string = 'miss_trial'
    d2_string = 'big_passive'
if (miss_trial & active):
    d1_string = 'active'
    d2_string = 'miss_trial'

log.info('Analyzing site: {0} with options: \n \
                simulate: {1} \n \
                pupil regressed: {2} \n \
                behavior regressed: {3} \n \
                using LDA axis: {4} \n \
                comparing {5} vs. {6} \n \
                reduce dim with pca {7}, ndims: {8} \n \
                uMatch: {9}'.format(site, simulate,
                regress_pupil, regress_behavior, lda, d1_string, d2_string, pca, nPCs, uMatch))

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

if pca & (nPCs > len(rec['resp'].chans)):
    raise ValueError("Too few neurons at this site \n \
                        {0} cells, requested {1} PCs".format(len(rec['resp'].chans), nPCs))

# perform state correction, if necessary
if regress_pupil & regress_behavior:
    rec = preproc.regress_state(rec, 
                                state_sigs=['pupil', 'behavior'],
                                regress=['pupil', 'behavior'])
elif regress_pupil:
    rec = preproc.regress_state(rec, 
                                state_sigs=['pupil'],
                                regress=['pupil'])
elif regress_behavior:
    rec = preproc.regress_state(rec, 
                                state_sigs=['behavior'],
                                regress=['behavior'])
else:
    pass

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
p_epochs = np.unique([t for t in rp.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
pb_epochs = np.unique([t for t in rpb.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
ps_epochs = np.unique([t for t in rps.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()
all_epochs = np.unique([t for t in rec.epochs.name if ('TAR_' in t) | ('STIM_' in t)]).tolist()

finish_script = True
if small_passive:
    epochs = [t for t in all_epochs if (t in a_epochs) & (t in p_epochs) & (t in pb_epochs) & (t in ps_epochs)]
    if (epochs == []) | ((rps.epochs.name=='TARGET').sum()==0):
        log.info("Not sufficient small pupil data. Break")
        finish_script = False  # to end script gracefully.
else:
    # find epochs present in all states. Small pupil  may have smaller number of epochs (less data)
    # so we do that one separately for those cases.
    epochs = [t for t in all_epochs if (t in a_epochs) & (t in p_epochs) & (t in pb_epochs)]

log.info('site {0} has {1} unique epochs'.format(site, len(epochs)))
if finish_script:
    # still include general target (i.e. lump all targets/refs together to see how compares)
    epochs += ['REFERENCE', 'TARGET']

    if active & passive:
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['p_mask'])
        all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['p_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    if active & big_passive:
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])
        all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['pb_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    if active & small_passive:
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['ps_mask'])
        all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['ps_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    if big_passive & small_passive:
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['ps_mask'])
        all_mask = rec['mask']._modified_copy(rec['pb_mask']._data | rec['ps_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    # miss trials may have limited target n, so be sure for this to only use
    # targets that were missed at least 3 times for miss
    if big_passive & miss_trial:
        rtemp = rec.copy()
        rtemp['mask'] = rtemp['miss_mask']
        rtemp = rtemp.apply_mask(reset_epochs=True)
        epochs = [e for e in epochs if ((rtemp.epochs.name==e).sum() > 2)]
        if len(epochs) == 0:
            log.info('no targets were presented a sufficient number of times duirng miss trials')
            sys.exit()
        log.info('updating epochs for miss trial comparison. Now using {} epochs'.format(len(epochs)-2))
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['pb_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['miss_mask'])
        all_mask = rec['mask']._modified_copy(rec['pb_mask']._data | rec['miss_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    if active & miss_trial:
        rtemp = rec.copy()
        rtemp['mask'] = rtemp['miss_mask']
        rtemp = rtemp.apply_mask(reset_epochs=True)
        epochs = [e for e in epochs if (rtemp.epochs.name==e).sum() > 2]
        if len(epochs) == 0:
            log.info('no targets were presented a sufficient number of times duirng miss trials')
            sys.exit()
        log.info('updating epochs for miss trial comparison. Now using {} epochs'.format(len(epochs)-2))
        d1 = rec['resp'].extract_epochs(epochs, mask=rec['a_mask'])
        d2 = rec['resp'].extract_epochs(epochs, mask=rec['miss_mask'])
        all_mask = rec['mask']._modified_copy(rec['a_mask']._data | rec['miss_mask']._data)
        all_dict = rec['resp'].extract_epochs(epochs, mask=all_mask)

    # make sure enough reps in both dictionaries, all epochs
    epochs = list(d1.keys())
    for e in epochs:
        if (d1[e].shape[0] < 3) | (d2[e].shape[0] < 3):
            del d1[e]
            del d2[e]
            del all_dict[e]
            log.info("removing epoch {} with insufficient rep count".format(e))

    if len(d1.keys()) <= 1:
        sys.exit("No epochs with sufficient data")

    # Keep all time bins
    if bin_pre_lick:
        nbins = int(fs * 0.2)
        # extract only the first 200ms
        for k in d1.keys():
            all_dict[k] = all_dict[k][:, :, :nbins].mean(axis=-1, keepdims=True)
            d1[k] = d1[k][:, :, :nbins].mean(axis=-1, keepdims=True)
            d2[k] = d2[k][:, :, :nbins].mean(axis=-1, keepdims=True)

    var = np.nan
    if pca:
        # balance number of reps between states of each stimulus for the pca_dict
        # so that you don't bias the PCA space towards the state with more stimuli
        pca_dict = preproc.get_balanced_rep_counts(d1, d2)
        d1, var = preproc.pca_reduce_dimensionality(d1, npcs=nPCs, dict_fit=pca_dict)
        d2, var = preproc.pca_reduce_dimensionality(d2, npcs=nPCs, dict_fit=pca_dict)
        # also, project all pca_dict data into this space for defining state-independent decoding axis
        # this way it has the same dimensions, and the data are balanced
        all_dict, var = preproc.pca_reduce_dimensionality(pca_dict, npcs=nPCs, dict_fit=pca_dict)
        

    # simulate data, if specified
    if '_sim12' in modelname:
        log.info('Keep first and second order stats')
        d1 = sim.generate_simulated_trials(d1, 
                                        r2=None,
                                        keep_stats=[1, 2], N=5000)
        d2 = sim.generate_simulated_trials(d2, 
                                        r2=None,
                                        keep_stats=[1, 2], N=5000)

    elif '_sim1' in modelname:
        if uMatch:
            log.info('Keep first order stats of {0}, impose second order stats from {1}'.format(d1_string, d2_string))
            d1 = sim.generate_simulated_trials(d1, 
                                            r2=all_dict,
                                            keep_stats=[1], N=5000)
            d2 = sim.generate_simulated_trials(d2, 
                                            r2=all_dict,
                                            keep_stats=[1], N=5000)
        else:
            log.info('Keep first order stats of {0}, impose second order stats from {1}'.format(d1_string, d2_string))
            d1 = sim.generate_simulated_trials(d1, 
                                            r2=d2,
                                            keep_stats=[1], N=5000)
            d2 = sim.generate_simulated_trials(d2, 
                                            r2=None,
                                            keep_stats=[1], N=5000)

    elif '_sim2' in modelname:
        if uMatch:
            log.info('Keep second order stats of {0}, impose first order stats from {1}'.format(d1_string, d2_string))
            d1 = sim.generate_simulated_trials(d1, 
                                            r2=all_dict,
                                            keep_stats=[2], N=5000)
            d2 = sim.generate_simulated_trials(d2, 
                                            r2=all_dict,
                                            keep_stats=[2], N=5000)
        else:
            log.info('Keep second order stats of {0}, impose first order stats from {1}'.format(d1_string, d2_string))
            d1 = sim.generate_simulated_trials(d1, 
                                            r2=d2,
                                            keep_stats=[2], N=5000)
            d2 = sim.generate_simulated_trials(d2, 
                                            r2=None,
                                            keep_stats=[2], N=5000)



    # compute dprime between in the two conditions
    log.info('computing dprime with state-specific decoding axis (ssa)')
    r1_ssa = di.compute_dprime_from_dicts(d1, LDA=lda, norm=True, equal_tbin=True)
    r2_ssa = di.compute_dprime_from_dicts(d2, LDA=lda, norm=True, equal_tbin=True)
    r1_ssa['state'] = d1_string
    r2_ssa['state'] = d2_string
    r1_ssa['axis'] = 'ssa'
    r2_ssa['axis'] = 'ssa'


    log.info('computing dprime with state-independent decoding axis (sia)')
    r1_sia = di.compute_dprime_from_dicts(d1, all_dict, LDA=lda, norm=True, equal_tbin=True)
    r2_sia = di.compute_dprime_from_dicts(d2, all_dict, LDA=lda, norm=True, equal_tbin=True)
    r1_sia['state'] = d1_string
    r2_sia['state'] = d2_string
    r1_sia['axis'] = 'sia'
    r2_sia['axis'] = 'sia'

    log.info('computing dprime with opposite state-specific decoding axis (soa)')
    r1_soa = di.compute_dprime_from_dicts(d1, d2, LDA=lda, norm=True, equal_tbin=True)
    r2_soa = di.compute_dprime_from_dicts(d2, d1, LDA=lda, norm=True, equal_tbin=True)
    r1_soa['state'] = d1_string
    r2_soa['state'] = d2_string
    r1_soa['axis'] = 'soa'
    r2_soa['axis'] = 'soa'

    df = pd.concat([r1_ssa, r2_ssa, r1_sia, r2_sia, r1_soa, r2_soa])

    df['pca_reduction_var_explained'] = var

    spath = '/auto/users/hellerc/code/projects/Cosyne2020_poster/dprime_analysis/results/{0}/'.format(site)
    save_fn = spath + '{0}.csv'.format(modelname)

    if os.path.isdir(spath):
        pass
    else:
        os.mkdir(spath, mode=0o777)

    df.to_csv(save_fn)

    log.info("Saved results for all conditions to {0}".format(save_fn))

else:
    pass
    # not sufficient data. Quit script. Don't run analysis.

if queueid:
    nd.update_job_complete(queueid)