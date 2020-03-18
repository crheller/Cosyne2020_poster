"""
Cache a df for TAR psths and REF psths for all cells, for Active, Passive, Big Pupil Passive
Add columns for site, BF, target freq(s), signficance, latency to peak
"""

import nems.db as nd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nems.recording import Recording
import nems_lbhb.baphy as nb
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import charlieTools.preprocessing as preproc
import charlieTools.psth_tools as pt

batch = 307
fs = 20
sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

cache_path = '/auto/users/hellerc/results/Cosyne2020_poster/single_cell_analyses/psths/'

ref_dfa = []
tar_dfa = []
ref_dfp = []
tar_dfp = []
ref_dfpb = []
tar_dfpb = []
ref_dfps = []
tar_dfps = []

ref_dfa_raw = []
tar_dfa_raw = []
ref_dfp_raw = []
tar_dfp_raw = []
ref_dfpb_raw = []
tar_dfpb_raw = []
ref_dfps_raw = []
tar_dfps_raw = []

strf_df = pd.read_pickle('/auto/users/hellerc/results/Cosyne2020_poster/single_cell_analyses/strfs/all_cache.pickle')

for i, site in enumerate(sites):
    print("{0} / {1}, {2}".format(i+1, len(sites), site))
    rawid = pu.which_rawids(site)
    ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': site, 'stim': 0,
            'rawid': rawid}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    rec['resp'] = rec['resp'].rasterize()
    rec = rec.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'PASSIVE_EXPERIMENT'])
    rec = rec.apply_mask(reset_epochs=True)

    rec = preproc.create_ptd_masks(rec)

    # get per neuron psth
    nbins = rec['resp'].extract_epoch('REFERENCE').shape[-1]
    tar_a = rec['resp'].extract_epoch('TARGET', mask=rec['a_mask'])[:, :, :nbins].mean(axis=0)
    ref_a = rec['resp'].extract_epoch('REFERENCE', mask=rec['a_mask'])[:, :, :nbins].mean(axis=0)
    tar_p = rec['resp'].extract_epoch('TARGET', mask=rec['p_mask'])[:, :, :nbins].mean(axis=0)
    ref_p = rec['resp'].extract_epoch('REFERENCE', mask=rec['p_mask'])[:, :, :nbins].mean(axis=0)
    tar_pb = rec['resp'].extract_epoch('TARGET', mask=rec['pb_mask'])[:, :, :nbins].mean(axis=0)
    ref_pb = rec['resp'].extract_epoch('REFERENCE', mask=rec['pb_mask'])[:, :, :nbins].mean(axis=0)
    try:
        tar_ps = rec['resp'].extract_epoch('TARGET', mask=rec['ps_mask'])[:, :, :nbins].mean(axis=0)
        ref_ps = rec['resp'].extract_epoch('REFERENCE', mask=rec['ps_mask'])[:, :, :nbins].mean(axis=0)
        if tar_ps.shape[0] > 5 & (ref_ps.shape[0] > 5):
            small_pupil = True
        else:
            small_pupil = False
            print('Not enough small pupil data for site {}'.format(site))
    except:
        small_pupil = False
        print('No small pupil data for site {}'.format(site))

    # save raw psth for each cell
    sp_bins = rec['resp'].extract_epoch('PreStimSilence').shape[-1]
    tbins = np.arange(-sp_bins, nbins-sp_bins)
    tdfa_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_a)
    tdfp_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_p)
    tdfpb_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_pb)
    
    rdfa_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_a)
    rdfp_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_p)
    rdfpb_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_pb)

    if small_pupil:
        tdfps_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_ps)
        rdfps_raw = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_ps)
        tar_dfps_raw.append(tdfps_raw)
        ref_dfps_raw.append(rdfps_raw)

    tar_dfa_raw.append(tdfa_raw)
    tar_dfp_raw.append(tdfp_raw)
    tar_dfpb_raw.append(tdfpb_raw)
    ref_dfa_raw.append(rdfa_raw)
    ref_dfp_raw.append(rdfp_raw)
    ref_dfpb_raw.append(rdfpb_raw)

    # normalize psth per neuron based on max response (across active only) and spont rate (individually)

    # targets
    sp_passive = tar_p[:, :sp_bins].mean(axis=-1, keepdims=True)
    sp_active = tar_a[:, :sp_bins].mean(axis=-1, keepdims=True)
    sp_pb = tar_pb[:, :sp_bins].mean(axis=-1, keepdims=True)
    if small_pupil:
        sp_ps = tar_ps[:, :sp_bins].mean(axis=-1, keepdims=True)

        sp_norm_resp = np.concatenate((tar_a[:, sp_bins:] - sp_active, 
                                    tar_p[:, sp_bins:] - sp_passive,
                                    tar_pb[:, sp_bins:] - sp_pb,
                                    tar_ps[:, sp_bins:] - sp_ps), axis=-1)
    else:
        sp_norm_resp = np.concatenate((tar_a[:, sp_bins:] - sp_active, 
                                    tar_p[:, sp_bins:] - sp_passive,
                                    tar_pb[:, sp_bins:] - sp_pb), axis=-1)

    max_resp_inds = abs(tar_a[:, sp_bins:] - sp_active).argmax(axis=-1)
    tar_max_resp = np.diag(np.take(tar_a[:, sp_bins:] - sp_active, max_resp_inds, axis=-1))[:, np.newaxis]

    # Normalize responses
    tar_a = (tar_a - sp_active) / tar_max_resp
    tar_p = (tar_p - sp_passive) / tar_max_resp
    tar_pb = (tar_pb - sp_pb) / tar_max_resp
    if small_pupil:
        tar_ps = (tar_ps - sp_ps) / tar_max_resp

    # latency to peak (relative to trial onset)
    tar_peak_lat = np.argmax(tar_a[:, sp_bins:], axis=-1) + sp_bins
    
    # latency to cm
    cm = np.cumsum(abs(tar_a[:, sp_bins:]), axis=-1) / np.sum(abs(tar_a[:, sp_bins:]), axis=-1)[:, np.newaxis]
    cm = np.concatenate((np.zeros((cm.shape[0], 1)), cm), axis=-1)
    cm_lat_idx = np.argwhere(np.diff(cm>(cm[:,-1] / 2)[:, np.newaxis], axis=-1))
    tar_cm_lat =  cm_lat_idx[:, 1] + sp_bins + 1

    # sign of peak
    tar_peak_sign = np.sign((tar_a - sp_active)[np.arange(0, tar_a.shape[0]), tar_peak_lat])

    # sign of cm
    tar_cm_sign = np.sign((tar_a - sp_active)[cm_lat_idx[:, 0], tar_cm_lat])

    # references
    sp_passive = ref_p[:, :sp_bins].mean(axis=-1, keepdims=True)
    sp_active = ref_a[:, :sp_bins].mean(axis=-1, keepdims=True)
    sp_pb = ref_pb[:, :sp_bins].mean(axis=-1, keepdims=True)
    if small_pupil:
        sp_ps = ref_ps[:, :sp_bins].mean(axis=-1, keepdims=True)

        sp_norm_resp = np.concatenate((ref_a[:, sp_bins:] - sp_active, 
                                    ref_p[:, sp_bins:] - sp_passive,
                                    ref_pb[:, sp_bins:] - sp_pb,
                                    ref_ps[:, sp_bins:] - sp_ps), axis=-1)
    else:
        sp_norm_resp = np.concatenate((ref_a[:, sp_bins:] - sp_active, 
                                    ref_p[:, sp_bins:] - sp_passive,
                                    ref_pb[:, sp_bins:] - sp_pb), axis=-1)

    max_resp_inds = abs(ref_a[:, sp_bins:] - sp_active).argmax(axis=-1)
    ref_max_resp = np.diag(np.take(ref_a[:, sp_bins:] - sp_active, max_resp_inds, axis=-1))[:, np.newaxis]

    # Normalize responses
    ref_a = (ref_a - sp_active) / ref_max_resp
    ref_p = (ref_p - sp_passive) / ref_max_resp
    ref_pb = (ref_pb - sp_pb) / ref_max_resp
    if small_pupil:
        ref_ps = (ref_ps - sp_ps) / ref_max_resp

    # latency to peak (relative to trial onset)
    ref_peak_lat = np.argmax(ref_a[:, sp_bins:], axis=-1) + sp_bins
    
    # latency to cm
    cm = (np.cumsum(abs(ref_a[:, sp_bins:]), axis=-1) / np.sum(abs(ref_a[:, sp_bins:]), axis=-1)[:, np.newaxis])
    cm = np.concatenate((np.zeros((cm.shape[0], 1)), cm), axis=-1)
    cm_lat_idx = np.argwhere(np.diff(cm>(cm[:,-1] / 2)[:, np.newaxis], axis=-1))
    ref_cm_lat =  cm_lat_idx[:, 1] + sp_bins + 1

    # sign of peak
    ref_peak_sign = np.sign((ref_a - sp_active)[range(0, ref_a.shape[0]), ref_peak_lat])

    # sign of cm
    ref_cm_sign = np.sign((ref_a - sp_active)[cm_lat_idx[:, 0], ref_cm_lat])


    # determine significance of responses for all conditions (i.e. did the cell have 
    # a significant auditory response, ignoring behavioral state)
    # (using jackknifed t-test in charlieTools.statistics)
    tar_st = rec['resp'].extract_epoch('TARGET')[:, :, :nbins]
    ref_st = rec['resp'].extract_epoch('REFERENCE')[:, :, :nbins]
    spont = rec['resp'].extract_epoch('PreStimSilence')
    sigtar = pt.get_psth_pvals(tar_st, spont, cellids=rec['resp'].chans)
    sigref = pt.get_psth_pvals(ref_st, spont, cellids=rec['resp'].chans)

    # convert sig df's to single vector (True if more than two evoked bins have p <= 0.01)
    #sigtar =  (pd.isnull(sigtar.loc[:, sp_bins:(nbins-sp_bins)]) == False).sum(axis=1) > 2
    #sigref = (pd.isnull(sigref.loc[:, sp_bins:(nbins-sp_bins)]) == False).sum(axis=1) > 2

    # bonferonni correction (22 time bins / alpha = 0.05) means need p < ~0.002
    # bonferonni correction for alpha = 0.01 mean need p < .0004 (so need at least 3 pvals of 0.001 or less)
    sigtar = (sigtar.loc[:, sp_bins:(nbins-sp_bins)]<=0.001).sum(axis=1)>2
    sigref = (sigref.loc[:, sp_bins:(nbins-sp_bins)]<=0.001).sum(axis=1)>2

    # get target frequency(s) at this site
    tar_epochs = [e for e in rec.epochs.name.unique() if ('STIM_' in e) & ('TORC_' not in e)]

    # get unit STRF BF and SNR
    strfdf = strf_df.loc[rec['resp'].chans]
    bf = strfdf['BF'].values
    snr = strfdf['SNR'].values

    # build result dataframe(s) for this site

    # target results
    tdfa = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_a)
    tdfp = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_p)
    tdfpb = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_pb)
    if small_pupil:
        tdfps = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=tar_ps)
        tdfps['site'] = site
        tdfps['BF'] = bf
        tdfps['SNR'] = snr
        tdfps['sig'] = sigtar
        tdfps['peak_latency'] = tar_peak_lat
        tdfps['cm_latency'] = tar_cm_lat
        tdfps['peak_sign'] = tar_peak_sign
        tdfps['cm_sign'] = tar_cm_sign
        tdfps['targets'] = [tar_epochs] * tdfa.shape[0]

        tar_dfps.append(tdfps)

    tdfa['site'] = site
    tdfa['BF'] = bf
    tdfa['SNR'] = snr
    tdfa['sig'] = sigtar
    tdfa['peak_latency'] = tar_peak_lat
    tdfa['cm_latency'] = tar_cm_lat
    tdfa['peak_sign'] = tar_peak_sign
    tdfa['cm_sign'] = tar_cm_sign
    tdfa['targets'] = [tar_epochs] * tdfa.shape[0]

    tar_dfa.append(tdfa)

    tdfp['site'] = site
    tdfp['BF'] = bf
    tdfp['SNR'] = snr
    tdfp['sig'] = sigtar
    tdfp['peak_latency'] = tar_peak_lat
    tdfp['cm_latency'] = tar_cm_lat
    tdfp['peak_sign'] = tar_peak_sign
    tdfp['cm_sign'] = tar_cm_sign
    tdfp['targets'] = [tar_epochs] * tdfa.shape[0]

    tar_dfp.append(tdfp)

    tdfpb['site'] = site
    tdfpb['BF'] = bf
    tdfpb['SNR'] = snr
    tdfpb['sig'] = sigtar
    tdfpb['peak_latency'] = tar_peak_lat
    tdfpb['cm_latency'] = tar_cm_lat
    tdfpb['peak_sign'] = tar_peak_sign
    tdfpb['cm_sign'] = tar_cm_sign
    tdfpb['targets'] = [tar_epochs] * tdfa.shape[0]

    tar_dfpb.append(tdfpb)


    # REFERENCE results
    rdfa = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_a)
    rdfp = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_p)
    rdfpb = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_pb)
    if small_pupil:
        rdfps = pd.DataFrame(index=rec['resp'].chans, columns=tbins, data=ref_ps)
        rdfps['site'] = site
        rdfps['BF'] = bf
        rdfps['SNR'] = snr
        rdfps['sig'] = sigref
        rdfps['peak_latency'] = ref_peak_lat
        rdfps['cm_latency'] = ref_cm_lat
        rdfps['peak_sign'] = ref_peak_sign
        rdfps['cm_sign'] = ref_cm_sign
        rdfps['targets'] = [tar_epochs] * tdfa.shape[0]

        ref_dfps.append(rdfps)

    rdfa['site'] = site
    rdfa['BF'] = bf
    rdfa['SNR'] = snr
    rdfa['sig'] = sigref
    rdfa['peak_latency'] = ref_peak_lat
    rdfa['cm_latency'] = ref_cm_lat
    rdfa['peak_sign'] = ref_peak_sign
    rdfa['cm_sign'] = ref_cm_sign
    rdfa['targets'] = [tar_epochs] * tdfa.shape[0]

    ref_dfa.append(rdfa)

    rdfp['site'] = site
    rdfp['BF'] = bf
    rdfp['SNR'] = snr
    rdfp['sig'] = sigref
    rdfp['peak_latency'] = ref_peak_lat
    rdfp['cm_latency'] = ref_cm_lat
    rdfp['peak_sign'] = ref_peak_sign
    rdfp['cm_sign'] = ref_cm_sign
    rdfp['targets'] = [tar_epochs] * tdfa.shape[0]

    ref_dfp.append(rdfp)

    rdfpb['site'] = site
    rdfpb['BF'] = bf
    rdfpb['SNR'] = snr
    rdfpb['sig'] = sigref
    rdfpb['peak_latency'] = ref_peak_lat
    rdfpb['cm_latency'] = ref_cm_lat
    rdfpb['peak_sign'] = ref_peak_sign
    rdfpb['cm_sign'] = ref_cm_sign
    rdfpb['targets'] = [tar_epochs] * tdfa.shape[0]

    ref_dfpb.append(rdfpb)


ref_dfa = pd.concat(ref_dfa)
ref_dfp = pd.concat(ref_dfp)
ref_dfpb = pd.concat(ref_dfpb)
ref_dfps = pd.concat(ref_dfps)

tar_dfa = pd.concat(tar_dfa)
tar_dfp = pd.concat(tar_dfp)
tar_dfpb = pd.concat(tar_dfpb)
tar_dfps = pd.concat(tar_dfps)

ref_dfa_raw = pd.concat(ref_dfa_raw)
ref_dfp_raw = pd.concat(ref_dfp_raw)
ref_dfpb_raw = pd.concat(ref_dfpb_raw)
ref_dfps_raw = pd.concat(ref_dfps_raw)

tar_dfa_raw = pd.concat(tar_dfa_raw)
tar_dfp_raw = pd.concat(tar_dfp_raw)
tar_dfpb_raw = pd.concat(tar_dfpb_raw)
tar_dfps_raw = pd.concat(tar_dfps_raw)

# cache results
ref_dfa.to_pickle(cache_path + 'ref_active_norm.pickle')
ref_dfp.to_pickle(cache_path + 'ref_passive_norm.pickle')
ref_dfpb.to_pickle(cache_path + 'ref_bigPassive_norm.pickle')
ref_dfps.to_pickle(cache_path + 'ref_smallPassive_norm.pickle')

tar_dfa.to_pickle(cache_path + 'tar_active_norm.pickle')
tar_dfp.to_pickle(cache_path + 'tar_passive_norm.pickle')
tar_dfpb.to_pickle(cache_path + 'tar_bigPassive_norm.pickle')
tar_dfps.to_pickle(cache_path + 'tar_smallPassive_norm.pickle')

ref_dfa_raw.to_pickle(cache_path + 'ref_active_raw.pickle')
ref_dfp_raw.to_pickle(cache_path + 'ref_passive_raw.pickle')
ref_dfpb_raw.to_pickle(cache_path + 'ref_bigPassive_raw.pickle')
ref_dfps_raw.to_pickle(cache_path + 'ref_smallPassive_raw.pickle')

tar_dfa_raw.to_pickle(cache_path + 'tar_active_raw.pickle')
tar_dfp_raw.to_pickle(cache_path + 'tar_passive_raw.pickle')
tar_dfpb_raw.to_pickle(cache_path + 'tar_bigPassive_raw.pickle')
tar_dfps_raw.to_pickle(cache_path + 'tar_smallPassive_raw.pickle')
