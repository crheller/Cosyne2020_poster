import nems_lbhb.strf.strf as strf
import matplotlib.pyplot as plt
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import nems_lbhb.baphy as nb
from nems.recording import Recording
import charlieTools.preprocessing as preproc
from nems_lbhb.strf.torc_subfunctions import strfplot
import pandas as pd
import nems.db as nd

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']
batch = 307
cells = [c for c in nd.get_batch_cells(batch).cellid if c[:7] in sites]
fs = 1000

save_fn = '/auto/users/hellerc/results/Cosyne2020_poster/single_cell_analyses/strfs/{}_cache.pickle'

df = pd.DataFrame(index=cells, columns=['BF', 'SNR', 'STRF', 'StimParms'])
dfa = pd.DataFrame(index=cells, columns=['BF', 'SNR', 'STRF', 'StimParms'])
dfp = pd.DataFrame(index=cells, columns=['BF', 'SNR', 'STRF', 'StimParms'])
dfpb = pd.DataFrame(index=cells, columns=['BF', 'SNR', 'STRF', 'StimParms'])
dfps = pd.DataFrame(index=cells, columns=['BF','SNR', 'STRF', 'StimParms'])
for s in sites:
    
    rawid = pu.which_rawids(s)
    ops = {'batch': batch, 'pupil': 1, 'rasterfs': fs, 'siteid': s, 'stim': 0,
            'rawid': rawid}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
 
    rec = preproc.create_ptd_masks(rec)

    ra = rec.copy()
    ra['mask'] = ra['a_mask']
    ra = ra.apply_mask(reset_epochs=True)
    
    rp = rec.copy()
    rp['mask'] = rp['p_mask']
    rp = rp.apply_mask(reset_epochs=True)
    
    rpb = rec.copy()
    rpb['mask'] = rpb['pb_mask']
    rpb = rpb.apply_mask(reset_epochs=True)
    
    rps = rec.copy()
    rps['mask'] = rps['ps_mask']
    rps = rps.apply_mask(reset_epochs=True)

    r = rec.and_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT'])
    r = r.apply_mask(reset_epochs=True)

    cellids = [c for c in cells if c in ra['resp'].chans]

    for c in cellids:
        print('Cellid: {}'.format(c))

        strf_a = strf.tor_tuning(c, rec=ra, plot=False)
        strf_p = strf.tor_tuning(c, rec=rp, plot=False)
        strf_pb = strf.tor_tuning(c, rec=rpb, plot=False)
        strf_ps = strf.tor_tuning(c, rec=rps, plot=False)
        strf_all = strf.tor_tuning(c, rec=r, plot=False)


        dfa.loc[c] = [strf_a.Best_Frequency_Hz, strf_a.Signal_to_Noise, [strf_a.STRF], [strf_a.StimParams]]
        dfp.loc[c] = [strf_p.Best_Frequency_Hz, strf_p.Signal_to_Noise, [strf_p.STRF], [strf_p.StimParams]]
        dfpb.loc[c] = [strf_pb.Best_Frequency_Hz, strf_pb.Signal_to_Noise, [strf_pb.STRF], [strf_pb.StimParams]]
        dfps.loc[c] = [strf_ps.Best_Frequency_Hz, strf_ps.Signal_to_Noise, [strf_ps.STRF], [strf_ps.StimParams]]
        df.loc[c] = [strf_all.Best_Frequency_Hz, strf_all.Signal_to_Noise, [strf_all.STRF], [strf_all.StimParams]]

dfa.to_pickle(save_fn.format('active'))
dfp.to_pickle(save_fn.format('passive'))
dfpb.to_pickle(save_fn.format('bigPassive'))
dfps.to_pickle(save_fn.format('smallPassive'))
df.to_pickle(save_fn.format('all'))
