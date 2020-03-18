from nems_lbhb.baphy_experiment import BAPHYExperiment
import sys
sys.path.append('/auto/users/hellerc/code/projects/PTD/')
import ptd_utils as pu
import pandas as pd

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

fs = 20
batch = 307
results = pd.DataFrame(index=sites, columns=['DI', 'dprime', 'targets', 'all_DI', 'all_dprime'])
for site in sites:
    rawid = tuple(pu.which_rawids(site))

    manager = BAPHYExperiment(batch=batch, siteid=site, rawid=rawid)

    options = {'rasterfs': fs, 'batch': batch, 'siteid': site, 'keep_following_incorrect_trial': False, 'keep_early_trials': False}
    correct_trials = manager.get_behavior_performance(**options)
    
    options = {'rasterfs': fs, 'batch': batch, 'siteid': site, 'keep_following_incorrect_trial': True, 'keep_early_trials': True}
    all_trials = manager.get_behavior_performance(**options)

    targets = [k for k in all_trials['dprime'].keys()]
    DI = [v for v in correct_trials['DI'].values()]
    dprime = [v for v in correct_trials['dprime'].values()]
    all_DI = [v for v in all_trials['DI'].values()]
    all_dprime = [v for v in all_trials['dprime'].values()]

    results.loc[site] = [DI, dprime, targets, all_DI, all_dprime]

results.to_pickle('/auto/users/hellerc/code/projects/Cosyne2020_poster/behavior/results.pickle')