import nems.db as nd
force_rerun = True
batch = 307

sites = ['TAR010c', 'BRT026c', 'BRT033b', 'BRT034f', 'BRT036b', 'BRT037b',
    'BRT039c', 'bbl102d', 'AMT018a', 'AMT020a', 'AMT022c', 'AMT026a']

modelnames = ['refTar_dprime_act_pass', 
              'refTar_dprime_act_pass_sim12', 
              'refTar_dprime_act_pass_sim1',
              'refTar_dprime_act_pass_sim2',
              'refTar_dprime_act_bigPass', 
              'refTar_dprime_act_bigPass_sim12', 
              'refTar_dprime_act_bigPass_sim1',
              'refTar_dprime_act_bigPass_sim2',
              'refTar_dprime_act_smallPass', 
              'refTar_dprime_act_smallPass_sim12', 
              'refTar_dprime_act_smallPass_sim1',
              'refTar_dprime_act_smallPass_sim2',
              'refTar_dprime_bigPass_smallPass', 
              'refTar_dprime_bigPass_smallPass_sim12', 
              'refTar_dprime_bigPass_smallPass_sim1',
              'refTar_dprime_bigPass_smallPass_sim2']
modelnames += [m + '_LDA' for m in modelnames]
modelnames += [m + '_binPreLick' for m in modelnames]
# only run the binPreLick for now (much faster, and based on the noise correlation result,
# seems like the critical period for where the sound is discriminated and decision is made)

# CRH 2/18 queue pca models only for binPreLick models
modelnames = [m for m in modelnames if 'binPreLick' in m]
mpc2 = [m.replace('refTar_', 'refTar_pca2_') for m in modelnames]
mpc3 = [m.replace('refTar_', 'refTar_pca3_') for m in modelnames]
mpc4 = [m.replace('refTar_', 'refTar_pca4_') for m in modelnames]
modelnames = mpc2 + mpc3 + mpc4

# only keep simulations
modelnames = [m for m in modelnames if '_sim' in m]
modelnames = [m.replace('_sim', '_uMatch_sim') for m in modelnames]

# queue regression models
modelnames = [m for m in modelnames if '_sim' not in m]
pr_models = [m.replace('dprime_', 'dprime_pr_') for m in modelnames]
br_models = [m.replace('dprime_', 'dprime_br_') for m in modelnames]
both_models = [m.replace('dprime_', 'dprime_pr_br_') for m in modelnames]
modelnames = pr_models + br_models + both_models


script = '/auto/users/hellerc/code/projects/Cosyne2020_poster/dprime_analysis/cache_dprime.py'
python_path = '/auto/users/hellerc/anaconda3/envs/crh_nems/bin/python'

nd.enqueue_models(celllist=sites,
                  batch=batch,
                  modellist=modelnames,
                  executable_path=python_path,
                  script_path=script,
                  user='hellerc',
                  force_rerun=force_rerun,
                  reserve_gb=4)