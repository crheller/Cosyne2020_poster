'''
results loading functions
'''

import os
import pandas as pd
import numpy as np

def load_dprime(modelname, results_path=None):
    if results_path is None:
        path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/dprime_analysis/results/'
    else:
        path = results_path

    sites = os.listdir(path)

    dfs = []
    for s in sites:
        try:
            df = pd.read_csv(path+s+'/'+modelname+'.csv', index_col=0)
            df['site'] = s
            dfs.append(df)
        except:
            print("no results found for site: {0}, model: {1}".format(s, modelname))
    
    df = pd.concat(dfs)

    
    for c in df.columns:
        if 'pc1' in c:
            df[c] = [abs(float(x.replace('[', '').replace(']', ''))) for x in df[c]]

    df['colFromIndex'] = df.index
    df = df.sort_values(by=['site', 'colFromIndex'])
    df = df.drop(columns='colFromIndex')
    
    return df


def load_rsc(modelname, path=None):
    if path is None:
        path = '/auto/users/hellerc/code/projects/Cosyne2020_poster/noise_correlations/results/'

    files = os.listdir(path)
    sites = np.unique([f.split('_')[-1] for f in files])

    dfs = []
    for site in sites:
        dfs.append(pd.read_csv(path+modelname+'_'+site, index_col=0))

    df = pd.concat(dfs)

    return df    