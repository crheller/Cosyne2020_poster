'''
Compare dprime in active vs. passive vs. passive big pupil
'''

import loading as ld
import charlieTools.plotting as cplt
import matplotlib.pyplot as plt
import numpy as np

# analysis over all prelick period (coarse binning)
# coarse target too (combine all targets)
ap_df = ld.load_dprime('refTar_dprime_act_pass_binPreLick')
# using 'TARGET not in i' assures that a different axis is used for each target, but we 
# still collapse across each target dprime for the coarse view here
filt = [i for i in ap_df.index if ('TARGET' not in i) and ('REFERENCE' in i)]
ap_df = ap_df[ap_df.index.isin(filt)]

f, ax = plt.subplots(3, 1, figsize=(5, 8))

mi = ap_df.groupby(by=['site', 'axis', 'state']).mean()['dprime'].min()
ma = ap_df.groupby(by=['site', 'axis', 'state']).mean()['dprime'].max()
# active decoding axis
y = ap_df[(ap_df['axis']=='ssa') & (ap_df['state']=='active')].groupby(by='site').mean()['dprime']
x = ap_df[(ap_df['axis']=='soa') & (ap_df['state']=='passive')].groupby(by='site').mean()['dprime']

ax[0].set_title('Active decoding axis', fontsize=8)
ax[0].scatter(x, y, color='k', edgecolor='white')
ax[0].plot([mi, ma], [mi, ma], '--', color='grey')
ax[0].set_xlabel('passive', fontsize=8)
ax[0].set_ylabel('active', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

# passive decoding axis
y = ap_df[(ap_df['axis']=='soa') & (ap_df['state']=='active')].groupby(by='site').mean()['dprime']
x = ap_df[(ap_df['axis']=='ssa') & (ap_df['state']=='passive')].groupby(by='site').mean()['dprime']

ax[1].set_title('Passive decoding axis', fontsize=8)
ax[1].scatter(x, y, color='k', edgecolor='white')
ax[1].plot([mi, ma], [mi, ma], '--', color='grey')
ax[1].set_xlabel('passive', fontsize=8)
ax[1].set_ylabel('active', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

# overall decoding axis
y = ap_df[(ap_df['axis']=='sia') & (ap_df['state']=='active')].groupby(by='site').mean()['dprime']
x = ap_df[(ap_df['axis']=='sia') & (ap_df['state']=='passive')].groupby(by='site').mean()['dprime']

ax[2].set_title('Overall decoding axis', fontsize=8)
ax[2].scatter(x, y, color='k', edgecolor='white')
ax[2].plot([mi, ma], [mi, ma], '--', color='grey')
ax[2].set_xlabel('passive', fontsize=8)
ax[2].set_ylabel('active', fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

# plot active / passive tar/ref dprime as function of trial time
# again collapse over targets (and sites)
ap_df = ld.load_dprime('refTar_dprime_act_pass')

# only keep dprime results where comparing the same time bins to each other
# and comparing a TAR to a REF
filt = [i for i in ap_df.index if ('TARGET' not in i) & ('REF' in i) & \
            ((i.split('_')[2]==i.split('_')[-1]) | (i.split('_')[1]==i.split('_')[-1]))]   

ap_df = ap_df[ap_df.index.isin(filt)]

active = ap_df[ap_df['state']=='active']
passive = ap_df[ap_df['state']=='passive']

# find range of unique time bins
time_bins = np.unique([int(i.split('_')[-1]) for i in ap_df.index]) 

aa = np.zeros(len(time_bins))
ap = np.zeros(len(time_bins))
ao = np.zeros(len(time_bins))
pa = np.zeros(len(time_bins))
pp = np.zeros(len(time_bins))
po = np.zeros(len(time_bins))

for i, t in enumerate(time_bins):
    # for active decoding axis
    idx = [i for i in active.index if str(t) == i.split('_')[-1]]
    a = active[(active['axis']=='ssa') & active.index.isin(idx)]['dprime'].mean()
    a_sem = active[(active['axis']=='ssa') & active.index.isin(idx)]['dprime'].sem()
    p = passive[(passive['axis']=='soa') & passive.index.isin(idx)]['dprime'].mean()
    p_sem = passive[(passive['axis']=='soa') & passive.index.isin(idx)]['dprime'].mean()

    aa[i] = a
    pa[i] = p
    # for passive decoding axis
    a = active[(active['axis']=='soa') & active.index.isin(idx)]['dprime'].mean()
    a_sem = active[(active['axis']=='soa') & active.index.isin(idx)]['dprime'].sem()
    p = passive[(passive['axis']=='ssa') & passive.index.isin(idx)]['dprime'].mean()
    p_sem = passive[(passive['axis']=='ssa') & passive.index.isin(idx)]['dprime'].mean()

    ap[i] = a
    pp[i] = p

    # for overall decoding axis
    a = active[(active['axis']=='sia') & active.index.isin(idx)]['dprime'].mean()
    a_sem = active[(active['axis']=='sia') & active.index.isin(idx)]['dprime'].sem()
    p = passive[(passive['axis']=='sia') & passive.index.isin(idx)]['dprime'].mean()
    p_sem = passive[(passive['axis']=='sia') & passive.index.isin(idx)]['dprime'].mean()

    ao[i] = a
    po[i] = p

f, ax = plt.subplots(3, 1, figsize=(6, 8), sharey=True, sharex=True)
time = time_bins * (1 / 20)

ax[0].set_title('Active decoding axis', fontsize=8)
ax[0].plot(time, aa, color='k', lw=2, label='active')
ax[0].plot(time, pa, color='grey', lw=2, label='passive')
ax[0].plot(time, aa - pa, color='r')
ax[0].axvline(0.2, color='orange', label='Early window')
ax[0].axvline(0.35, color='purple', label='Sound offset')
ax[0].axhline(0, linestyle='--', color='k')
ax[0].set_ylabel('dprime', fontsize=8)
ax[0].legend(fontsize=8, frameon=False)

ax[1].set_title('Passive decoding axis', fontsize=8)
ax[1].plot(time, ap, color='k', lw=2, label='active')
ax[1].plot(time, pp, color='grey', lw=2, label='passive')
ax[1].plot(time, ap - pp, color='r')
ax[1].axvline(0.2, color='orange')
ax[1].axvline(0.35, color='purple')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_ylabel('dprime', fontsize=8)

ax[2].set_title('Overall decoding axis', fontsize=8)
ax[2].plot(time, ao, color='k', lw=2, label='active')
ax[2].plot(time, po, color='grey', lw=2, label='passive')
ax[2].plot(time, ao - po, color='r')
ax[2].axvline(0.2, color='orange')
ax[2].axvline(0.35, color='purple')
ax[2].axhline(0, linestyle='--', color='k')
ax[2].set_ylabel('dprime', fontsize=8)
ax[2].set_xlabel('Time from sound onset', fontsize=8)

f.tight_layout()

plt.show()