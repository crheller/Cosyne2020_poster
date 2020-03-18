"""
In spirit of NAT paper, plot the change in decoding for each site
as a function of where on the heatmap they lie
"""

import loading as ld
import charlieTools.plotting as cplt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# analysis over all prelick period (coarse binning)
# coarse target too (combine all targets)
ap_df = ld.load_dprime('refTar_dprime_bigPass_smallPass_binPreLick')
# using 'TARGET not in i' assures that a different axis is used for each target, but we 
# still collapse across each target dprime for the coarse view here
#filt = [i for i in ap_df.index if ('TARGET' in i) and ('REFERENCE' in i)]
filt = [i for i in ap_df.index if ('TORC' in i) & (('TARGET' not in i) & ('TAR_' not in i) & ('REFERENCE' not in i))]
ap_df = ap_df[ap_df.index.isin(filt)]

a_axis = 'sia'
p_axis = 'sia'
active = ap_df[(ap_df['state']=='big_passive') & (ap_df['axis']==a_axis)]
passive = ap_df[(ap_df['state']=='small_passive') & (ap_df['axis']==p_axis)]
nbins = 10

if (p_axis == 'soa') & (a_axis == 'ssa'):
    X = active['similarity'] # this is the similarity of responses in passive condition
    Y = passive['stim1_pc1_proj_on_dec'] # this is similarity of passive noise to active decoding axis
    title = 'active decoding axis'

elif (p_axis == 'ssa') & (a_axis == 'soa'):
    X = passive['similarity'] # this is the similarity of responses in passive condition
    Y = passive['stim1_pc1_proj_on_dec'] # this is similarity of passive noise to passive decoding axis
    title = 'passive decoding axis'

elif (p_axis == 'sia') & (a_axis == 'sia'):
    X = passive['similarity'] # this is the similarity of responses over all conditions
    Y = passive['stim1_pc1_proj_on_dec'] # this is similarity of overall noise to decoding axis
    title = 'overall decoding axis'

dat = ss.binned_statistic_2d(X, Y,
        values=active['dprime'] - passive['dprime'],
        bins=nbins, statistic='mean')

f, ax = plt.subplots(1, 1)

vmin = -4
vmax = 4
im = ax.imshow(dat[0].T, cmap='PRGn', origin='lower', vmin=vmin, vmax=vmax, interpolation=None)
ax.set_xticks(np.arange(0, len(dat[1])-1))
ax.set_xticklabels(np.round(dat[1], 2), fontsize=6)
ax.set_yticks(np.arange(0, len(dat[2])-1))
ax.set_yticklabels(np.round(dat[2], 2), fontsize=6)

ax.set_title(title+', delta dprime', fontsize=8)
ax.set_xlabel('Similarity of passive responses', fontsize=8)
ax.set_ylabel('Overlap of Target variability with decoding axis', fontsize=8)

f.colorbar(im)

# do the same as above, but plot the change in noise projection between act / pass
# i.e. colormap will be y axis minus the same for active


dat = ss.binned_statistic_2d(X, Y,
        values=passive['stim1_pc1_proj_on_dec'] - active['stim1_pc1_proj_on_dec'],
        bins=nbins, statistic='mean')
vmin = -np.nanmax(dat[0])
vmax = np.nanmax(dat[0])
f, ax = plt.subplots(1, 1)

im = ax.imshow(dat[0].T, cmap='PRGn', origin='lower', vmin=vmin, vmax=vmax, interpolation=None)
ax.set_xticks(np.arange(0, len(dat[1])-1))
ax.set_xticklabels(np.round(dat[1], 2), fontsize=6)
ax.set_yticks(np.arange(0, len(dat[2])-1))
ax.set_yticklabels(np.round(dat[2], 2), fontsize=6)
ax.set_title(title+', delta yaxis', fontsize=8)
ax.set_xlabel('Similarity of passive responses', fontsize=8)
ax.set_ylabel('Overlap of Target variability with decoding axis', fontsize=8)

f.colorbar(im)

plt.show()