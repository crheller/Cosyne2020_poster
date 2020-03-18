"""
Compare active/passive projection of noise on decoding axis (for Tar / Ref)
Remember that 'noise' for REF is really more like signal correlation
"""
import loading as ld
import charlieTools.plotting as cplt
import matplotlib.pyplot as plt

ap_df = ld.load_dprime('refTar_dprime_act_pass')

# only keep dprime results where comparing the same time bins to each other
# and comparing a TAR to a REF
filt = [i for i in ap_df.index if ('TAR' in i) & ('REF' in i) & (i.split('_')[1]==i.split('_')[3])]   

ap_df = ap_df[ap_df.index.isin(filt)]

active = ap_df[ap_df['state']=='active']
passive = ap_df[ap_df['state']=='passive']

# compare the passive to active to overall decoding axis (how similar are they?)
# by plotting the similarity of stim2 (target) noise and dec. axis
# for ssa, sia, and soa. This works bc target data is constant
# but dec. axis changes for each of these cases
f, ax = plt.subplots(1, 3)

col = 'stim2_pc1_proj_on_dec'
active_ax = active['axis']=='ssa'
passive_ax = active['axis']=='soa'
overall_ax = active['axis']=='sia'
mi = active[col].min()
ma = active[col].max()

ax[0].plot(active[active_ax][col], active[passive_ax][col], 'k.')
ax[0].plot([mi, ma], [mi, ma], '--', color='grey')
ax[0].set_xlabel('active decoding axis', fontsize=8)
ax[0].set_ylabel('passive decoding axis', fontsize=8)
ax[0].set_title('Similarity of decoding axis', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(active[active_ax][col], active[overall_ax][col], 'k.')
ax[1].plot([mi, ma], [mi, ma], '--', color='grey')
ax[1].set_xlabel('active decoding axis', fontsize=8)
ax[1].set_ylabel('overall decoding axis', fontsize=8)
ax[1].set_title('Similarity of decoding axis', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[2]))

ax[2].plot(active[overall_ax][col], active[passive_ax][col], 'k.')
ax[2].plot([mi, ma], [mi, ma], '--', color='grey')
ax[2].set_xlabel('overall decoding axis', fontsize=8)
ax[2].set_ylabel('passive decoding axis', fontsize=8)
ax[2].set_title('Similarity of decoding axis', fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()


# for each decoding axis, compare the target 
# noise overlap between active and passive
f, ax = plt.subplots(1, 3, figsize=(8, 6))
col = 'stim2_pc1_proj_on_dec'

ax[0].plot(active[active['axis']=='ssa'][col], passive[passive['axis']=='soa'][col], 'k.')
ax[0].plot([0, 1], [0, 1], '--', color='grey')
ax[0].set_title('Similarity noise / decoding axis \n Active decoding axis', fontsize=8)
ax[0].set_xlabel('Passive', fontsize=8)
ax[0].set_ylabel('Active', fontsize=8)
ax[0].set_aspect(cplt.get_square_asp(ax[0]))

ax[1].plot(active[active['axis']=='soa'][col], passive[passive['axis']=='ssa'][col], 'k.')
ax[1].plot([0, 1], [0, 1], '--', color='grey')
ax[1].set_title('Similarity noise / decoding axis \n Passive decoding axis', fontsize=8)
ax[1].set_xlabel('Passive', fontsize=8)
ax[1].set_ylabel('Active', fontsize=8)
ax[1].set_aspect(cplt.get_square_asp(ax[1]))

ax[2].plot(active[active['axis']=='sia'][col], passive[passive['axis']=='sia'][col], 'k.')
ax[2].plot([0, 1], [0, 1], '--', color='grey')
ax[2].set_title('Similarity noise / decoding axis \n Overall decoding axis', fontsize=8)
ax[2].set_xlabel('Passive', fontsize=8)
ax[2].set_ylabel('Active', fontsize=8)
ax[2].set_aspect(cplt.get_square_asp(ax[2]))

f.tight_layout()

plt.show()