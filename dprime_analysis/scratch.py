import loading as ld
import matplotlib.pyplot as plt
import numpy as np

df = ld.load_dprime('refTar_dprime_act_pass_binPreLick')
# get indices of interest (indices comparing equivalent timebins)
filt = [i for i in df.index if ('TAR' in i) & ('REF' in i) & (i.split('_')[1]==i.split('_')[3])]   


# ============= various notes / warnings about dprime dataframe ====================
 
# there will be many negative dprime values for 'soa' axis
print(sum(df[df.axis=='soa']['dprime']<0))

# if variance of stim1 / stim2 is 0 on decoding axis, dprime is nan
print(sum(np.isnan(df['dprime'])))
df[np.isnan(df['dprime'])].head()

# I didn't include spont data in any of the dprime analysis. 0 indicates first evoked bin
df['category'].unique()

# Remember that pc1 var explained will be different depending on the decoding axis
# because it's computed after normalizing the std of each neuron to the decoding data
#                       pc1_var_explained  pc1_var_explained_all axis    state
# TAR_14000_0_TARGET_0           0.122312               0.122312  ssa   active
# TAR_14000_0_TARGET_0           0.204615               0.204615  ssa  passive
# TAR_14000_0_TARGET_0           0.136004               0.152553  sia   active
# TAR_14000_0_TARGET_0           0.221761               0.152553  sia  passive
# TAR_14000_0_TARGET_0           0.561243               0.204615  soa   active
# TAR_14000_0_TARGET_0           0.305938               0.122312  soa  passive

# some dprime values will explode, and this is particularly likely for cases with low
# trial numbers (TORC - TORC comparisons in particular). These should probably be excluded
# when averaging over stuff
df['dprime'].max()
f = round(df[df['dprime'] > df['dprime'].std() * 3].shape[0] / df.shape[0] * 100, 3)
print("{} percent of data is greater than 3 std of the data".format(f))
print("which is a dprime of {}".format(df['dprime'].std() * 3))

# 'difference' metric is computed on (and normalized to) the data used for the 
# decoding axis. also, remember it's computed on the variance normalized data
df[(df.state=='active') & (df.axis=='sia')]['difference'] # difference computed over all data
df[(df.state=='active') & (df.axis=='ssa')]['difference'] # difference computed over active data
df[(df.state=='active') & (df.axis=='soa')]['difference'] # difference computed over passive data


# ================ random exploratory plots ==========================

# distribution of dprime for each axis / state
f, ax = plt.subplots(2, 3, figsize=(12, 6))
metric = 'dprime'
bins = np.arange(-df[metric].std(), 3 * df[metric].std(), 0.1)

ax[0, 0].set_title('active, ssa', fontsize=8)
df[(df.state=='active') & (df.axis=='ssa')][metric].hist(bins=bins, ax=ax[0, 0], grid=False)

ax[0, 1].set_title('active, sia', fontsize=8)
df[(df.state=='active') & (df.axis=='sia')][metric].hist(bins=bins, ax=ax[0, 1], grid=False)

ax[0, 2].set_title('active, soa', fontsize=8)
df[(df.state=='active') & (df.axis=='soa')][metric].hist(bins=bins, ax=ax[0, 2], grid=False)

ax[1, 0].set_title('passive, ssa', fontsize=8)
df[(df.state=='passive') & (df.axis=='ssa')][metric].hist(bins=bins, ax=ax[1, 0], grid=False)

ax[1, 1].set_title('passive, sia', fontsize=8)
df[(df.state=='passive') & (df.axis=='sia')][metric].hist(bins=bins, ax=ax[1, 1], grid=False)

ax[1, 2].set_title('passive, soa', fontsize=8)
df[(df.state=='passive') & (df.axis=='soa')][metric].hist(bins=bins, ax=ax[1, 2], grid=False)
f.tight_layout()

# distribution of pc1_proj_dec for each axis / state
f, ax = plt.subplots(2, 3, figsize=(12, 6))
metric = 'stim1_pc1_proj_on_u1'
bins = np.arange(0, 1, 0.01)

ax[0, 0].set_title('active, ssa', fontsize=8)
df[(df.state=='active') & (df.axis=='ssa')][metric].hist(bins=bins, ax=ax[0, 0], grid=False)

ax[0, 1].set_title('active, sia', fontsize=8)
df[(df.state=='active') & (df.axis=='sia')][metric].hist(bins=bins, ax=ax[0, 1], grid=False)

ax[0, 2].set_title('active, soa', fontsize=8)
df[(df.state=='active') & (df.axis=='soa')][metric].hist(bins=bins, ax=ax[0, 2], grid=False)

ax[1, 0].set_title('passive, ssa', fontsize=8)
df[(df.state=='passive') & (df.axis=='ssa')][metric].hist(bins=bins, ax=ax[1, 0], grid=False)

ax[1, 1].set_title('passive, sia', fontsize=8)
df[(df.state=='passive') & (df.axis=='sia')][metric].hist(bins=bins, ax=ax[1, 1], grid=False)

ax[1, 2].set_title('passive, soa', fontsize=8)
df[(df.state=='passive') & (df.axis=='soa')][metric].hist(bins=bins, ax=ax[1, 2], grid=False)
f.tight_layout()

plt.show()