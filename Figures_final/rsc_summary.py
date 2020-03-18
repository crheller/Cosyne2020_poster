"""
Change in noise correlations as function of behavioral state.
"""

import loading as ld
import matplotlib.pyplot as plt
import charlieTools.plotting as cplt
import matplotlib as mp
mp.rcParams.update({'svg.fonttype': 'none'})

fn = '/auto/users/hellerc/code/projects/Cosyne2020_poster/svgs/rsc_summary.svg'

rsc_df = ld.load_rsc('tar_rsc_0_0,2')
rsc_df_site = rsc_df.groupby(by='site').mean()

f, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.bar([0, 1, 2],
        [rsc_df_site['act_rsc'].mean(),
        rsc_df_site['passBig_rsc'].mean(),
        rsc_df_site['passSmall_rsc'].mean()],
        edgecolor='k', color='lightgrey', lw=2)

for s in rsc_df_site.index:
    vals = rsc_df_site.loc[s][['act_rsc', 'passBig_rsc', 'passSmall_rsc']]
    ax.plot([0, 1, 2], vals, 'o-', color='k')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Active', 'Pupil-matched \n passive', 'Small pupil \n passive'], rotation=45)
ax.set_ylabel('Noise Correlation')
ax.set_aspect(cplt.get_square_asp(ax))

f.tight_layout()

f.savefig(fn)

plt.show()
