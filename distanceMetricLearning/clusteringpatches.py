import pydevd
pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import cPickle
from tsne import bh_sne
from sklearn.preprocessing import MinMaxScaler

# dataDict = cPickle.load(open('patchIdFeaturePair.p', 'rb'))
# data = []
# [data.extend(value) for key, value in dataDict.items()]
# X_raw = np.array(data, dtype=np.float64)
# X_2d = bh_sne(X_raw)
# X_scaled = MinMaxScaler().fit_transform(X_2d)
# cPickle.dump(X_scaled, open('2dpatchfeatures.p','wb'))

data = cPickle.load(open('2dpatchfeatures.p', 'rb'))

fig, axes23 = plt.subplots(2, 3)

for method, axes in zip(['single', 'complete'], axes23):
    z = hac.linkage(data, method=method)

    # Plotting
    axes[0].plot(range(1, len(z)+1), z[::-1, 2])
    knee = np.diff(z[::-1, 2], 2)
    axes[0].plot(range(2, len(z)), knee)

    num_clust1 = knee.argmax() + 2
    knee[knee.argmax()] = 0
    num_clust2 = knee.argmax() + 2

    axes[0].text(num_clust1, z[::-1, 2][num_clust1-1], 'possible\n<- knee point')

    part1 = hac.fcluster(z, num_clust1, 'maxclust')
    part2 = hac.fcluster(z, num_clust2, 'maxclust')

    clr = ['#0000ff', '#8a2be2', '#a52a2a', '#deb887', '#5f9ea0', '#7fff00', '#d2691e', '#ff7f50',
            '#6495ed', '#dc143c', '#00ffff', '#00008b', '#008b8b', '#b8860b', '#a9a9a9', '#006400', '#8b0000']

    #clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    #'#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']

    for part, ax in zip([part1, part2], axes[1:]):
        for cluster in set(part):
            ax.scatter(data[part == cluster, 0], data[part == cluster, 1],
                       color=clr[cluster]
                       )

    m = '\n(method: {})'.format(method)
    plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
             ylabel='{}\ncluster distance'.format(m))
    plt.setp(axes[1], title='{} Clusters'.format(num_clust1))
    plt.setp(axes[2], title='{} Clusters'.format(num_clust2))

plt.tight_layout()
plt.show()
