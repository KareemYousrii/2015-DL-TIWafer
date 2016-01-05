import os
import numpy
import skimage
import sklearn
import scipy
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


path = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\preprocessed\\images"
Array = numpy.zeros((16296, 27*27*3), dtype='float32')

file_Names = []
Labels = []

Kmeans = KMeans(n_clusters=8,
                       init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001, precompute_distances='auto',
                       verbose=0, random_state=None, copy_x=True, n_jobs=1)
print "Loading Files ... "

index = 0
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".png")):
            # Array.append(scipy.misc.imread(name=name, flatten=0))
            #img = scipy.misc.imread(name=os.path.join(root, name), flatten=0)

            img = Image.open(os.path.join(root, name))
            img = img.resize((27,27), Image.ANTIALIAS)
            # img = list(img.get_data())
            # img = map(list, img)
            image = numpy.array(img)
            image = normalize(image)
            image = image.flatten()
            Array[index] = image
            """
            if not os.path.exists(root+"\\new"):
                os.makedirs(root+"\\new")

            scipy.misc.imsave(os.path.join(root, name), image)
            """
            # print "Files Processed : ", index, " out of 16368"
            index += 1
            file_Names.append(str(os.path.join(root, name)))
print "Files Processed : ", index, " out of 16368"

"""Clusters = Kmeans.fit(Array)
Labels = Clusters.labels_
file = open('output.txt', 'w')
for i in xrange(len(file_Names)):
    file.write("\n" + "file: " + str(file_Names[i]) + ", Label: " + str(Labels[i]))

bins = numpy.linspace(0, 8, 8)
freq, bins = numpy.histogram(Labels, bins)
plt.hist(Labels, bins)
plt.title("Cluster Histogram")
plt.xlabel("Cluster")
plt.ylabel("Frequency")

plt.show()
Clusters = Clusters"""


reduced_data = PCA(n_components=2).fit_transform(Array[:5000])
kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(numpy.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
