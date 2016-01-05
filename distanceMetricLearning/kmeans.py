#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import sys
import lmdb
from collections import defaultdict

# Make sure that caffe is on the python path:
sys.path.append('/home/karn_s/2015-DL-TIWafer/python')
import caffe
from caffe.proto import caffe_pb2


LMDB_PATH_Features = "TIWafer_Patches/features/"
env_features = lmdb.open(LMDB_PATH_Features, readonly=True, lock=False)

if __name__ == '__main__':
    datum_features = caffe_pb2.Datum()
    with env_features.begin() as txn_features:
        cur_features = txn_features.cursor()
        id_feature = {}
        for i in xrange(20756):
            if not cur_features.next():
                break
            # Read the current cursor
            key_feature, value_feature = cur_features.item()
            # convert to datum
            datum_features.ParseFromString(value_feature)
            features = np.array(datum_features.float_data).astype(np.float32)
            id_feature[str(key_feature)] = features

    data = [value for key, value in id_feature.items()]
    reduced_data = PCA(n_components=2).fit_transform(data)

    keys = [key for key, value in id_feature.items()]

    id_reduced_feature = zip(keys, reduced_data)

    with open('id_reduced_feature.p', 'wb') as handle:
        cPickle.dump(id_reduced_feature, handle)

    kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)
    kmeans.fit(reduced_data)

    with open('kmeans_patch.p', 'wb') as handle:
        cPickle.dump(kmeans, handle)

    id_cluster = {}
    for ident, feat in id_reduced_feature:
        cluster = kmeans.predict(feat)[0]
        id_cluster[ident] = cluster

    with open('id_cluster.p', 'wb') as handle:
        cPickle.dump(id_cluster, handle)
