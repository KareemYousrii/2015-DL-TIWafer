"""
"""
from os.path import dirname
from pandas import read_csv
from itertools import combinations
from random import shuffle
import pandas as pd
import cPickle
import numpy as np
import lmdb
import cPickle
from collections import defaultdict
import sys

# Make sure that caffe is on the python path:
sys.path.append('/home/karn_s/2015-DL-TIWafer/python')
import caffe
from caffe.proto import caffe_pb2

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object that
    exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_data_sdml():
    id_class_pair = cPickle.load(open('patch_id_class.p', 'rb'))
    patch_triplets = cPickle.load(open('patch_triplets.p', 'rb'))
    train_index = cPickle.load(open('train_indexes.p', 'rb'))
    IdFeature = cPickle.load(open('IdFeature.p', 'rb'))

    ids = np.array([key for key, val in id_class_pair.items()])
    train_ids = sorted(list(ids[train_index]))

    y = []
    ti_data = []
    for patch_id in train_ids:
        y.append(id_class_pair[patch_id])
        ti_data.append(IdFeature[patch_id])

    sim_pairs = []
    diff_pairs = []
    for item in patch_triplets:
        sim_pairs.append([item[0], item[1]])
        diff_pairs.append([item[0], item[2]])
    shuffle(diff_pairs)

    res = Bunch(sortedIds=train_ids, data=ti_data, target=y, sim_pairs=sim_pairs, diff_pairs=diff_pairs)
    return res


def load_sample_data():
    id_class_pair = cPickle.load(open('patch_id_class.p', 'rb'))
    patch_triplets = cPickle.load(open('patch_triplets.p', 'rb'))
    train_index = cPickle.load(open('train_indexes.p', 'rb'))
    IdFeature = cPickle.load(open('IdFeature.p', 'rb'))

    ids = np.array([key for key, val in id_class_pair.items()])
    train_ids = list(ids[train_index])

    y = []
    train_id_feat = {}
    for patch_id in train_ids:
        y.append(id_class_pair[patch_id])
        train_id_feat[patch_id] = IdFeature[patch_id]

    sim_pairs = []
    diff_pairs = []
    for item in patch_triplets:
        sim_pairs.append([item[0], item[1]])
        diff_pairs.append([item[0], item[2]])
    shuffle(diff_pairs)

    X_new = pd.DataFrame.from_dict(train_id_feat, orient='index')

    res = Bunch(data=X_new, target=y, sim_pairs=sim_pairs, diff_pairs=diff_pairs)
    return res

    # IDs = [key for key, val in IdFeature.items()]
    # data = [val for key, val in IdFeature.items()]
    # X = np.array(data)
    # nrows, ncols = X.shape
    # column_features = ['ID']
    # for i in xrange(ncols):
    #     column_features.append("feature_{}".format(i))
    #
    # X_new = pd.DataFrame.from_records(X, index=IDs, columns=column_features)



    # datum_features = caffe_pb2.Datum()
    # LMDB_PATH_Features = "TIWafer_Patches/features/"
    # env_features = lmdb.open(LMDB_PATH_Features, readonly=True, lock=False)
    # IdFeature = defaultdict(list)
    # with env_features.begin() as txn_features:
    #     cur_features = txn_features.cursor()
    #     for i in xrange(20756):
    #         if not cur_features.next():
    #             break
    #         # Read the current cursor
    #         key_feature, value_feature = cur_features.item()
    #         datum_features.ParseFromString(value_feature)
    #         features = np.array(datum_features.float_data).astype(np.float32)
    #         IdFeature[key_feature] = list(features)
    # with open('IdFeature.p', 'wb') as handle:
    #     cPickle.dump(IdFeature, handle)


