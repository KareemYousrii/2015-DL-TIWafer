# import pydevd

# pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
import cPickle
from collections import defaultdict
import itertools
import random
import numpy as np


def triplets(id_class, id_cluster):
    # group ids by cluster
    cluster_ids_pair = defaultdict(list)
    for key, value in id_cluster.items():
        cluster_ids_pair[value].append(key)

    # split each group based on class of values
    cluster_pos_neg_ids = {}
    for key1, value1 in cluster_ids_pair.items():
        pos_neg_ids = defaultdict(list)
        pos = [item for item in value1 if id_class_train[item]=='0']
        neg = list(set(value1)-set(pos))
        pos_neg_ids['0'] = pos
        pos_neg_ids['1'] = neg
        cluster_pos_neg_ids[key1] = pos_neg_ids

    result = []
    # for each patch find triples and add to result
    for patch_id, cluster_value in id_cluster.items():
        dict_pos_neg = cluster_pos_neg_ids[cluster_value]
        patch_class = id_class[patch_id]
        if patch_class == '0':
            # for ith patch find positives
            positives = dict_pos_neg['0']
            positives = list(set(positives)-set([patch_id]))
            # for ith patch find negatives
            negatives = dict_pos_neg['1']
        elif patch_class == '1':
            # for ith patch find positives
            positives = dict_pos_neg['1']
            positives = list(set(positives)-set([patch_id]))
            # for ith patch find negatives
            negatives = dict_pos_neg['0']
        else:
            print "undefined class error"

        if len(positives)>5:
            randIdxP = random.sample(xrange(0, len(positives)-1), 5)
            positives = list(np.array(positives)[randIdxP])
        if len(negatives)>5:
            randIdxN = random.sample(xrange(0, len(negatives)-1), 5)
            negatives = list(np.array(negatives)[randIdxN])

        print "patch id {} cluster value {} positives {} " \
              "negatives {}".format(patch_id, cluster_value, len(positives), len(negatives))

        ids_combination = list(itertools.product([patch_id], positives, negatives))
        if len(ids_combination) > 5:
            randIdx = random.sample(xrange(0, len(ids_combination)-1), 5)
            ids_combination = np.array(ids_combination)[randIdx]
        else:
            print "no. of samples for {} is {}".format(patch_id, len(ids_combination))
            continue
        [result.append(item) for item in ids_combination]

    with open('patch_triplets.p', 'wb') as handle:
        cPickle.dump(result, handle)
        print "Saving patch triplets into patch_triplets.p"



if __name__ == '__main__':
    id_class_pair = cPickle.load(open('patch_id_class.p', 'rb'))
    id_cluster_pair = cPickle.load(open('id_cluster.p', 'rb'))
    # split items to train and test
    index_train = random.sample(xrange(0, len(id_class_pair)-1), int(.75*len(id_class_pair)))
    ids = np.array([key for key, val in id_class_pair.items()])
    train_ids = list(ids[index_train])
    with open('train_indexes.p', 'wb') as handle:
        cPickle.dump(index_train, handle)

    id_class_train = {}
    id_cluster_train = {}
    for train_key in train_ids:
        id_class_train[train_key] = id_class_pair[train_key]
        id_cluster_train[train_key] = id_cluster_pair[train_key]

    triplets(id_class_train, id_cluster_train)
