#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
import sys
import lmdb
import numpy
import matplotlib.pyplot as plt
import cPickle
from collections import defaultdict

# Make sure that caffe is on the python path:
sys.path.append('/home/karn_s/2015-DL-TIWafer/python')
import caffe
from caffe.proto import caffe_pb2


def fetchClass(filename):
    items = filename.split('/')
    name = items[len(items)-1]
    classterm = name.split('__')[0]
    classval = classterm.split('_')[1]
    return classval


def fetchID(filename):
    items = filename.split('/')
    name = items[0]
    id = name.split('_')[0]
    return int(id)

LMDB_PATH_Features = "TIWafer_Patches/features/"
LMDB_PATH_Patch = "Lmdb_Patches/"
env_patch = lmdb.open(LMDB_PATH_Patch, readonly=True, lock=False)
env_features = lmdb.open(LMDB_PATH_Features, readonly=True, lock=False)
if __name__ == '__main__':
    # dictionary for id-class pair
    id_class = {}
    feature_class_Pair = defaultdict(list)
    datum_features = caffe_pb2.Datum()
    datum_patch = caffe_pb2.Datum()
    with env_features.begin() as txn_features, env_patch.begin() as txn_patch:
        cur_features = txn_features.cursor()
        cur_patch = txn_patch.cursor()
        count_patch = 0
        while cur_patch.next():
            count_patch +=1
        for i in xrange(count_patch):
            cur_features.next()
            cur_patch.next()
            # Read the current cursor
            key_feature, value_feature = cur_features.item()
            key_patch, value_patch = cur_patch.item()

            class_value = fetchClass(key_patch)
            # fetch id-class pair
            id_class[key_feature] = class_value

        #     idNum = fetchID(key_patch)
        #     if idNum-int(key_feature) == 0:
        #         # convert to datum
        #         datum_features.ParseFromString(value_feature)
        #         features = numpy.array(datum_features.float_data).astype(numpy.float32)
        #         feature_class_Pair[class_value].append(features)
        #         #print "saved feature for {}".format(key_patch)
        #
        # with open('patchIdFeaturePair.p', 'wb') as handle:
        #     cPickle.dump(feature_class_Pair, handle)
        # print "Saving Feature for "+str(i)+" Patches into patchIdFeaturePair.p"

        with open('patch_id_class.p', 'wb') as handle:
            cPickle.dump(id_class, handle)
        print "Saving patch id-class pair for "+str(i)+" Patches into patch_id_class.p"