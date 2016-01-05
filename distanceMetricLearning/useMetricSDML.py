#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
from metric_learn import SDML
from load_data_dml import load_data_sdml
import numpy as np
import pandas as pd
import cPickle
from random import choice


def prepare_constraints_old(labels, num_points, num_constraints):
    np.random.seed(1234)
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    b, d = np.empty((2, num_constraints), dtype=int)
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
        b[i] = choice(np.nonzero(labels == al)[0])
        d[i] = choice(np.nonzero(labels != cl)[0])
    W = np.zeros((num_points,num_points))
    W[a,b] = 1
    W[c,d] = -1
    # make W symmetric
    W[b,a] = 1
    W[d,c] = -1
    return W


def prepare_constraints(sorted_ids, sim_pairs, diff_pairs):
    num_points = len(sorted_ids)
    W = np.zeros((num_points, num_points))
    # Weight Matrix should be symmetrical
    # prepare a lower triangular matrix and later convert to full
    col_range = 1
    for row, idr in enumerate(sorted_ids):
        for col, idc in enumerate(sorted_ids[col_range]):
            col_range += 1
            if row == col:
                W[row, col] = 1
                continue
            elif [idr, idc] in sim_pairs:
                W[row, col] = 1
                # make W symmetric
                W[col, row] = 1
                continue
            elif [idr, idc] in diff_pairs:
                W[row, col] = -1
                # make W symmetric
                W[col, row] = -1
                continue
            else:
                raise ValueError('The pair doesnot exist in either similar or disimilar pairs.'
                                 ' something must be wrong while generating pair')
    return W


def test_tiwafer():
    num_constraints = 1500
    print "Loading Data...."
    tiwafer_data = load_data_sdml()
    sim_pairs = tiwafer_data.sim_pairs
    diff_pairs = tiwafer_data.diff_pairs
    sorted_ids = tiwafer_data.sortedIds
    ti_data = np.array(tiwafer_data.data)
    labels = np.array(tiwafer_data.target)

    print "Done Loading Data.\nLearning Distance Metric...."

    num_points = len(sorted_ids)
    W = prepare_constraints_old(labels, num_points, num_constraints)

    sdml = SDML()
    # W = prepare_constraints(sorted_ids, sim_pairs, diff_pairs)

    sdml.fit(ti_data, W)
    W_metric = sdml.metric()
    cPickle.dump(W_metric, open('W_metric_sdml.p', 'wb'))
    W_trans = sdml.transformer()
    with open('W_trans_sdml.p', 'wb') as handle:
        cPickle.dump(W_trans, handle)

if __name__ == '__main__':
    test_tiwafer()
    print 'Done Learning'