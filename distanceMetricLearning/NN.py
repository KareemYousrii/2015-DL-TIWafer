# import pydevd
# pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)

import cPickle
from learning_dist_metrics.ldm import LDM
from load_data_dml import load_sample_data

print "Loading Data...."
sample_data = load_sample_data()
print "Done Loading Data. Learning Distance Metric...."
ldm = LDM()
ldm.fit(sample_data.data, sample_data.sim_pairs, sample_data.diff_pairs)
with open('ldm.p', 'wb') as handle:
    cPickle.dump(ldm, handle)
