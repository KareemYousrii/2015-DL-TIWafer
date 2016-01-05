import pydevd

pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
__author__ = 'z003fafb'
from learning_dist_metrics.datasets import load_data_new
from learning_dist_metrics.ldm import LDM


sample_data = load_data_new.load_sample_data()

ldm = LDM()

ldm.fit(sample_data.data, sample_data.sim_pairs, sample_data.diff_pairs)