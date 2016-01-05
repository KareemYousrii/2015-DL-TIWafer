#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
from metric_learn.itml import ITML
from load_data_dml import load_data
import cPickle


def test_tiwafer():
    num_constraints = 2000
    print 'Loading Data...\n'
    tiwafer = load_data()

    n = tiwafer.data.shape[0]
    C = ITML.prepare_constraints(tiwafer.target, n, num_constraints)

    print 'Learning Distance Metric...\n'
    itml = ITML().fit(tiwafer.data, C, verbose=True)
    itmlLearnedMat = itml.transformer()
    cPickle.dump(itmlLearnedMat, open('itmlLearnedMat.p', 'wb'))
    with open('itml.p', 'wb') as handle:
        cPickle.dump(itml, handle)

if __name__ == '__main__':
    test_tiwafer()
    print 'Done Learning'