import pydevd
pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
import sys
import lmdb
import numpy
import matplotlib.pyplot as plt
import cPickle

# First compile the Datum, protobuf so that we can load using protobuf
# This will create datum_pb2.py
#os.system('protoc -I={0} --python_out={1} {0}datum.proto'.format("./", "./"))
#import datum_pb2

sys.path.append('/home/karn_s/caffe/python')
import caffe
from caffe.proto import caffe_pb2

LMDB_PATH = "TIWaffer_Features_lmdb/"
env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
visualize = True

def vis_square(key, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = numpy.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    cPickle.dump(data, open('outImages/'+key+'.p', 'wb'))
    #plt.imshow(data)

if __name__ == '__main__':
    #datum = datum_pb2.Datum()
    imageIdFeaturePair = {}
    datum = caffe_pb2.Datum()
    with env.begin() as txn:
        cur = txn.cursor()
        for i in xrange(4096):
            if not cur.next():
                break
                #cur.first()
            # Read the current cursor
            key, value = cur.item()
            # convert to datum
            datum.ParseFromString(value)
            # Read the datum.data
            #img_data = numpy.array(bytearray(datum.data))\
            #    .reshape(datum.channels, datum.height, datum.width)
            #below works
            #img_data = numpy.array(datum.float_data).astype(numpy.float32).reshape(datum.channels, datum.height, datum.width)
            features = numpy.array(datum.float_data).astype(numpy.float32)
            imageIdFeaturePair[key] = features
            #vis_square(key, img_data)
            #print key
        with open('imageIdFeaturePairLMDB.p', 'wb') as handle:
            cPickle.dump(imageIdFeaturePair, handle)
        print "Saving Feature for "+str(i)+" Images into imageIdFeaturePairLMDB.p"