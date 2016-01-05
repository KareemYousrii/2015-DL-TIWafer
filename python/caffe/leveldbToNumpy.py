import os
import plyvel
import numpy
import matplotlib.pyplot as plt
import numpy as np

# First compile the Datum, protobuf so that we can load using protobuf
# This will create datum_pb2.py
os.system('protoc -I={0} --python_out={1} {0}datum.proto'.format("./", "./"))

import datum_pb2

LMDB_PATH = "features/"

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)

if __name__ == '__main__':
    db = plyvel.DB(LMDB_PATH)

    visualize = True
    datum = datum_pb2.Datum()

    for key, value in db:
        datum.ParseFromString(value)
        # Read the datum.data
        #img_data = numpy.array(bytearray(datum.data))\
        #      .reshape(datum.channels, datum.height, datum.width)
        #if visualize:
        #   plt.imshow(img_data.transpose([1,2,0]))
        #  plt.show()
        filters = np.array(bytearray(datum.data)).reshape(datum.channels, datum.height, datum.width)
        vis_square(filters.transpose(0, 2, 3, 1))
        print key


