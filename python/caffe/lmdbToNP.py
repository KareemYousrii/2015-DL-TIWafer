import os
import lmdb
import numpy
import matplotlib.pyplot as plt

# First compile the Datum, protobuf so that we can load using protobuf
# This will create datum_pb2.py
os.system('protoc -I={0} --python_out={1} {0}datum.proto'.format("./", "./"))

import datum_pb2

LMDB_PATH = "DB_TIWafer_lmdb/"

env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

visualize = True

datum = datum_pb2.Datum()
with env.begin() as txn:
    cur = txn.cursor()
    for i in xrange(10):
        if not cur.next():
            cur.first()
        # Read the current cursor
        key, value = cur.item()
        # convert to datum
        datum.ParseFromString(value)
        # Read the datum.data
        img_data = numpy.array(bytearray(datum.data))\
            .reshape(datum.channels, datum.height, datum.width)
        if visualize:
            plt.imshow(img_data.transpose([1,2,0]))
            plt.show()

        print key