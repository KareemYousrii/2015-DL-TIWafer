#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest
import tempfile
import os
import six
import Image
import cv2

sys.path.append('/home/karn_s/caffe/python')
import caffe
from caffe.proto import caffe_pb2

# helper functions
def get_kernel_size(factor):
    return 2 * factor - factor % 2


def get_pad(factor):
    return int(np.ceil((factor - 1) / 2.))

def upsample_creator(factor, num_in):
    kernel = get_kernel_size(factor)
    stride = factor
    lp = caffe.LayerParameter("""name: "upsample", type: "Deconvolution"
        convolution_param { kernel_size: %d stride: %d num_output: %d group: %d pad: %d
        weight_filler: { type: "bilinear_upsampling" } bias_term: false }""" % (
            kernel, stride, num_in, num_in, get_pad(factor)
        ))
    return caffe.create_layer(lp.to_python())


def python_net_file(factor, num_in):
    kernel = get_kernel_size(factor)
    stride = factor
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: "upsample", type: "Deconvolution"
        convolution_param { kernel_size: {} stride: {} num_output: {} group: {} pad: {}
        weight_filler: { type: "bilinear_upsampling" } bias_term: false }""".format(
            kernel, stride, num_in, num_in, get_pad(factor)
            )
        )
        return f.name

def simple_net_file(factor, num_in):
    """Make a simple net prototxt, based on test_net.cpp, returning the name
    of the (temporary) file."""
    kernel = get_kernel_size(factor)
    stride = factor
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    # f.write("""name: 'testnet'
    #     input: "data"
    #     input_dim: 1
    #     input_dim: 1
    #     input_dim: 100
    #     input_dim: 100
    #     layer {
    #         type: 'Deconvolution'
    #         name: 'upsample'
    #         bottom: 'data'
    #         top: 'upsample'
    #         convolution_param { num_output: """+str(num_in)+""" kernel_size: """+str(kernel)
    #         + """ pad: """+str(get_pad(factor))+""" stride: """+str(stride)+""" weight_filler { type: 'bilinear' }
    #         bias_term: false
    #         }
    #     }
    # """)
    f.write("""name: 'testnet'
        layer {
            name: "data"
            type: "ImageData"
            top: "data"
            top: "label"
            image_data_param {
                source: "file_list.txt"
                batch_size: 1
                new_height: 256
                new_width: 256
            }
        }
        layer {
            type: 'Deconvolution'
            name: 'upsample'
            bottom: 'data'
            top: 'upsample'
            convolution_param { num_output: """+str(num_in)+""" kernel_size: """+str(kernel)
            + """ pad: """+str(get_pad(factor))+""" stride: """+str(stride)+""" weight_filler { type: 'bilinear' }
            bias_term: false
            }
        }
    """)
    f.close()
    return f.name


if __name__ == '__main__':
    factor = 3

    net_file = simple_net_file(factor=factor, num_in=3)
    net = caffe.Net(net_file, caffe.TRAIN)

    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

    # load image and prepare as a single input batch for Caffe
    #im = np.array(Image.open('images/cat_gray.jpg'))
    #im_input = im[np.newaxis, np.newaxis, :, :]
    #net.blobs['data'].reshape(*im_input.shape)
    #net.blobs['data'].data[...] = im_input
    datum = caffe_pb2.Datum()
    net.forward()
    im = net.blobs['data'].data[...][0]
    im_data = Image.fromarray(im, 'RGB')
    filt_min, filt_max = net.blobs['upsample'].data.min(), net.blobs['upsample'].data.max()

    imgo = net.blobs['upsample'].data.copy().transpose(0, 2, 3, 1)[0]

    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    #lum_img = img[:,:,0]
    #imgplot = plt.imshow(lum_img)
    imgplot = plt.imshow(im_data.astype(np.uint8), interpolation='none', cmap='gray')
    a.set_title('Before')
    #plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(imgo.astype(np.uint8), interpolation='none', cmap='gray')
    a.set_title('After')

    plt.show()

    b = cv2.cvtColor(imgo.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.SaveImage('transformedCat.png', b)
    #plt.title("original image")
    #plt.imshow(im)
    #plt.axis('off')
    #plt.imshow(imgo, interpolation='none')
    #plt.show()





