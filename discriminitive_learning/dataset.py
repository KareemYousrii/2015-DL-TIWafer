import os
import random
from PIL import Image
import numpy

# Function for creating the patched dataset from reading patches #
# Takes "dir_path" as the path to the directory that contains the image files contaning patches #
# Also takes the "label" of the directory i.e. if it is good or bad patches. label = 1 for good and 0 for bad #

def make_array_labels(dir_path, label):

    if label == 1:
        print "Processing good patches ... \n\n"
    else:
        print "procesing bad patches ... \n\n"

    index = 0
    # Array = numpy.zeros((16296, 27*27*3), dtype='float32')
    Array = []
    labels = []

    for root, dirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith((".png")):

                # print "Procesing file ", os.path.join(root, name)

                img = Image.open(os.path.join(root, name))
                image_size = img.size

                if (image_size == (27,27)):
                    image = numpy.array(img)
                    image = image.flatten()
                    Array.append(image)
                    labels.append(label)
                    index += 1

    Array = Array
    return Array, labels

def get_data():

    return_list = []

    good_path = "/home/mustajab/PycharmProjects/Deep Learning/cropping_input/images/good_patches_"
    bad_path = "/home/mustajab/PycharmProjects/Deep Learning/cropping_input/images/bad_patches_"

    good_array, good_labels = make_array_labels(good_path, 1)
    bad_array, bad_labels = make_array_labels(bad_path, 0)

    good_array, bad_array = good_array[:35000], bad_array[:15000]
    good_labels, bad_labels = good_labels[:35000], bad_labels[:15000]

    array = numpy.concatenate((good_array, bad_array), axis=0)
    labels = numpy.concatenate((good_labels, bad_labels), axis=0)

    c = numpy.c_[array.reshape(len(array), -1), labels.reshape(len(labels), -1)]

    numpy.random.shuffle(c)

    new_array = c[:, :array.size//len(array)].reshape(array.shape)
    new_labels = c[:, array.size//len(array):].reshape(labels.shape)

    temp_array, temp_labels = new_array[35000:], new_labels[35000:]

    return_list.append([new_array[:35000], new_labels[:35000]])
    return_list.append([temp_array[:5000], temp_labels[:5000]])
    return_list.append([temp_array[5000:], temp_labels[5000:]])

    return return_list


if __name__ == '__main__':
    get_data()