#import pydevd
#pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
__author__ = 'z003fafb'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

with open('cropping_input_data.txt', 'r') as f1:
    targetDir = "/home/karn_s/deeplearning/TIWafer/patches"
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    lines = f1.readlines()
    for i, line in enumerate(lines):
        crop_img = []
        items = line.split(' ')
        fileName = items[0]
        x = int(items[1])
        y = int(items[2])
        label = items[3]
        w = h = 32
        img = cv2.imread(fileName)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newXStart = x-w/2
        newYStart = y-h/2
        newXEnd = x+w/2
        newYEnd = y+h/2
        # check if the range of window slides out of image
        # img.shape gives height, width, channels of image
        if newXStart < 0 or newYStart < 0 or newXEnd > img.shape[1] or newYEnd > img.shape[0]:
            continue
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        # Example crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
        crop_img = img[newYStart:newYEnd, newXStart:newXEnd]

        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(crop_img.astype(np.uint8), interpolation='none', cmap='gray')
        a.set_title('Before '+label)

        scale_img = cv2.resize(crop_img, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
        a = fig.add_subplot(1, 2, 2)
        plt.imshow(scale_img.astype(np.uint8), interpolation='none', cmap='gray')
        a.set_title('After '+label)

        # fn = fileName.replace('/home/karn_s/deeplearning/TIWafer/preprocessed/images/', '').replace('/', '__')
        # cv2.imwrite(os.path.join(targetDir, "Class_{}__{}".format(label, fn)), scale_img)
        # print 'saving patch {}'.format(i)
        plt.show()

    print 'success in cropping resizing {} patches'.format(i)