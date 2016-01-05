__author__ = 'z003fafb'
import os
import cPickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for fn in os.listdir('outImages/'):
        img_data = cPickle.load(open('outImages/'+fn, 'rb'))
        #plt.show(img_data)
        plt.imshow(img_data)
        plt.show()