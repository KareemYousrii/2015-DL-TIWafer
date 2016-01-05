import os
import random
from PIL import Image
import numpy

def getCoords(filename, requestedLabel, requestedNumPairs, maxIter):
    random.seed(1337)
    
    im = Image.open(filename)
    pix = im.load()
    
    offset = 13
    i = 0
    width = im.size[0]
    height = im.size[1]
    
    # Set the starting x and ycoord to ensure that each patch is complete
    numPairs = 0;
    
    if(requestedLabel == 0):
        rgbVal = (0,0,0)
    else:
        rgbVal = (255,255,255)
    
    result = []
    
    while(numPairs < requestedNumPairs and i < maxIter):
        i+= 1
        xcoord = random.randint(0,width-1)
        ycoord = random.randint(0,height-1)
        
        if (pix[xcoord,ycoord] == rgbVal):
            numPairs += 1
            result.append({"xcoord": xcoord+offset, "ycoord": ycoord+offset})
        
    return result

## Configuration ##
rootdir = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\preprocessed\\images"
labeldir = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\preprocessed\\labels"
GOODLABEL = 0
BADLABEL = 1
SEPERATOR = " "
OUTPUTFILE = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\cropping_input\\cropping_input"
OUTPUTGOOD = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\cropping_input\\images\\good_patches"
OUTPUTBAD = "C:\\Users\\Amir487\\PycharmProjects\\Deep Learning\\Data\\cropping_input\\images\\bad_patches"
OUTPUTFILE_EXTENSION = '.txt'
MAX_SAMPLES_PER_IMG = 3
MAX_ITER = 50000

## Creating the two txt files
with open(OUTPUTFILE + '_good' + OUTPUTFILE_EXTENSION, 'w') as fgood:
    with open(OUTPUTFILE + '_bad' + OUTPUTFILE_EXTENSION, 'w') as fbad:
        for folder, subs, files in os.walk(rootdir):
            for filename in files:

                print "Procssing file ", os.path.join(folder, filename)

                labelImg = os.path.join(labeldir, filename) + ".mask.png"
                if os.path.exists(labelImg):
                    originalImg = os.path.join(folder, filename)
                    for coords in getCoords(labelImg, GOODLABEL, MAX_SAMPLES_PER_IMG, MAX_ITER):
                        x1 = coords.get("xcoord")
                        y1 = coords.get("ycoord")
                        x = int(x1 - 13)
                        y = int(y1 - 13)
                        fgood.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(GOODLABEL) + "\n")
                        im = Image.open(originalImg)
                        im = im.crop((13,13,461,461))
                        im = numpy.array(im)
                        rect = numpy.copy(im[y:y+27,x:x+27])
                        patch = Image.fromarray(rect)
                        patch.save(OUTPUTGOOD + "\\" + filename + "_" + str(x) + "_" + str(y1) + ".png")
                        
                    for coords in getCoords(labelImg, BADLABEL, MAX_SAMPLES_PER_IMG, MAX_ITER):
                        x1 = coords.get("xcoord")
                        y1 = coords.get("ycoord")
                        x = int(x1 - 13)
                        y = int(y1 - 13)
                        fbad.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(BADLABEL) + "\n")
                        rect = numpy.copy(im[y:y+27,x:x+27])
                        patch = Image.fromarray(rect)
                        patch.save(OUTPUTBAD + "\\" + filename + "_" + str(x) + "_" + str(y) + ".png")

fgood.closed
fbad.closed