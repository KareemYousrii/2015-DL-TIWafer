import os
import random
from PIL import Image

def getCoords(filename, requestedNumPairs, maxIter, GOODLABEL, BADLABEL):
    random.seed(1337)

    im = Image.open(filename)
    pix = im.load()

    offset = 13
    i = 0
    width = im.size[0]
    height = im.size[1]

    # Set the starting x and ycoord to ensure that each patch is complete
    numPairsGood = 0;
    numPairsBad = 0;

    result = []

    while((numPairsGood < requestedNumPairs or numPairsBad < requestedNumPairs) and i < maxIter):
        i+= 1
        xcoord = random.randint(0,width-1)
        ycoord = random.randint(0,height-1)

        if(pix[xcoord,ycoord] == (0,0,0) and numPairsGood < requestedNumPairs): # Blackareas are good

            numPairsGood += 1
            result.append({"xcoord": xcoord+offset, "ycoord": ycoord+offset, "label": GOODLABEL })

        elif (pix[xcoord,ycoord] == (255,255,255) and numPairsBad < requestedNumPairs): # White areas are bad
            numPairsBad += 1
            result.append({"xcoord": xcoord+offset, "ycoord": ycoord+offset, "label": BADLABEL })

    random.shuffle(result)

    return result

## Configuration ##
TIWAFER_DIR = "/media/schmiddi/Medien/TIWafer" # Desktop PC Dennis
#TIWAFER_DIR = "/media/schmiddi/DropDatabase/TIWafer" # Notebook Dennis
#TIWAFER_DIR = "/usr/stud/ahmedk/Documents/TIWafer" # Kareem
rootdir = TIWAFER_DIR + "/preprocessed/images"
labeldir = TIWAFER_DIR + "/preprocessed/labels"
GOODLABEL = 0
BADLABEL = 1
SEPERATOR = " "
OUTPUTFILE = "Data/cropping_input_data"
OUTPUTFILE_EXTENSION = '.txt'
MAX_SAMPLES_PER_IMG = 25
EVERY_X_PATCH_IN_TEST = 5
MAX_ITER = 50000

## Creating the two txt files
with open(OUTPUTFILE + '_train' + OUTPUTFILE_EXTENSION, 'w') as fTrain:
    with open(OUTPUTFILE + '_test' + OUTPUTFILE_EXTENSION, 'w') as fTest:
        for folder, subs, files in os.walk(rootdir):
            for filename in files:
                labelImg = os.path.join(labeldir, filename) + ".mask.png"
                if os.path.exists(labelImg):
                    originalImg = os.path.join(folder, filename)
                    i = 0
                    for coords in getCoords(labelImg, MAX_SAMPLES_PER_IMG, MAX_ITER, GOODLABEL, BADLABEL):
                        i+=1
                        if (i%EVERY_X_PATCH_IN_TEST == 0):
                            fTest.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(coords.get("label")) + "\n")
                        else:
                            fTrain.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(coords.get("label")) + "\n")

fTest.closed
fTrain.closed
