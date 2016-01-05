import os
import random
from PIL import Image

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
    
    if(requestedLabel == 1):
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
rootdir = "/usr/stud/ahmedk/Documents/TIWafer/preprocessed/images"
labeldir = "/usr/stud/ahmedk/Documents/TIWafer/preprocessed/labels"
GOODLABEL = 0
BADLABEL = 1
SEPERATOR = " "
OUTPUTFILE = "/usr/stud/ahmedk/Documents/cropping_input"
OUTPUTFILE_EXTENSION = '.txt'
MAX_SAMPLES_PER_IMG = 3
MAX_ITER = 50000

## Creating the two txt files
with open(OUTPUTFILE + '_good' + OUTPUTFILE_EXTENSION, 'w') as fgood:
    with open(OUTPUTFILE + '_bad' + OUTPUTFILE_EXTENSION, 'w') as fbad:
        for folder, subs, files in os.walk(rootdir):
            for filename in files:
                labelImg = os.path.join(labeldir, filename) + ".mask.png"
                if os.path.exists(labelImg):
                    originalImg = os.path.join(folder, filename)
                    for coords in getCoords(labelImg, GOODLABEL, MAX_SAMPLES_PER_IMG, MAX_ITER):
                        fgood.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(GOODLABEL) + "\n")
                        
                    for coords in getCoords(labelImg, BADLABEL, MAX_SAMPLES_PER_IMG, MAX_ITER):
                        fbad.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(BADLABEL) + "\n")

fgood.closed
fbad.closed
