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
rootdir = "/usr/stud/ahmedk/Documents/TIWafer/preprocessed/images"
labeldir = "/usr/stud/ahmedk/Documents/TIWafer/preprocessed/labels"
GOODLABEL = 0
BADLABEL = 1
SEPERATOR = " "
OUTPUTFILE = "/usr/stud/ahmedk/Documents/cropping_input_data"
OUTPUTFILE_EXTENSION = '.txt'
MAX_SAMPLES_PER_IMG = 5
MAX_ITER = 40000

## Creating the two txt files
with open(OUTPUTFILE + OUTPUTFILE_EXTENSION, 'w') as fout:
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            labelImg = os.path.join(labeldir, filename) + ".mask.png"
            if os.path.exists(labelImg):
                originalImg = os.path.join(folder, filename)
                for coords in getCoords(labelImg, MAX_SAMPLES_PER_IMG, MAX_ITER, GOODLABEL, BADLABEL):
                    fout.write(originalImg + SEPERATOR + str(coords.get("xcoord")) + SEPERATOR + str(coords.get("ycoord")) + SEPERATOR + str(coords.get("label")) + "\n")
                    

fout.closed
