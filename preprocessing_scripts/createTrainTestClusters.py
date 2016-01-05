import os
import random
from PIL import Image

## Configuration ##
#TIWAFER_DIR = "/usr/stud/ahmedk/Documents/TIWafer"
WORKINGDIR = "Data"

DATA_INPUTFILE_BASE = WORKINGDIR + "/cropping_input_data"
FILE_EXTENSION = '.txt'
CLUSTERFILE = WORKINGDIR + "/output.txt"

fClusterTrain = range(8)
fClusterTest = range(8)
## Creating the two txt files
clusters = [[line.split(' ')[0],int(line.split(' ')[1][0])] for line in open(CLUSTERFILE)]

with open(DATA_INPUTFILE_BASE + '_cluster' + "0" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[0]:
 with open(DATA_INPUTFILE_BASE + '_cluster' + "0" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[0]:
  with open(DATA_INPUTFILE_BASE + '_cluster' + "1" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[1]:
   with open(DATA_INPUTFILE_BASE + '_cluster' + "1" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[1]:
    with open(DATA_INPUTFILE_BASE + '_cluster' + "2" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[2]:
     with open(DATA_INPUTFILE_BASE + '_cluster' + "2" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[2]:
      with open(DATA_INPUTFILE_BASE + '_cluster' + "3" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[3]:
       with open(DATA_INPUTFILE_BASE + '_cluster' + "3" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[3]:
        with open(DATA_INPUTFILE_BASE + '_cluster' + "4" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[4]:
         with open(DATA_INPUTFILE_BASE + '_cluster' + "4" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[4]:
          with open(DATA_INPUTFILE_BASE + '_cluster' + "5" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[5]:
           with open(DATA_INPUTFILE_BASE + '_cluster' + "5" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[5]:
            with open(DATA_INPUTFILE_BASE + '_cluster' + "6" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[6]:
             with open(DATA_INPUTFILE_BASE + '_cluster' + "6" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[6]:
              with open(DATA_INPUTFILE_BASE + '_cluster' + "7" + '_train' + FILE_EXTENSION, 'w') as fClusterTrain[7]:
               with open(DATA_INPUTFILE_BASE + '_cluster' + "7" + '_test' + FILE_EXTENSION, 'w') as fClusterTest[7]:
                with open(DATA_INPUTFILE_BASE+'_train'+FILE_EXTENSION, 'r') as fTrain:
                    with open(DATA_INPUTFILE_BASE+'_test'+FILE_EXTENSION, 'r') as fTest:
                        clusterPath = "-"
                        cluster = -1

                        # Get first URL from train
                        trainLine = fTrain.readline()
                        trainPath = trainLine.split(' ')[0]

                        testLine = fTest.readline()
                        testPath = testLine.split(' ')[0]

                        while(trainLine and testLine):
                            change = False;
                            # Find matching clusters
                            for x in clusters:
                                if(trainPath.endswith(x[0])):
                                    cluster = x[1]
                                    clusterPath = x[0]
                                    break

                            # Store all lines belonging to the file into the cluster
                            while(True):
                                if not trainPath.endswith(clusterPath):
                                    break

                                fClusterTrain[cluster].write(trainLine)

                                trainLine = fTrain.readline()
                                trainPath = trainLine.split(' ')[0]
                                change = True;

                            while(True):
                                if not testPath.endswith(clusterPath):
                                    break

                                fClusterTest[cluster].write(trainLine)

                                testLine = fTest.readline()
                                testPath = testLine.split(' ')[0]
                                change = True;

                            if not change:
                                break

#print clusters
fTrain.close()
fTest.close()
for x in fClusterTest:
    x.close()
for x in fClusterTrain:
    x.close()
