import pydevd
pydevd.settrace('localhost', port=51234, stdoutToServer=True, stderrToServer=True)
__author__ = 'z003fafb'

import os
from os import walk, rename

listFN = []
for (dirpath, dirnames, filenames) in walk("/home/karn_s/deeplearning/TIWafer/patches"):
    for f in filenames:
        absf= os.path.abspath(os.path.join(dirpath, f))
        newf = absf.replace('\n','')
        os.rename(absf, newf)
        listFN.append(absf)
    break

with open('patchList.txt', 'w') as f1:
    for item in listFN:
        #item = item.replace('\n','')
        vals = item.replace('/home/karn_s/deeplearning/TIWafer/patches/','').split('__')
        itemclass = vals[0].split('_')[1]
        f1.write("{} {}\n".format(item, itemclass))
        #print(item)
    f1.close

