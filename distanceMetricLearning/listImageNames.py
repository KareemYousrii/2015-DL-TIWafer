import os                                                                                                             

def list_files(dir):
    words = ['horizontal','transpose','vertical']
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                if "horizontal" in file or "transpose" in file or "vertical" in file:
                    continue
                else:
                    r.append(subdir + "/" + file)
    return r

if __name__ == '__main__':
    out = list_files("TIWafer\\preprocessed\\images\\")
    outfile = open("LISTFILE.txt", "w")
    print >> outfile, " 0\n".join(str(i) for i in out)
    outfile.close()