import numpy as np
import os

htkpos = "/n/sd3/feng/data/IEMOCAP_htk/"

for i in range(1,5):
    root = "/n/sd3/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/EmoEvaluation/"

    g = os.walk(root)
    for path, dir_list, file_list in g:
        if "Categorical" in path or "Attribute" in path:
            continue
        elif "Self-evaluation" in path:
            continue
        for File in file_list:
            file_name = os.path.join(path,File)
            #print(File, path)
            f = open(file_name, "r")
            name = File.strip(".txt")

            line = f.readline()
            while line:
                if name in line:
                    _,_,emotion,_ = line.split("\t")
                    print(emotion)
                line = f.readline()
