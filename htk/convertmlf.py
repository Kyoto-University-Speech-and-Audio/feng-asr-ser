import numpy as np
import sys
import hparams as hp

name = sys.argv[1]
f = open(name.strip(), "r")
res = open(name.strip().split(".txt")[0]+".mlf", "w")
line = f.readline()
wordid = open("/n/work1/ueno/data/librispeech/texts/word.id")
words = wordid.readlines()

res.write("#!MLF!#\n")
while line:
    if len(line.strip().split(" ")) == 1:
        line = f.readline()
        continue
    file_name, laborg, labemo = line.split("\t")

    #file_name = file_name.strip().split("/")[-1].split(".")[0].strip()
    file_name = file_name.strip()
    file_name = file_name.strip().split(".")[0].strip()
    htk_name = "\"" + file_name + ".htk\"" + "\n"
    res.write(htk_name)
    laborg = laborg.strip()
    cpulab = np.array([int(i) for i in laborg.split(' ')], dtype=np.int32)
    for j in range(cpulab.shape[0]):
        word = words[cpulab[j]].split(" ")[0].strip()
        print(word)
        if word == "<sos>":
            continue
        if word == "<eos>":
            break
        res.write(word + "\n")

    res.write(".\n")

    line = f.readline()
