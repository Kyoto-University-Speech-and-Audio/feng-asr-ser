import numpy as np
from utils import load_dat

train = open("/n/work1/feng/data/scripts_4emotion/IEMOCAP_train.csv","r")
test = open("/n/work1/feng/data/scripts_4emotion/IEMOCAP_test.csv","r")

flag = 0
for line in train:
    x_file, laborg, emotion = line.strip().split("\t")
    f = load_dat(x_file)
    if flag == 0:
        temp = f
        flag = 1
    else:
        temp = np.concatenate((temp,f),axis=0)

print(test)
for line in test:
    x_file, laborg, emotion = line.strip().split("\t")
    f = load_dat(x_file)
    temp = np.concatenate((temp,f),axis=0)

print(temp.shape)
mean = np.mean(temp, axis=0)
var = np.var(temp, axis=0)
np.save("/n/work1/feng/src/htk/mean.npy", mean)
np.save("/n/work1/feng/src/htk/var.npy", var)
print(mean.shape)
print(var.shape)
