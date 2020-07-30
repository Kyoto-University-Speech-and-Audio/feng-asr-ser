import numpy as np
import random
train1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/1/IEMOCAP_train.csv","r")
test1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/1/IEMOCAP_test.csv","r")
train2 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/2/IEMOCAP_train.csv","r")
test2 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/2/IEMOCAP_test.csv","r")
train3 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/3/IEMOCAP_train.csv","r")
test3 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/3/IEMOCAP_test.csv","r")
train4 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/4/IEMOCAP_train.csv","r")
test4 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/4/IEMOCAP_test.csv","r")
train5 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/5/IEMOCAP_train.csv","r")
test5 = open("/n/work1/feng/data/scripts_4emotion_ASR_small/5/IEMOCAP_test.csv","r")

train1_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/1/IEMOCAP_train.csv","w+")
test1_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/1/IEMOCAP_test.csv","w+")
train2_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/2/IEMOCAP_train.csv","w+")
test2_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/2/IEMOCAP_test.csv","w+")
train3_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/3/IEMOCAP_train.csv","w+")
test3_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/3/IEMOCAP_test.csv","w+")
train4_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/4/IEMOCAP_train.csv","w+")
test4_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/4/IEMOCAP_test.csv","w+")
train5_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/5/IEMOCAP_train.csv","w+")
test5_1 = open("/n/work1/feng/data/scripts_4emotion_ASR_small_pip/5/IEMOCAP_test.csv","w+")

train = open("/n/work1/feng/data/scripts_4emotion_ASR_total/results_5531_40.txt","r")
lines = train.readlines()
for l in train1:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            train1_1.write(m)
            break
for l in train2:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            train2_1.write(m)
            break
for l in train3:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            train3_1.write(m)
            break
for l in train4:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            train4_1.write(m)
            break
for l in train5:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            train5_1.write(m)
            break

for l in test1:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            test1_1.write(m)
            break
for l in test2:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            test2_1.write(m)
            break
for l in test3:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            test3_1.write(m)
            break
for l in test4:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            test4_1.write(m)
            break
for l in test5:
    files, _ = l.strip().split('\t',1)
    for m in lines:
        if files in m:
            test5_1.write(m)
            break
