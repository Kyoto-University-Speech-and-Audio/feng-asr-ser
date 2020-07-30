import numpy as np
import random

train1 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/1/IEMOCAP_train.csv","w+")
test1 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/1/IEMOCAP_test.csv","w+")
train2 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/2/IEMOCAP_train.csv","w+")
test2 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/2/IEMOCAP_test.csv","w+")
train3 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/3/IEMOCAP_train.csv","w+")
test3 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/3/IEMOCAP_test.csv","w+")
train4 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/4/IEMOCAP_train.csv","w+")
test4 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/4/IEMOCAP_test.csv","w+")
train5 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/5/IEMOCAP_train.csv","w+")
test5 = open("/n/work1/feng/data/scripts_4emotion_ASR_random_hap/5/IEMOCAP_test.csv","w+")

total_set = []
with open('/n/work1/feng/data/scripts_4emotion_ASR_total/IEMOCAP_total_hap.csv') as f:
    for line in f:
        total_set.append(line)

random.shuffle(total_set)

num_exc = int(1633 * 0.2)
num_neu = int(1707 * 0.2)
num_sad = int(1074 * 0.2)
num_ang = int(1101 * 0.2)
num = 5515
exc = 0
neu = 0
ang = 0
sad = 0

for i in range(num):
    x_file, text, emotion = total_set[i].strip().split("\t")
    emotion = int(emotion.strip())
    if emotion == 0:
        if exc < num_exc:
            exc += 1
            test1.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_exc<=exc<num_exc*2:
            exc += 1
            test2.write(total_set[i])
            train1.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_exc*2<=exc<num_exc*3:
            exc += 1
            test3.write(total_set[i])
            train2.write(total_set[i])
            train1.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_exc*3<=exc<num_exc*4:
            exc += 1
            test4.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train1.write(total_set[i])
            train5.write(total_set[i])
        elif num_exc*4<=exc:
            exc += 1
            test5.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train1.write(total_set[i])

    elif emotion == 1:
        if sad < num_sad:
            sad += 1
            test1.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_sad<=sad<num_sad*2:
            sad += 1
            test2.write(total_set[i])
            train1.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_sad*2<=sad<num_sad*3:
            sad += 1
            test3.write(total_set[i])
            train2.write(total_set[i])
            train1.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_sad*3<=sad<num_sad*4:
            sad += 1
            test4.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train1.write(total_set[i])
            train5.write(total_set[i])
        elif num_sad*4<=sad:
            sad += 1
            test5.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train1.write(total_set[i])


    elif emotion == 2:
        if neu < num_neu:
            neu += 1
            test1.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_neu<=neu<num_neu*2:
            neu += 1
            test2.write(total_set[i])
            train1.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_neu*2<=neu<num_neu*3:
            neu += 1
            test3.write(total_set[i])
            train2.write(total_set[i])
            train1.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_neu*3<=neu<num_neu*4:
            neu += 1
            test4.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train1.write(total_set[i])
            train5.write(total_set[i])
        elif num_neu*4<=neu:
            neu += 1
            test5.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train1.write(total_set[i])


    elif emotion == 3:
        if ang < num_ang:
            ang += 1
            test1.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_ang<=ang<num_ang*2:
            ang += 1
            test2.write(total_set[i])
            train1.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_ang*2<=ang<num_ang*3:
            ang += 1
            test3.write(total_set[i])
            train2.write(total_set[i])
            train1.write(total_set[i])
            train4.write(total_set[i])
            train5.write(total_set[i])
        elif num_ang*3<=ang<num_ang*4:
            ang += 1
            test4.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train1.write(total_set[i])
            train5.write(total_set[i])
        elif num_ang*4<=ang:
            ang += 1
            test5.write(total_set[i])
            train2.write(total_set[i])
            train3.write(total_set[i])
            train4.write(total_set[i])
            train1.write(total_set[i])



train1.close()
test1.close()
train2.close()
test2.close()
train3.close()
test3.close()
train4.close()
test4.close()
train5.close()
test5.close()
