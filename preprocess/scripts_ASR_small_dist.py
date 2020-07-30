import os
import string
import numpy as np
from utils import load_dat

train = open("/n/work1/feng/data/scripts_4emotion_ASR_small_dist/5/IEMOCAP_train.csv","w+")
test = open("/n/work1/feng/data/scripts_4emotion_ASR_small_dist/5/IEMOCAP_test.csv","w+")
wordid = open("/n/work1/ueno/data/librispeech/texts/word.id","r")
emotioncount = open("/n/work1/feng/data/scripts_4emotion_ASR_small_dist/5/emotioncount.txt","w+")
htkpos = "/n/work1/feng/data/IEMOCAP_ASR/"
lines = wordid.readlines()
temp = []
for l in lines:
    ww, id = l.strip().split(' ', 1)
    temp.append(ww.strip())
lines = temp
#lines = np.array(lines)
#print(lines)
count = 0
hap = 0
neu = 0
sad = 0
ang = 0
unk = 0
delete = 0
delete = [0,0,0,0]

for i in [1,2,3,4]:
    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/EmoEvaluation/"

    g = os.walk(root)
    name_list = []
    emotion_list =[]
    dist_list = []
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
                flag = 0
                if name in line:
                    _,temp,emotion,_ = line.split("\t")
                    emotion = emotion.strip()
                    temp = temp.strip()
                    if "hap" in emotion or "exc" in emotion:
                        name_list.append(temp)
                        emotion_list.append("0")
                        hap += 1
                        flag = 1
                    if "sad" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                        flag = 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                        flag = 1
                    if "ang" in emotion:
                        name_list.append(temp)
                        emotion_list.append("3")
                        ang += 1
                        flag = 1
                    #print(emotion)
                    if flag:
                        dist = ""
                        for j in range(3):
                            line = f.readline()
                            eva_name, eva_emotion, _ = line.split("\t")
                            if "C-E" in eva_name:
                                if "Sadness" in eva_emotion:
                                    dist += "1"
                                elif "Neutral" in eva_emotion:
                                    dist += "2"
                                elif "Anger" in eva_emotion:
                                    dist += "3"
                                elif "Happiness" in eva_emotion or "Excited" in eva_emotion:
                                    dist += "0"
                        dist_list.append(dist)

                line = f.readline()

    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/transcriptions/"

    g = os.walk(root)
    #print(name_list)
    for path, dir_list, file_list in g:
        for File in file_list:
            file_name = os.path.join(path,File)
            f = open(file_name, "r")

            line = f.readline()
            x_file, _, org = line.split(" ", 2)
            remove = string.punctuation.replace("'","")
            table = \
            str.maketrans("abcdefghijklmnopqrstuvwxyz"+remove,"ABCDEFGHIJKLMNOPQRSTUVWXYZ"+" "*len(remove))

            while line:
                if "Ses0" not in line:
                    line = f.readline()
                    continue
                x_file, _, sentence = line.split(' ',2)
                sentence = sentence.translate(table)
                temp = x_file
                if temp not in name_list:
                    #print(temp)
                    line = f.readline()
                    continue
                else:
                    ind_emo = name_list.index(temp)

                position = htkpos+x_file+".htk"
                cpudat = load_dat(position)
                cpudat = cpudat[:,:40]
                if cpudat.shape[0]>2000:
                    line = f.readline()
                    string_emotion = emotion_list[ind_emo]
                    delete[int(string_emotion)] += 1
                    continue

                transcripts = []
                transcripts = (htkpos+x_file+".htk\t")
                word_list = sentence.strip().split(' ')
                #print(word_list)
                i = 0
                transcripts += "2 "
                for word in word_list:
                    word = word.strip()
                    if word == "\n" or word == " " or word == "":
                        continue
                    if word not in lines:
                        ind = lines.index("<UNK>")
                        #print(word)
                        count += 1
                        unk += 1
                        transcripts += (str(ind) + " ")
                    else:
                        ind = lines.index(word)
                        #print(word)
                        count += 1
                        transcripts += (str(ind) + " ")

                transcripts += "1\t"
                transcripts += emotion_list[ind_emo]
                transcripts += "\t"
                transcripts += dist_list[ind_emo]
                transcripts += "\n"
                #print(transcripts)
                train.write(transcripts)
                train.flush()

                line = f.readline()

for i in [5]:
    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/EmoEvaluation/"

    g = os.walk(root)
    name_list = []
    emotion_list =[]
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
                flag = 0
                if name in line:
                    _,temp,emotion,_ = line.split("\t")
                    emotion = emotion.strip()
                    temp = temp.strip()
                    if "hap" in emotion or "exc" in emotion:
                        name_list.append(temp)
                        emotion_list.append("0")
                        hap += 1
                        flag = 1
                    if "sad" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                        flag = 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                        flag = 1
                    if "ang" in emotion:
                        name_list.append(temp)
                        emotion_list.append("3")
                        ang += 1
                        flag = 1
                    #print(emotion)
                if flag:
                    dist = ""
                    for j in range(3):
                        line = f.readline()
                        eva_name, eva_emotion, _ = line.split("\t")
                        if "C-E" in eva_name:
                            if "Sadness" in eva_emotion:
                                dist += "1"
                            elif "Neutral" in eva_emotion:
                                dist += "2"
                            elif "Anger" in eva_emotion:
                                dist += "3"
                            elif "Happiness" in eva_emotion or "Excited" in eva_emotion:
                                dist += "0"
                    if dist.count('0') >= 2 or dist.count('1') >= 2 or dist.count('2') >= 2 or dist.count('3') >= 2:
                        dist_list.append(dist)
                    else:
                        dist = emotion_list[-1]+emotion_list[-1]+emotion_list[-1]
                        dist_list.append(dist)

                line = f.readline()

    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/transcriptions/"

    g = os.walk(root)
    #print(name_list)
    for path, dir_list, file_list in g:
        for File in file_list:
            file_name = os.path.join(path,File)
            f = open(file_name, "r")

            line = f.readline()
            x_file, _, org = line.split(" ", 2)
            remove = string.punctuation.replace("'","")
            table = \
            str.maketrans("abcdefghijklmnopqrstuvwxyz"+remove,"ABCDEFGHIJKLMNOPQRSTUVWXYZ"+" "*len(remove))

            while line:
                if "Ses0" not in line:
                    line = f.readline()
                    continue
                x_file, _, sentence = line.split(' ',2)
                sentence = sentence.translate(table)
                temp = x_file.strip(".txt")
                if temp not in name_list:
                    #print(temp)
                    line = f.readline()
                    continue
                else:
                    ind_emo = name_list.index(temp)

                position = htkpos+x_file+".htk"
                cpudat = load_dat(position)
                cpudat = cpudat[:,:40]
                if cpudat.shape[0]>2000:
                    line = f.readline()
                    string_emotion = emotion_list[ind_emo]
                    delete[int(string_emotion)] += 1
                    continue

                transcripts = []
                transcripts = (htkpos+x_file+".htk\t")
                word_list = sentence.strip().split(' ')
                #print(word_list)
                i = 0
                transcripts += "2 "
                for word in word_list:
                    word = word.strip()
                    if word == "\n" or word == " " or word == "":
                        continue
                    if word not in lines:
                        ind = lines.index("<UNK>")
                        #print(word)
                        count += 1
                        unk += 1
                        transcripts += (str(ind) + " ")
                    else:
                        ind = lines.index(word)
                        #print(word)
                        count += 1
                        transcripts += (str(ind) + " ")

                transcripts += "1\t"
                transcripts += emotion_list[ind_emo]
                transcripts += "\t"
                transcripts += dist_list[ind_emo]
                transcripts += "\n"
                #print(transcripts)
                test.write(transcripts)
                test.flush()

                line = f.readline()

print(count)
print(unk)
print(hap)
print(neu)
print(sad)
print(ang)
print(delete[0])
print(delete[1])
print(delete[2])
print(delete[3])

emotioncount.write("neutral:"+str(neu-delete[2])+"\n")
emotioncount.write("happy:"+str(hap-delete[0])+"\n")
emotioncount.write("sad:"+str(sad-delete[1])+"\n")
emotioncount.write("angry:"+str(ang-delete[3])+"\n")

wordid.close()
train.close()
test.close()
