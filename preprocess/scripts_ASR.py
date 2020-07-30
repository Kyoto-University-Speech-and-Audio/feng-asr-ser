import os
import string
import numpy as np

train = open("/n/work1/feng/data/scripts_4emotion_ASR/IEMOCAP_train.csv","w+")
test = open("/n/work1/feng/data/scripts_4emotion_ASR/IEMOCAP_test.csv","w+")
wordid = open("/n/work1/ueno/data/librispeech/texts/word.id","r")
emotioncount = open("/n/work1/feng/data/scripts_4emotion_ASR/emotioncount.txt","w+")
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


for i in range(1,5):
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
                if name in line:
                    _,temp,emotion,_ = line.split("\t")
                    emotion = emotion.strip()
                    temp = temp.strip()
                    if "hap" in emotion or "exc" in emotion:
                        name_list.append(temp)
                        emotion_list.append("0")
                        hap += 1
                    if "sad" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                    if "ang" in emotion:
                        name_list.append(temp)
                        emotion_list.append("3")
                        ang += 1
                    #print(emotion)
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
                transcripts += "\n"
                #print(transcripts)
                train.write(transcripts)
                train.flush()

                line = f.readline()

for i in range(5,6):
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
                if name in line:
                    _,temp,emotion,_ = line.split("\t")
                    emotion = emotion.strip()
                    temp = temp.strip()
                    if "hap" in emotion or "exc" in emotion:
                        name_list.append(temp)
                        emotion_list.append("0")
                        hap += 1
                    if "sad" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                    if "ang" in emotion:
                        name_list.append(temp)
                        emotion_list.append("3")
                        ang += 1
                    #print(emotion)
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

emotioncount.write("neutral:"+str(neu)+"\n")
emotioncount.write("happy:"+str(hap)+"\n")
emotioncount.write("sad:"+str(sad)+"\n")
emotioncount.write("angry:"+str(ang)+"\n")

wordid.close()
train.close()
test.close()
