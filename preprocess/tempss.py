import os
import string
import numpy as np

train = open("/n/sd3/feng/data/scripts/IEMOCAP_train.csv","w+")
test = open("/n/sd3/feng/data/scripts/IEMOCAP_test.csv","w+")
wordid = open("/n/sd3/feng/data/word.id","r")
htkpos = "/n/sd3/feng/data/IEMOCAP_htk/"
lines = wordid.readlines()
#lines = np.array(lines)
#print(lines)
count = 0
hap = 0
neu = 0
sad = 0
ang = 0


for i in range(1,5):
    root = "/n/sd3/feng/data/IEMOCAP_full_release/Session" + str(i) + \
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
                    if "sad" in emotion or "fru" in emotion or "fea" in emotion or "dis" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                    #if "ang" in emotion:
                    #    name_list.append(temp)
                    #    emotion_list.append(emotion)
                    #    ang += 1
                    #print(emotion)
                line = f.readline()

    root = "/n/sd3/feng/data/IEMOCAP_full_release/Session" + str(i) + \
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
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ"+remove,"abcdefghijklmnopqrstuvwxyz"+" "*len(remove))

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
                    word = word.strip()+"\n"
                    if word not in lines or word == "\n" or word == " ":
                        continue
                        #print(word)
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
    root = "/n/sd3/feng/data/IEMOCAP_full_release/Session" + str(i) + \
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
                    if "sad" in emotion or "fru" in emotion or "fea" in emotion or "dis" in emotion:
                        name_list.append(temp)
                        emotion_list.append("1")
                        sad += 1
                    if "neu" in emotion:
                        name_list.append(temp)
                        emotion_list.append("2")
                        neu += 1
                    #if "ang" in emotion:
                    #    name_list.append(temp)
                    #    emotion_list.append(emotion)
                    #    ang += 1
                    #print(emotion)
                line = f.readline()

    root = "/n/sd3/feng/data/IEMOCAP_full_release/Session" + str(i) + \
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
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ"+remove,"abcdefghijklmnopqrstuvwxyz"+" "*len(remove))

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
                    word = word.strip()+"\n"
                    if word not in lines or word == "\n" or word == " ":
                        continue
                        #print(word)
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
print(hap)
print(neu)
print(sad)

wordid.close()
train.close()
test.close()
