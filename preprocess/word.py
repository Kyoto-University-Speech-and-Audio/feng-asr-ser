import os
import string

wordid = open("/n/work1/feng/data/word.id","w+")

count = 3
wordid.write("<UNK>\n")
wordid.write("<eos>\n")
wordid.write("<sos>\n")
for i in range(1,6):
    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/dialog/transcriptions/"

    g = os.walk(root)
    for path, dir_list, file_list in g:
        for File in file_list:
            file_name = os.path.join(path,File)
            f = open(file_name, "r")

            line = f.readline()
            remove = string.punctuation.replace("'","")
            table = \
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ"+remove,"abcdefghijklmnopqrstuvwxyz"+" "*len(remove))

            while line:
                if "Ses0" not in line:
                    line = f.readline()
                    continue
                sentence = line.split(' ',2)
                sentence = sentence[2]
                sentence = sentence.translate(table)
                word_list = sentence.strip().split(' ')

                for word in word_list:
                    if word == " ":
                        continue
                    word = word.strip() + "\n"
                    oldwordid = open("/n/work1/feng/data/word.id", "r")
                    lines = oldwordid.readlines()
                    if word not in lines and word != "\n":
                        wordid.write(word)
                        if word == "\n" or word == "\n":
                            print(sentence)
                            print(word_list)
                            print(file_name)
                        wordid.flush()
                        count += 1
                    oldwordid.close()
            
                line = f.readline()

wordid.close()

counttxt = open("/n/word1/feng/data/wordcount.txt","w")
counttxt.write(str(count))
counttxt.close()
