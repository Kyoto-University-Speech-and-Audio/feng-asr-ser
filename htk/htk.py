import os

f = open("/n/work1/feng/data/IEMOCAP.scp","w")

for i in range(1,6):
    #root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    #"/dialog/transcriptions/"
    root = "/n/work1/feng/data/IEMOCAP_full_release/Session" + str(i) + \
    "/sentences/wav"

    g = os.walk(root)
    #print(name_list)
    for path, dir_list, file_list in g:
        for File in file_list:
            file_name = os.path.join(path,File)
            if ".wav" not in file_name:
                continue
            htk_name = "/n/work1/feng/data/htk/" + file_name.split("/")[-1].strip(".wav") + ".htk\n"
            print(htk_name)
            f.write(file_name + " " + htk_name)

f.close()
