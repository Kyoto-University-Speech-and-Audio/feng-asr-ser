import numpy as np

def edit(word1, word2):
    m = len(word1) + 1
    n = len(word2) + 1
    dp = [[0 for i in range(n)] for j in range(m)]
    for i in range(n):
        dp[0][i] = i
    for i in range(m):
        dp[i][0] = i

    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (0 if word1[i-1]==word2[j-1] else 1))

    return dp

def getStepList(r, h ,d):
    x = len(r)
    y = len(h)
    results_stats = np.zeros((5))
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]:
            results_stats[0] += 1
            x = x-1
            y = y-1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            results_stats[3] += 1
            x = x
            y = y-1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] + 1:
            results_stats[2] += 1
            x = x-1
            y = y-1
        else:
            results_stats[1] += 1
            x = x-1
            y = y
    results_stats[4] = len(r)
    return results_stats

file = "/n/work1/feng/src/text_ASR/results35.txt"
f = open(file, "r")
#scripts = open("/n/work1/ueno/data/librispeech/eval/ref/ref.test_clean")
scripts = open("/n/work1/feng/data/scripts_4emotion_ASR/IEMOCAP_test.csv")
wordid = open("/n/work1/ueno/data/librispeech/texts/word.id")

line1 = f.readline()
line2 = scripts.readline()
lines = wordid.readlines()
temp = []
for l in lines:
    ww, id = l.strip().split(' ', 1)
    temp.append(ww.strip())
lines = temp
err = 0
total = 0
results_all = np.zeros((5))

while line1 and line2:
    #x_file1, laborg = line1.split(" ", 1)
    #print(line1)
    if len(line1.strip().split(" ",1)) == 1:
        line1 = f.readline()
        line2 = scripts.readline()
        continue
    x_file1, laborg = line1.strip().split(" ",1)
    #print(x_file1)
    #print(laborg)
    laborg = laborg.strip()
    predict = ([int(i) for i in laborg.strip().split(' ')])

    #x_file2, laborg = line2.split(" ",1)
    x_file2, laborg, labemo = line2.split("\t")
    #gt = ([int(i) for i in laborg.strip().split(' ')])
    gt = ([i for i in laborg.strip().split(' ')])
    gt_num = []
    #print(x_file2)

    for word in gt:
        word = word.strip()
        if word == "\n" or word == " " or word == "":
            continue
        if word not in lines:
            ind = 0
            #print(word)
            gt_num.append(ind)
        else:
            ind = lines.index(word)
            #print(word)
            gt_num.append(ind)

    total += len(gt)

    if x_file1.strip() == x_file2.strip():
        print(predict)
        predict = predict[1:]
        #print(len(predict))
        for i in range(len(predict)):
            if predict[i] == 1:
                predict = predict[0:i]
                break
        #print(gt_num)
        #print(predict)
        # gt_num = gt_num[1:-1]

        #err += edit(gt_num, predict)
        results_all += getStepList(gt, predict, edit(gt, predict))

    else:
        exit(1)

    line1 = f.readline()
    line2 = scripts.readline()

#print("WER of ASR: {}".format(err/total))
print('WER {}% H={}, D={}, S={}, I={}, N={}'.format(results_all[1:-1].sum()/ results_all[-1] * 100, results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]))
