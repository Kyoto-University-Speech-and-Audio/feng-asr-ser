import argparse
import copy
import numpy as np

mode = 'file'
#mode = 'print'

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, deletion)
    return d


def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    results_stats = np.zeros((5)) # 'H', 'D', 'S', 'I', 'N'
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]:
            #results_list.append("h")
            results_stats[0] += 1
            x = x - 1
            y = y - 1
        elif x >= 1 and d[x][y] == d[x-1][y]+1:
            #results_list.append("d")
            results_stats[1] += 1
            x = x - 1
            y = y
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            #results_list.append("s")
            results_stats[2] += 1
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            results_stats[3] += 1
            #results_list.append("i")
            x = x
            y = y - 1
        else:
            print('There are some bugs')
    results_stats[4] = len(r)
    return results_stats


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    results_list = getStepList(r, h, d)

    # print the result in aligned way
    #print(r,h)
    result = float(d[len(r)][len(h)]) / len(r) * 100
    #print('WER {0:.2f}'.format(result))
    return results_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_file')
    parser.add_argument('ref_file')
    parser.add_argument('-w', '--word_list')
    parser.add_argument('--ignore_sos_eos', action='store_true')
    parser.add_argument('--no_ASR', action='store_true')
    parser.add_argument('--only_ASR', action='store_true')
    args = parser.parse_args()

    hyp_file = args.hyp_file
    ref_file = args.ref_file
    word_list = args.word_list
    ignore_sos_eos = args.ignore_sos_eos
    no_ASR = args.no_ASR
    only_ASR = args.only_ASR

    hyp_dict = {}
    ref_dict = {}

    word_dict = {}
    acc = 0
    neutral = 0
    positive = 0
    negative = 0
    ang = 0
    ang_total = 0
    neutral_total = 0
    positive_total = 0
    negative_total = 0
    total = 0
    confusion_matrix = np.zeros((4,4))
    results_name = hyp_file.strip().split('/')[-1].strip()
    set_name = hyp_file.strip().split('/')[-2].strip().strip('/')
    dest = '/n/work1/feng/res/'+set_name+ '_total_results.txt'
    if "results1.txt" in results_name:
        dest = open(dest,'w+')
    else:
        dest = open(dest,'a+')
    if word_list is not None:
        with open(word_list) as f:
            for line in f:
                word, word_id = line.strip().split(' ')
                word_dict[word_id] = word
        word_dict['<dummy>'] = '<dummy>'

    with open(ref_file) as f:
        for line in f:
            # When the decoding didn't output anything
            file_id, words, emotion = line.strip().split('\t')
            words = words.strip().split(' ')

            # When word dict exists
            #print(words)
            if no_ASR:
                continue
            if len(word_dict) != 0:
                new_words = []
                for w in words:
                    new_words.append(word_dict[w])
                words = copy.deepcopy(new_words)

            if ignore_sos_eos and '<sos>' in words:
                words.remove('<sos>')
            if ignore_sos_eos and '<eos>' in words:
                words.remove('<eos>')

            hyp_dict[file_id] = words

    with open(hyp_file) as f:
        for line in f:
            # When the decoding didn't output anything
            if only_ASR:
                file_id, words, labemo = line.strip().split('\t')
                words = words.strip().split(' ')
            elif no_ASR:
                file_id, labemo, emo  = line.strip().split('\t')
            else:
                file_id, words, labemo, emo = line.strip().split('\t')
                words = words.strip().split(' ')

            # When word dict exists
            if only_ASR == False:
                if int(labemo.strip())==2:
                    neutral_total += 1
                elif int(labemo.strip())==0:
                    positive_total += 1
                elif int(labemo.strip())==1:
                    negative_total += 1
                elif int(labemo.strip())==3:
                    ang_total += 1

                emotion = int(emo.strip())
                if int(emo.strip()) == int(labemo.strip()):
                    acc += 1
                    if emotion == 2:
                        neutral += 1
                    elif emotion == 0:
                        positive += 1
                    elif emotion == 1:
                        negative += 1
                    elif emotion == 3:
                        ang += 1
                total += 1
                confusion_matrix[int(labemo.strip()), emotion] += 1
            if no_ASR:
                continue
            if len(word_dict) != 0:
                new_words = []
                for w in words:
                    new_words.append(word_dict[w])
                words = copy.deepcopy(new_words)

            if ignore_sos_eos and '<sos>' in words:
                words.remove('<sos>')
            if ignore_sos_eos and '<eos>' in words:
                words.remove('<eos>')
            if words == []:
                words = ['<dummy>']

            ref_dict[file_id] = words

    results_all = np.zeros(5)
    #print(hyp_dict)
    #print(hyp_dict["/n/work1/feng/data/IEMOCAP_ASR/Ses05M_impro04_M041.htk"])
    #print(ref_dict["/n/work1/feng/data/IEMOCAP_ASR/Ses05M_impro04_M041.htk"])
    for k, v in hyp_dict.items():
        assert (k in ref_dict.keys()), '{} is not found in ref'.format(k)

        ref_v = ref_dict[k]
        #print(k, end=' ')
        results_all += wer(ref_v, v)

    results_all = results_all.astype(np.int32)
    if total == 0 and only_ASR == False:
        print("wrong")
        exit(1)
    if mode == 'file':
        print(results_name,file=dest)
        if no_ASR == False:
            print('WER {0:.2f}% [H={1:d}, D={2:d}, S={3:d}, I={4:d}, N={5:d}]'.format(results_all[1:-1].sum()/ results_all[-1] * 100, results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]),file=dest)
        if only_ASR == False:
            res = acc/total
            print("Accuracy of emotion\t", res, acc, total, file=dest)
            res = neutral/neutral_total
            print("neu:"+str(res), neutral, neutral_total, file=dest)
            res = positive/positive_total
            print("hap:"+str(res), positive, positive_total, file=dest)
            res = negative/negative_total
            print("sad:"+str(res), negative, negative_total, file=dest)
            print("ang:"+str(ang/ang_total), ang, ang_total, file=dest)
            print(confusion_matrix, file=dest)
    elif mode == 'print':
        print(results_name)
        if no_ASR == False:
            print('WER {0:.2f}% [H={1:d}, D={2:d}, S={3:d}, I={4:d}, N={5:d}]'.format(results_all[1:-1].sum()/ results_all[-1] * 100, results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]))
        if only_ASR == False:
            res = acc/total
            print("Accuracy of emotion\t", res, acc, total)
            res = neutral/neutral_total
            print("neu:"+str(res), neutral, neutral_total)
            res = positive/positive_total
            print("hap:"+str(res), positive, positive_total)
            res = negative/negative_total
            print("sad:"+str(res), negative, negative_total)
            print("ang:"+str(ang/ang_total), ang, ang_total)
            print(confusion_matrix)
