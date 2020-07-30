root = '/n/work1/feng/res/'
eval = "ASR_combined_self_5531.session.8head-2.2"

file_name = root + eval + '_total_results.txt'
f = open(file_name,"r")
lines = f.readlines()
max = 0
pos = 0
for index,j in enumerate(lines):
    if 'Accuracy' in j:
        _, result = j.strip().split('\t')
        acc, res, total = result.strip().split(' ')
        res = int(res.strip())
        if res > max:
            max = res
            #print(max,index)
            pos = index
#dest.write(eval[eval_num].strip('.')+'\n')
for i in range(11):
    #print(i)
    print(lines[pos-2+i],end='')
f.close()
