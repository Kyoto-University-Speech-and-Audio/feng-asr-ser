#!/bin/bash

# model.word.mb40_LS_APS_SPSartificial.multi.twospeakers.sorted
i=${1}
#CUDA_VISIBLE_DEVICES=3 python test.py --load_name /n/rd24/ueno/stable_version/ASR/checkpoints.word.libri960/network.epoch$i > ASR/results$i.txt
#CUDA_VISIBLE_DEVICES=7 python test.py --load_name /n/rd24/ueno/stable_version/ASR/checkpoints.word.libri960/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_total/IEMOCAP_total_hap.csv > /n/work1/feng/data/scripts_4emotion_ASR_total/results_5531_$i.txt
#CUDA_VISIBLE_DEVICES=4 python test.py --load_name /n/work1/feng/src/combined_emotion.2/network.epoch$i
#CUDA_VISIBLE_DEVICES=4 python test.py --load_name /n/work1/feng/src/ASR_emotion.2/network.epoch$i > ASR_emotion.2/results$i.txt
#CUDA_VISIBLE_DEVICES=4 python test.py --load_name /n/rd24/ueno/stable_version/ASR/checkpoints.word.libri960/network.epoch$i > text_ASR/train$i.txt
#CUDA_VISIBLE_DEVICES=4 python test.py --load_name /n/work1/feng/src/ASR_combined.2/network.epoch$i > ASR_combined.2/results$i
#CUDA_VISIBLE_DEVICES=4 python test.py --load_name /n/work1/feng/src/ASR_base_5531.session.2/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/2/IEMOCAP_test.csv > /n/work1/feng/src/ASR_base_5531.session.2/results.txt
#CUDA_VISIBLE_DEVICES=6 python test.py --load_name /n/rd24/ueno/stable_version/ASR/checkpoints.word.libri960/network.epoch$i > text_ASR/swb$i.txt
#CUDA_VISIBLE_DEVICES=7 python test.py --load_name /n/work1/feng/src/ASR_combined_self_2280.impro.8head.5/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_improvised/5/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_2280.impro.8head.5/results.txt
CUDA_VISIBLE_DEVICES=5 python test.py --load_name /n/work1/feng/src/ASR_combined_self_2280.impro.8head.5/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_improvised/5/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_2280.impro.8head.5/results.txt
