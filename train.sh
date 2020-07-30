CDVICE=3
#for i in {3..4}
#do
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_improvised/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_2280.impro.8head-1.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_5531.random.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_base_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_base_5531.random.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_text_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_text_5531.random.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_combine_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_combine_5531.random.8head.$i
#
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_based_self_5515.random.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_improvised/$i/IEMOCAP_train.csv --save_dir ASR_based_self_5515.impro.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$i/IEMOCAP_train.csv --save_dir ASR_text_pip_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$i/IEMOCAP_train.csv --save_dir ASR_text_pip_5531.random.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$i/IEMOCAP_train.csv --save_dir ASR_combine_pip_5531.session.8head.$i
#  CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$i/IEMOCAP_train.csv --save_dir ASR_combine_pip_5531.random.8head.$i
#done
i=5
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small_dist/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_5515.session.8head.dist.$i
#done
CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir test_code
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_based_self_5515.session.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_improvised/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_2280.impro.8head-1.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_5531.random.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_improvised/$i/IEMOCAP_train.csv --save_dir ASR_combined_self_2280.impro.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_base_5531.session.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$i/IEMOCAP_train.csv --save_dir ASR_base_5531.random.8head.$i
#CUDA_VISIBLE_DEVICES=${CDVICE} python train.py --train /n/work1/feng/data/scripts_4emotion_ASR_small/$i/IEMOCAP_train.csv --save_dir ASR_base_5531.session.8head.$i
