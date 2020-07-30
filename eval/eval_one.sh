CDIVE=0
set=4
#for i in {1..61}
for i in {1..2}
do
	  CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/model/ASR_combined_self_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/model/ASR_based_self_5515.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_based_self_5515.random.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_5531.session-2.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_5531.session-2.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_5531.random.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_5515.session.8head.dist.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_dist/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_5515.session.8head.dist.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_base_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_base_5531.session.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_base_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_base_5531.random.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_5515.session.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_5531.random.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_5515.session.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_5531.random.8head.$set/results$i.txt
#
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_pip_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_pip_5531.session.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_pip_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_pip_5531.random.8head.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_pip_5531.session.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_pip_5531.session.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_pip_5531.random.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_pip_5531.random.$set/results$i.txt
done
