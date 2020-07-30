CDIVE=6
for set in {1..5}
do
	for i in {1..61}
	do
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_5531.session-2.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_5531.session-2.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_5531.random-2.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_5531.random-2.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combined_self_2280.impro-1.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_improvised/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combined_self_2280.impro-1.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_base_5531.session.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_base_5531.session.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_base_5531.random.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_base_5531.random.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_5531.session.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_5531.session.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_5531.random.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_5531.random.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_5531.session.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_5531.session.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_5531.random.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_5531.random.$set/results$i.txt
#
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_pip_5531.session.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_pip_5531.session.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_text_pip_5531.random.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_text_pip_5531.random.$set/results$i.txt
#		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_pip_5531.session.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_pip_5531.session.8head.$set/results$i.txt
		CUDA_VISIBLE_DEVICES=${CDIVE} python ../test.py --load_name /n/work1/feng/src/ASR_combine_pip_5531.random.8head.$set/network.epoch$i --test /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv > /n/work1/feng/src/ASR_combine_pip_5531.random.8head.$set/results$i.txt
	done
done
