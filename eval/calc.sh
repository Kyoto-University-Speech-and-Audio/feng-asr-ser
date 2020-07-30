set=5
#for set in {5}
#do
  for i in {1..35}
  do
    #echo "results$i.txt"
#    python ../calc_wer.py /n/work1/feng/src/ASR_base_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv --no_ASR
#	   python ../calc_wer.py /n/work1/feng/src/ASR_base_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv --no_ASR
#	python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5515.session.8head.dist.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small_dist/$set/IEMOCAP_test.csv
#    python ../calc_wer.py /n/work1/feng/src/ASR_text_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv --no_ASR
#	python ../calc_wer.py /n/work1/feng/src/ASR_based_self_5515.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv
#	python ../calc_wer.py /n/work1/feng/src/ASR_text_5515.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv --no_ASR
#	python ../calc_wer.py /n/work1/feng/src/ASR_combine_5515.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_text_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_text_pip_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_text_pip_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv --no_ASR

#    python ../calc_wer.py /n/work1/feng/src/ASR_combine_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_combine_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_combine_pip_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small_pip/$set/IEMOCAP_test.csv --no_ASR
#    python ../calc_wer.py /n/work1/feng/src/ASR_combine_pip_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap_pip/$set/IEMOCAP_test.csv --no_ASR

#    python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id
#    python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.random.8head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_random_hap/$set/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id
#    python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.8head-1.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id
#    python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_2280.impro.8head-1.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_improvised/$set/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id

	python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv
	python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.4head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv
	python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.48head.$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/$set/IEMOCAP_test.csv
  done
#done

#for i in {1..61}
#do
#  python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_5531.session.8head-2.2/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_small/2/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id
#done

#for set in {1..5}
#do 
#	for i in {1..61}
#	do
#		python ../calc_wer.py /n/work1/feng/src/ASR_combined_self_2280.1/$set/results$i.txt /n/work1/feng/data/scripts_4emotion_ASR_improvised/$set/IEMOCAP_test.csv --ignore_sos_eos -w /n/work1/ueno/data/librispeech/texts/word.id 
#	done
#done
#python ../calc_wer.py /n/work1/feng/data/scripts_4emotion_ASR_total/results_5531_40.txt /n/work1/feng/data/scripts_4emotion_ASR_total/IEMOCAP_total.csv --only_ASR
