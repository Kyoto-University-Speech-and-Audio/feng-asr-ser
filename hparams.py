## CONFIG

#train_script = '/n/work1/feng/data/scripts_4emotion_3440/IEMOCAP_train.csv'
#train_script = '/n/work1/feng/data/scripts_4emotion_ASR/IEMOCAP_train.csv'
#train_script = '/n/work1/feng/data/scripts_4emotion_ASR_nohap_pip/5/IEMOCAP_train.csv'
#train_script = '/n/work1/feng/data/scripts_4emotion_bpe/IEMOCAP_train.csv'
#train_script = '/n/work1/feng/src/text_ASR/train40.txt'
#save_dir = 'ASR_combined_bpe.1'
#save_dir = 'ASR_combine_pip.2.5'

load_checkpoints = True# True # you must use only train.py if you want to load the saved model
#load_checkpoints = False# True # you must use only train.py if you want to load the saved model
#load_checkpoints_path = "/n/rd24/ueno/stable_version/ASR/checkpoints.bpe5000.libri960"
load_checkpoints_path = "/n/rd24/ueno/stable_version/ASR/checkpoints.word.libri960/" #'/n/sd3/feng/speech2emotion/src/checkpoints.3/' # or path
#load_checkpoints_path = "/n/work1/feng/src/ASR_combined_self_5531.session.8head-1.2/"
load_checkpoints_epoch = 40 #50 #None

#test_script = '/n/work1/ueno/data/librispeech/eval/lmfblist.test_clean'
#test_script = '/n/work1/feng/data/scripts_4emotion_ASR/IEMOCAP_test.csv'
#test_script = '/n/work1/feng/data/scripts_4emotion_ASR_nohap/1/IEMOCAP_test.csv'
#test_script = '/n/work1/feng/data/scripts_4emotion_bpe/IEMOCAP_test.csv'
#test_script = '/n/work1/feng/data/scripts_4emotion_3440/IEMOCAP_test.csv'
#test_script = '/n/work1/feng/data/scripts_4emotion_ASR_total/IEMOCAP_total.csv'
#test_script = '/n/work1/feng/data/swb/swb/inputs_test.txt'

# general config
lmfb_dim = 40
#num_classes = 3440
#num_classes = 5048
num_classes = 47465
num_emotion = 4
eos_id = 1

# network config
frame_stacking = 3 # or False
num_hidden_nodes = 320
num_encoder_layer = 5
num_acencoder_layer = 2
num_baseline_nodes = 256
encoder_dropout = 0.2
encoder_type = 'None' # 'CNN', 'Wave'
decoder_type = 'Attention' #'CTC' #or 'Attention'
out_channels = 32
head = 8
self_attention_nodes = 512
#self_attention_nodes = 512

# training setting
batch_size = 20
max_epoch = 60

# inference config
max_decoder_seq_len = 200
beam_width = 4

score_func = 'log_softmax' # or 'logit' or 'softmax'

# others
nan_analyze_type = 'ignore' # 'stop'

pretrained = True
ASR = False
baseline = False
text_based = False
combined = False
combined_ASR = True
ASR_based = False
dist = False
#baseline_type = 'None'
#baseline_type = 'CNN_BLSTM'
baseline_type = 'lim_BLSTM'
attention_type = 'Self_Attention'
#attention_type = 'None'
