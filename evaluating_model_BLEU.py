#采用n-gram的BLEU(Bilingual Evaluation Understudy)来对翻译结果进行评估
from nltk.translate.bleu_score import sentence_bleu
import tqdm
import json
from attention_seq2seq_model import translation_mode
from run_seq2seq_model import load_dict

# English source data
with open("data/small_vocab_en", "r", encoding="utf-8") as f:
    source_text = f.read()

# French target data
with open("data/small_vocab_fr", "r", encoding="utf-8") as f:
    target_text = f.read()

with open("model_info.json") as f:
    train_info = json.load(f)

print("正在加载训练模型参数，模型参数如下：")
print(train_info)
encoder_Bi_LSTM_units_numbers = train_info['encoder_Bi_LSTM_units_numbers']
decoder_LSTM_units_numbers = train_info['decoder_LSTM_units_numbers']
model_weights_file = train_info['model_weight_file_name']
# 加载字典
source_vocab_to_int, _, _, target_vocab_to_word = load_dict()

translate_tool = translation_mode(model_weights_file=model_weights_file,
                                  source_vocab_to_int=source_vocab_to_int,
                                  target_vocab_to_word=target_vocab_to_word,
                                  source_sequence_lenth=20,
                                  target_sequence_lenth=25,
                                  source_vocab_length=len(source_vocab_to_int),
                                  target_vocab_length=len(target_vocab_to_word),
                                  encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                  decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)


source_sentences_list = source_text.split("\n")
target_sentences_list = target_text.split("\n")


Predict_sentences_list = []
Target_sentences_list = []
batch_size = 5000
for i in tqdm.tqdm(range(0,len(source_sentences_list)//batch_size+1)):
    batch_source_sentences = source_sentences_list[i*batch_size:(i+1)*batch_size]
    batch_target_sentences = target_sentences_list[i*batch_size:(i+1)*batch_size]
    batch_predict_sentences = translate_tool.traslate_batch_sentences(batch_source_sentences)
    Predict_sentences_list.extend(batch_predict_sentences)
    Target_sentences_list.extend(batch_predict_sentences)


# 存储每个句子的BLEU分数
bleu_score = []
for i in tqdm.tqdm(range(len(Predict_sentences_list))):
    # 去掉特殊字符
    a_predict_sentence = Predict_sentences_list[i].replace("<EOS>", "").replace("<PAD>", "").rstrip()
    a_target_sentence = Target_sentences_list[i]
    # 计算BLEU分数
    score = sentence_bleu(a_target_sentence.split(), a_predict_sentence.split())
    bleu_score.append(score)

print("The BLEU score on our corpus is about {}".format(sum(bleu_score) / len(bleu_score)))


'''
Using TensorFlow backend.
正在加载训练模型参数，模型参数如下：
{'model_name': 'attention_seq2seq_model_include_Pre_trained_Word_Vector', 'train_time': 1544089042.470125, 'model_weight_file_name': 'WEIGHT_attention_seq2seq_model_include_Pre_trained_Word_Vector.h5', 'trained_Word_Vector': '/home/b418/jupyter_workspace/B418_common/袁宵/data/Glove/glove.6B.100d.txt', 'source_sequence_lenth': 20, 'target_sequence_lenth': 25, 'source_vocab_length': 229, 'target_vocab_length': 358, 'encoder_Bi_LSTM_units_numbers': 32, 'decoder_LSTM_units_numbers': 128}
The size of source dict is : 229
The size of target dict is : 358
2018-12-06 19:29:26.976435: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
100%|███████████████████████████████████████████| 28/28 [07:46<00:00, 14.43s/it]
  0%|                                                | 0/137861 [00:00<?, ?it/s]/home/b418/anaconda3/envs/yuanxiao/lib/python3.6/site-packages/nltk/translate/bleu_score.py:490: UserWarning: 
Corpus/Sentence contains 0 counts of 2-gram overlaps.
BLEU scores might be undesirable; use SmoothingFunction().
  warnings.warn(_msg)
100%|██████████████████████████████████| 137861/137861 [05:11<00:00, 442.82it/s]
The BLEU score on our corpus is about 0.6051313546590134
'''