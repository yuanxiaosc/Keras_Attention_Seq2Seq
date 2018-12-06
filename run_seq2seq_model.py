import keras
import numpy as np
import pickle
import json
import os
import time
from attention_seq2seq_model import train_model_from_scratch,translation_mode,\
    train_model_include_Pre_trained_Word_Vector

# 加载预训练好的词向量
def get_word_to_vec_map(glove_file_path=None):
    if glove_file_path is None:
        return None
    with open(glove_file_path, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map

def load_dict():
    # 加载字典
    with open(os.path.join("preparing_resources", "en_vocab_to_int.pickle"), 'rb') as f:
        source_vocab_to_int = pickle.load(f)
    with open(os.path.join("preparing_resources", "fa_vocab_to_int.pickle"), 'rb') as f:
        target_vocab_to_int = pickle.load(f)
    source_vocab_to_word = {idx: word for word, idx in source_vocab_to_int.items()}
    target_vocab_to_word = {idx: word for word, idx in target_vocab_to_int.items()}
    print("The size of source dict is : {}".format(len(source_vocab_to_int)))
    print("The size of target dict is : {}".format(len(target_vocab_to_int)))
    return source_vocab_to_int, target_vocab_to_int, source_vocab_to_word, target_vocab_to_word

#加载数据和字典
def load_data_and_dict():
    '''
    :return:
    DATA shape:
    source_text_shape:	 (137861, 20) ndaary
    target_text_shape:	 (137861, 25) ndaary
    The size of English Map is : 229  dict
    The size of French Map is : 358   dict
    '''
    # 加载数据
    prepared_data = np.load(os.path.join("preparing_resources", "prepared_data.npz"))
    source_text = prepared_data['X']
    target_text = prepared_data['Y']
    print("\nDATA shape:")
    print("source_text_shape:\t", source_text.shape)
    print("target_text_shape:\t", target_text.shape)
    # 加载字典
    with open(os.path.join("preparing_resources", "en_vocab_to_int.pickle"), 'rb') as f:
        source_vocab_to_int = pickle.load(f)
    with open(os.path.join("preparing_resources", "fa_vocab_to_int.pickle"), 'rb') as f:
        target_vocab_to_int = pickle.load(f)
    print("The size of English Map is : {}".format(len(source_vocab_to_int)))
    print("The size of French Map is : {}".format(len(target_vocab_to_int)))
    return source_text, target_text, source_vocab_to_int, target_vocab_to_int

def load_data_trained_Word_Vector_and_train_model():
    # 加载预训练词向量（非必须）
    # Glove官方页面：https://nlp.stanford.edu/projects/glove/
    # Glove数据下载地址：https://link.zhihu.com/?target=http%3A//nlp.stanford.edu/data/glove.6B.zip
    glove_100d_file_path = "/home/b418/jupyter_workspace/B418_common/袁宵/data/Glove/glove.6B.100d.txt"
    word_to_vec_map = get_word_to_vec_map(glove_100d_file_path)
    #加载数据和对应的字典
    source_text, target_text, source_vocab_to_int, target_vocab_to_int = load_data_and_dict()

    source_sequence_lenth = source_text.shape[1]  # 20 源序列长度
    target_sequence_lenth = target_text.shape[1]  # 25 目标序列长度
    source_vocab_length = len(source_vocab_to_int)
    target_vocab_length = len(target_vocab_to_int) # The size of target vocab is : 358
    encoder_Bi_LSTM_units_numbers = 32  # 编码端 Bi-LSTM 的隐藏状态单元个数
    decoder_LSTM_units_numbers = 128  # 解码端 LSTM 的隐藏状态单元个数

    model, model_name = train_model_include_Pre_trained_Word_Vector(source_vocab_to_int=source_vocab_to_int,
                                                                    word_to_vec_map=word_to_vec_map,
                                                                    target_vocab_length=target_vocab_length,
                                                                    source_sequence_lenth=source_sequence_lenth,
                                                                    target_sequence_lenth=target_sequence_lenth,
                                                encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                                decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)

    # 初始化解码端LSTM初始化张量
    seq_numbers = source_text.shape[0]  # 训练样本个数
    decoder_lstm_initial_H_state = np.zeros((seq_numbers, decoder_LSTM_units_numbers))
    decoder_lstm_initial_C_state = np.zeros((seq_numbers, decoder_LSTM_units_numbers))
    decoder_lstm_time0_output = np.zeros((seq_numbers, len(target_vocab_to_int)))

    # 把目标序列转换成onehot形式
    target_text_to_onehot = np.array(
        list(map(lambda x: keras.utils.to_categorical(x, num_classes=len(target_vocab_to_int)), target_text)))
    outputs = list(target_text_to_onehot.swapaxes(0, 1))

    # 训练模型
    model.fit([source_text, decoder_lstm_initial_H_state, decoder_lstm_initial_C_state, decoder_lstm_time0_output],
              outputs, epochs=2, batch_size=1024)

    now_time = time.time()
    model_weight_file_name = "WEIGHT_{}.h5".format(model_name)
    # 保存模型参数信息
    model.save_weights(model_weight_file_name)
    model_info = {"model_name":model_name,"train_time":now_time,
                  "model_weight_file_name":model_weight_file_name,
                  "trained_Word_Vector":glove_100d_file_path,
                  "source_sequence_lenth":source_sequence_lenth,"target_sequence_lenth":target_sequence_lenth,
                  "source_vocab_length":source_vocab_length,"target_vocab_length":target_vocab_length,
                  "encoder_Bi_LSTM_units_numbers":encoder_Bi_LSTM_units_numbers,
                  "decoder_LSTM_units_numbers":decoder_LSTM_units_numbers}

    with open("model_info.json", "w") as f:
        json.dump(model_info, f)




def load_data_and_translate_from_scrach():
    #加载数据和对应的字典
    source_text, target_text, source_vocab_to_int, target_vocab_to_int = load_data_and_dict()

    source_sequence_lenth = source_text.shape[1]  # 20 源序列长度
    target_sequence_lenth = target_text.shape[1]  # 25 目标序列长度
    source_vocab_length = len(source_vocab_to_int) # The size of source vocab is : 229
    target_vocab_length = len(target_vocab_to_int) # The size of target vocab is : 358
    encoder_Bi_LSTM_units_numbers = 32  # 编码端 Bi-LSTM 的隐藏状态单元个数
    decoder_LSTM_units_numbers = 128  # 解码端 LSTM 的隐藏状态单元个数

    # 把目标序列转换成onehot形式
    target_text_to_onehot = np.array(
        list(map(lambda x: keras.utils.to_categorical(x, num_classes=len(target_vocab_to_int)), target_text)))
    outputs = list(target_text_to_onehot.swapaxes(0, 1))

    # 初始化解码端LSTM初始化张量
    seq_numbers = source_text.shape[0]  # 训练样本个数
    decoder_lstm_initial_H_state = np.zeros((seq_numbers, decoder_LSTM_units_numbers))
    decoder_lstm_initial_C_state = np.zeros((seq_numbers, decoder_LSTM_units_numbers))
    decoder_lstm_time0_output = np.zeros((seq_numbers, len(target_vocab_to_int)))


    # 获取模型结构和模型名称
    model, model_name = train_model_from_scratch(source_vocab_length=source_vocab_length,
                                                 target_vocab_length=target_vocab_length,
                                                 source_sequence_lenth=source_sequence_lenth,
                                                 target_sequence_lenth=target_sequence_lenth,
                                                 encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                                 decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)
    # 训练模型
    model.fit([source_text, decoder_lstm_initial_H_state, decoder_lstm_initial_C_state, decoder_lstm_time0_output],
              outputs, epochs=2, batch_size=1024)

    now_time = time.time()
    model_weight_file_name = "WEIGHT_{}.h5".format(model_name)
    # 保存模型参数信息
    model.save_weights(model_weight_file_name)
    model_info = {"model_name":model_name,"train_time":now_time,
                  "model_weight_file_name":model_weight_file_name,
                  "source_sequence_lenth":source_sequence_lenth,"target_sequence_lenth":target_sequence_lenth,
                  "source_vocab_length":source_vocab_length,"target_vocab_length":target_vocab_length,
                  "encoder_Bi_LSTM_units_numbers":encoder_Bi_LSTM_units_numbers,
                  "decoder_LSTM_units_numbers":decoder_LSTM_units_numbers}

    with open("model_info.json", "w") as f:
        json.dump(model_info, f)


def load_dict_and_translate():
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
    while True:
        source_sentence = input("Please input your sentences: ")
        if "exit()" in source_sentence:
            break
        target_sentence = translate_tool.traslate_a_sentence(source_sentence, delet_PAD=False)
        print(target_sentence)

if __name__=='__main__':

    is_translation_mode = True
    if is_translation_mode:
        load_dict_and_translate()

    is_load_data_and_translate_from_scrach = False
    if is_load_data_and_translate_from_scrach:
        load_data_and_translate_from_scrach()

    is_load_data_trained_Word_Vector_and_train_model = False
    if is_load_data_trained_Word_Vector_and_train_model:
        load_data_trained_Word_Vector_and_train_model()


'''
Using TensorFlow backend.
正在加载训练模型参数，模型参数如下：
{'model_name': 'attention_seq2seq_model_include_Pre_trained_Word_Vector', 'train_time': 1544089042.470125, 'model_weight_file_name': 'WEIGHT_attention_seq2seq_model_include_Pre_trained_Word_Vector.h5', 'trained_Word_Vector': '/home/b418/jupyter_workspace/B418_common/袁宵/data/Glove/glove.6B.100d.txt', 'source_sequence_lenth': 20, 'target_sequence_lenth': 25, 'source_vocab_length': 229, 'target_vocab_length': 358, 'encoder_Bi_LSTM_units_numbers': 32, 'decoder_LSTM_units_numbers': 128}
The size of source dict is : 229
The size of target dict is : 358
2018-12-06 19:53:06.278300: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Please input your sentences: california is usually quiet during march , and it is usually hot in june .
california is usually quiet during march , and it is usually hot in june .
chine est généralement agréable en mois , et il est généralement en en . . <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
Please input your sentences: china is usually dry during march , but it is nice in november .
china is usually dry during march , but it is nice in november .
chine est parfois agréable en mois , et il est généralement en en . . <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
Please input your sentences: exit()
'''