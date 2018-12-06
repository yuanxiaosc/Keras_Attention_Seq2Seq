import numpy as np
import random
import string
import keras
from attention_seq2seq_model import train_model_from_scratch
from attention_seq2seq_model import Translation

def get_random_word(word_length=7):
    random_word = ''.join(random.sample(string.ascii_letters + string.digits, word_length))
    return random_word

def generate_analog_data_and_dict():
    seq_numbers = 100000 #样本数
    source_vocab_length = 229 #源词典词汇数目
    target_vocab_length = 358 #目标词典词汇数目
    source_sequence_lenth = 20 #源序列长度
    target_sequence_lenth = 25 #目标序列长度
    source_text = np.random.randint(0,source_vocab_length,size=(seq_numbers, source_sequence_lenth))
    target_text = np.random.randint(0,target_vocab_length, size=(seq_numbers, target_sequence_lenth))
    source_vocab_to_int = {get_random_word():i for i in range(source_vocab_length)}
    target_vocab_to_int =  {get_random_word():i for i in range(target_vocab_length)}
    print("\nDATA shape:")
    print("source_text_shape:\t", source_text.shape)
    print("target_text_shape:\t", target_text.shape)
    print("The size of source dict is : {}".format(len(source_vocab_to_int)))
    print("The size of target dict is : {}".format(len(target_vocab_to_int)))
    return source_text, target_text, source_vocab_to_int, target_vocab_to_int


if __name__=='__main__':
    source_text, target_text, source_vocab_to_int, target_vocab_to_int = generate_analog_data_and_dict()
    target_vocab_to_word = {idx: word for word, idx in target_vocab_to_int.items()}

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
    #  加载参数
    #model.load_weights("Test_model_WEIGHT.h5")

    # 训练模型
    model.fit([source_text, decoder_lstm_initial_H_state, decoder_lstm_initial_C_state, decoder_lstm_time0_output],
              outputs, epochs=2, batch_size=1024)
    
    # 保存参数
    model.save_weights("Test_model_WEIGHT.h5")

    # 使用模型进行翻译
    translate_tool = Translation(model=model,
                                 source_vocab_to_int=source_vocab_to_int,
                                 target_vocab_to_word=target_vocab_to_word,
                                 source_sequence_lenth=source_sequence_lenth,
                                 decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)
    while True:
        source_sentence = input("Please input your sentences: ")
        target_sentence = translate_tool.traslate_a_sentence(source_sentence, delet_PAD=False)
        print(target_sentence)
        if "exit()" in source_sentence:
            break
