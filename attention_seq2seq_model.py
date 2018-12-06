# 代码参考 https://github.com/NELSONZHAO/zhihu/tree/master/mt_attention_birnn?1527252866900
# 代码解析参考 https://zhuanlan.zhihu.com/p/37290775
#
#
# ==============================================================================

import keras.backend as K
import keras
from keras.utils import plot_model
import numpy as np




class  First_Dim_Softmax(object):
    # 自定义softmax函数
    # 因为Keras提供的Softmax函数只对张量的最后一个维度进行计算，这里需要只对第一个维度进行计算
    def __call__(self,a_tensor, axis=1):
        """
        Softmax activation function.
        """
        ndim = K.ndim(a_tensor)
        if ndim == 2:
            return K.softmax(a_tensor)
        elif ndim > 2:
            e = K.exp(a_tensor - K.max(a_tensor, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

def first_dim_softmax(a_tensor, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(a_tensor)
    if ndim == 2:
        return K.softmax(a_tensor)
    elif ndim > 2:
        e = K.exp(a_tensor - K.max(a_tensor, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

class Luong_multiplicative_style_Attention_Mechanism(object):
    """
     注意力机制参考 Neural Machine Translation (seq2seq) Tutorial https://github.com/tensorflow/nmt
     Luong's additive style_Attention Mechanism 精要：
     把解码端状态向量与编码端的状态向量相乘（为了匹配维度中间可能有个矩阵）来算注意力权重大小。
     """

    pass

class Bahdanau_additive_style_Attention_Mechanism(object):
    """
    注意力机制参考 Neural Machine Translation (seq2seq) Tutorial https://github.com/tensorflow/nmt
    Bahdanau's additive style_Attention Mechanism 精要：
    把解码端状态向量与编码端的状态向量先连接再通过前馈神经网络算注意力权重大小。
    """
    def __init__(self, sequence_time_step,dense_tanh=32):
        self.repeat = keras.layers.RepeatVector(sequence_time_step)  # shape(batch_size, hidden_size) --> shape(batch_size,Tx, hidden_size)
        self.concate = keras.layers.Concatenate(axis=-1)
        self.dense_tanh = keras.layers.Dense(dense_tanh, activation="tanh")
        self.dense_relu = keras.layers.Dense(1, activation="relu")
        self.activate = keras.layers.Activation(first_dim_softmax, name='attention_weights')
        self.dot = keras.layers.Dot(axes=1)

    def __call__(self, encoder_output_sequences, s_prev):
        """
        Attention机制的实现，返回加权后的Context Vector
        @param encoder_output_sequences: BiRNN的隐层状态  # shape(batch_size, time_step, hidden_size_1)
        @param s_prev: Decoder端LSTM的上一轮隐层输出  # shape(batch_size, hidden_size_2)
        Returns:
        context: 加权后的Context Vector # shape(batch_size, hidden_siz_1)
        """
        # 将s_prev复制Tx次
        s_prev = self.repeat(s_prev)  # shape(batch_size, time_step, hidden_size_2)
        # 拼接BiRNN隐层状态与s_prev
        concat = self.concate([encoder_output_sequences, s_prev])  # shape(batch_size, time_step, hidden_size_1+ hidden_size_2)
        # 计算energies
        e = self.dense_tanh(concat)  # shape(batch_size, time_step, 32)
        energies = self.dense_relu(e)  # shape(batch_size, time_step, 1)
        # 计算weights
        alphas = self.activate(energies)  # shape(batch_size, time_step, 1)
        # 加权得到Context Vector
        context = self.dot([alphas, encoder_output_sequences])  # shape(batch_size, 1, hidden_size_1)
        return context #shape(batch_size, hidden_siz_1)


class Attention_seq2seq(object):
    '''
    self.source_vocab_length 源序列字典词个数，模型嵌入层需要
    self.target_vocab_length 目标序列字典词个数，模型输出层需要
    encoder_Bi_LSTM_units_numbers 编码端BiLSTM隐藏单元个数
    decoder_LSTM_units_numbers    解码端LSTM隐藏单元个数
    以上参数是模型结构的关键参数，若有更改，模型必须重新训练

    source_sequence_lenth 源序列长度，模型训练后可以更改
    target_sequence_lenth 目标序列长度，模型训练后可以更改
    '''
    def __init__(self,
                 source_vocab_length=None, target_vocab_length=None,
                 source_vocab_to_int=None, target_vocab_to_int=None,
                 pre_embed_word_to_vec_map=None, embed_word_dim=100):
        self.source_vocab_to_int = source_vocab_to_int #源序列的词映射整数的字典
        self.target_vocab_to_int = target_vocab_to_int #目标序列的词映射整数的字典
        self.pre_embed_word_to_vec_map = pre_embed_word_to_vec_map #预训练词向量的字典
        self.embed_word_dim = embed_word_dim #预训练词向量的维度
        if source_vocab_to_int is None and source_vocab_length is None:
            raise ValueError("must give source_vocab_to_int or source_vocab_length!")
        if self.source_vocab_to_int is None:
            self.source_vocab_length = source_vocab_length #必须明确！源序列字典词个数，模型嵌入层需要
        else:
            self.source_vocab_length = len(self.source_vocab_to_int)
        if target_vocab_to_int is None and target_vocab_length is None:
            raise ValueError("must givetarget_vocab_to_int or target_vocab_length!")
        if self.target_vocab_to_int is None:
            self.target_vocab_length = target_vocab_length #必须明确！目标序列字典词个数，模型输出层需要
        else:
            self.target_vocab_length = len(self.target_vocab_to_int)

    #构造Embedding层，并加载预训练好的词向量（非必须）
    def pretrained_embedding_layer(self):
        vocab_len = self.source_vocab_length + 1  # Keras Embedding的API要求+1
        # 如果不使用预训练词向量则随机初始化一个嵌入层
        if self.pre_embed_word_to_vec_map is None or self.source_vocab_to_int is None:
            embedding_layer = keras.layers.Embedding(vocab_len, self.embed_word_dim)
            return embedding_layer
        self.embed_word_dim = self.pre_embed_word_to_vec_map["the"].shape[0]
        # 初始化embedding矩阵
        emb_matrix = np.zeros((vocab_len, self.embed_word_dim))
        # 用词向量填充embedding矩阵
        for word, index in self.source_vocab_to_int.items():
            word_vector = self.pre_embed_word_to_vec_map.get(word, np.zeros(self.embed_word_dim))
            emb_matrix[index, :] = word_vector
        # 定义Embedding层，并指定不需要训练该层的权重
        embedding_layer = keras.layers.Embedding(vocab_len, self.embed_word_dim, trainable=False)
        # build
        embedding_layer.build((None,))
        # set weights
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer

    def create_model(self, source_sequence_lenth=None, target_sequence_lenth=None,
                     encoder_Bi_LSTM_units_numbers=None,
                     decoder_LSTM_units_numbers=None,
                     Attention_Mechanism=Bahdanau_additive_style_Attention_Mechanism):
        """
        构造模型
        @param source_sequence_lenth: 输入序列的长度 int
        @param target_sequence_lenth: 输出序列的长度 int
        @param Attention_Mechanism: 注意力机制 class
        @param encoder_Bi_LSTM_units_numbers:  编码端 Bi-LSTM 的隐藏状态单元个数（默认输出维度会乘2）int
        @param decoder_LSTM_units_numbers:  解码端 LSTM 的隐藏状态单元个数 int
        重要说明：
        encoder_Bi_LSTM_units_numbers 或 decoder_LSTM_units_numbers 若改变，需要重新训练模型！
        """
        embedding_layer = self.pretrained_embedding_layer() #嵌入层
        one_step_attention = Attention_Mechanism(source_sequence_lenth)  #注意力层
        reshape = keras.layers.Reshape((1, self.target_vocab_length))
        concate = keras.layers.Concatenate(axis=-1) #连接层
        decoder_LSTM_cell = keras.layers.LSTM(decoder_LSTM_units_numbers, return_state=True) #解码层
        output_layer = keras.layers.Dense(self.target_vocab_length, activation=first_dim_softmax) #输出层

        # 定义输入层
        X = keras.layers.Input(shape=(source_sequence_lenth,))  # shape(batch_size, source_sequence_lenth)
        # Embedding层
        embed = embedding_layer(X)  # shape(batch_size, source_sequence_lenth, embedding_size=100)

        # Decoder端LSTM的初始状态  # shape(batch_size, decoder_LSTM_units_numbers)
        decoder_lstm_initial_H_state = keras.layers.Input(shape=(decoder_LSTM_units_numbers,),
                                                       name='decoder_lstm_initial_H_state')
        decoder_lstm_initial_C_state = keras.layers.Input(shape=(decoder_LSTM_units_numbers,),
                                                        name='decoder_lstm_initial_C_state')

        # Decoder端LSTM的初始输入 shape(batch_size, target_vocab_size)
        decoder_lstm_time0_output = keras.layers.Input(shape=(self.target_vocab_length,), name='decoder_lstm_time0_output')
        decoder_lstm_output = reshape(decoder_lstm_time0_output)  # shape(batch_size, 1, target_vocab_size)
        decoder_lstm_H_state = decoder_lstm_initial_H_state
        decoder_lstm_C_state = decoder_lstm_initial_C_state

        # 模型输出列表，用来存储翻译的结果
        outputs = []
        # 定义Bi-LSTM   #shape(batch_size, Tx, 2*encoder_Bi_LSTM_units_numbers)
        encoder_output_sequences_hidden_state = keras.layers.Bidirectional(
            keras.layers.LSTM(encoder_Bi_LSTM_units_numbers, return_sequences=True))(embed)
        # Decoder端，迭代Ty轮，每轮生成一个翻译结果
        for t in range(target_sequence_lenth):
            # 获取Context Vector
            # shape(batch_size,1, 2*encoder_Bi_LSTM_units_numbers)
            attention_encoder_hidden_context = one_step_attention(encoder_output_sequences_hidden_state, decoder_lstm_H_state)
            # 将Context Vector与上一轮的翻译结果进行concat
            # shape(batch_size,1, 2*encoder_Bi_LSTM_units_numbers+target_vocab_size)
            attention_encoder_hidden_context = concate([attention_encoder_hidden_context, reshape(decoder_lstm_output)])
            # lstm1, state_h, state_c = LSTM(return_state=True)
            #lstm1 和 state_h 结果都是 hidden state。在这种参数设定下，它们俩的值相同。
            # 都是最后一个时间步的 hidden state。 state_c 是最后一个时间步 cell state结果。
            decoder_lstm_H_state, _, decoder_lstm_C_state = decoder_LSTM_cell(
                attention_encoder_hidden_context, initial_state=[decoder_lstm_H_state, decoder_lstm_C_state])
            # 将LSTM的输出结果与全连接层链接
            decoder_lstm_output = output_layer(decoder_lstm_H_state)  # shape(batch_size, target_vocab_size)
            # 存储输出结果
            outputs.append(decoder_lstm_output)
        model = keras.Model([X, decoder_lstm_initial_H_state, decoder_lstm_initial_C_state, decoder_lstm_time0_output], outputs)
        return model


class Translation(object):
    def __init__(self,
                 model=None,
                 source_vocab_to_int=None,
                 target_vocab_to_word=None,
                 source_sequence_lenth=None,
                 decoder_LSTM_units_numbers=None):
        self.source_vocab_to_int = source_vocab_to_int
        self.target_vocab_to_word = target_vocab_to_word
        self.traslation_model = model
        self.source_sequence_lenth = source_sequence_lenth
        self.target_vocab_to_word = target_vocab_to_word
        self.decoder_LSTM_units_numbers = decoder_LSTM_units_numbers

    def traslate_a_sentence(self,source_sentence, delet_PAD=True):
        '''
        :param source_sentence str
        :return: target_sentence str
        '''
        decoder_lstm_initial_H_state = np.zeros((1, self.decoder_LSTM_units_numbers))
        decoder_lstm_initial_C_state = np.zeros((1, self.decoder_LSTM_units_numbers))
        decoder_lstm_time0_output = np.zeros((1, len(self.target_vocab_to_word)))

        # 将句子分词后转化为数字编码
        unk_idx = self.source_vocab_to_int["<UNK>"]
        word_idx_list = [self.source_vocab_to_int.get(word, unk_idx) for word in source_sentence.lower().split()]
        word_idx_list = word_idx_list + [0] * self.source_sequence_lenth
        word_idx_list = word_idx_list[:self.source_sequence_lenth]
        word_idx_array = np.array(word_idx_list) # shape(source_sequence_lenth,)
        # 翻译结果
        preds = self.traslation_model.predict([word_idx_array.reshape(-1, self.source_sequence_lenth),
                                               decoder_lstm_initial_H_state, decoder_lstm_initial_C_state,
                                               decoder_lstm_time0_output])
        predictions = np.argmax(preds, axis=-1)
        # 转换为单词
        word_list = [self.target_vocab_to_word.get(idx[0], "<UNK>") for idx in predictions]
        if delet_PAD:
            word_list = [word for word in word_list if "<PAD>" not in word]
        # 返回句子
        return " ".join(word_list)

    def traslate_batch_sentences(self,batch_sentences, delet_PAD=False):
        batch_size = len(batch_sentences)
        decoder_lstm_initial_H_state = np.zeros((batch_size, self.decoder_LSTM_units_numbers))
        decoder_lstm_initial_C_state = np.zeros((batch_size, self.decoder_LSTM_units_numbers))
        decoder_lstm_time0_output = np.zeros((batch_size, len(self.target_vocab_to_word)))

        # 将句子分词后转化为数字编码
        unk_idx = self.source_vocab_to_int["<UNK>"]
        source_sentences_list = []
        for source_sentence in batch_sentences:
            word_idx_list = [self.source_vocab_to_int.get(word, unk_idx) for word in source_sentence.lower().split()]
            word_idx_list = word_idx_list + [0] * self.source_sequence_lenth
            word_idx_list = word_idx_list[:self.source_sequence_lenth]
            word_idx_array = np.array(word_idx_list) # shape(source_sequence_lenth,)
            source_sentences_list.append(word_idx_array)
        sentences_array = np.array(source_sentences_list)
        sentences_array = sentences_array.reshape(batch_size, self.source_sequence_lenth)
        # 翻译结果 list 含有 time_step 个  (batch_size, target_vocab_length)
        preds = self.traslation_model.predict([sentences_array,
                                               decoder_lstm_initial_H_state,
                                               decoder_lstm_initial_C_state,
                                               decoder_lstm_time0_output])

        predictions = np.stack(preds, axis=0) #shape(time_step, batch_size, target_vocab_length)
        predictions = predictions.swapaxes(0,1) #shape(batch_size,time_step, target_vocab_length)
        predictions = np.argmax(predictions, axis=-1) #shape(batch_size,time_step)
        predicted_target_sentences_list = []
        for predicted_sentence in predictions:
            # 转换为单词
            word_list = [self.target_vocab_to_word.get(idx, "<UNK>") for idx in predicted_sentence]
            if delet_PAD:
                word_list = [word for word in word_list if "<PAD>" not in word]
            a_sentence_str = " ".join(word_list)
            predicted_target_sentences_list.append(a_sentence_str)
        return predicted_target_sentences_list

def train_model_from_scratch(source_vocab_length=229,target_vocab_length=358,
                             source_sequence_lenth=20, target_sequence_lenth=25,
                             encoder_Bi_LSTM_units_numbers=32,decoder_LSTM_units_numbers=128):

    model_name = "attention_seq2seq_model_from_scratch"
    seq2seq_model = Attention_seq2seq(source_vocab_length=source_vocab_length,
                                      target_vocab_length=target_vocab_length)
    model = seq2seq_model.create_model(source_sequence_lenth=source_sequence_lenth,
                                       target_sequence_lenth=target_sequence_lenth,
                                       encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                       decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)

    model.summary()
    # 绘制模型图像
    plot_model(model, to_file="{}.png".format(model_name), show_shapes=True)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model, model_name


def train_model_include_Pre_trained_Word_Vector(source_vocab_to_int=None,word_to_vec_map=None,embed_word_dim=100,
                                                source_sequence_lenth=20, target_sequence_lenth=25,
                                                target_vocab_length=None,
                                                encoder_Bi_LSTM_units_numbers=32, decoder_LSTM_units_numbers=128):
    model_name = "attention_seq2seq_model_include_Pre_trained_Word_Vector"
    seq2seq_model = Attention_seq2seq(source_vocab_to_int=source_vocab_to_int,
                                      target_vocab_length=target_vocab_length,
                                      pre_embed_word_to_vec_map=word_to_vec_map,
                                      embed_word_dim=embed_word_dim)

    model = seq2seq_model.create_model(source_sequence_lenth=source_sequence_lenth,
                                       target_sequence_lenth=target_sequence_lenth,
                                       encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                       decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)
    model.summary()
    # 绘制模型图像
    plot_model(model, to_file="{}.png".format(model_name), show_shapes=True)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model, model_name



def translation_mode(model_weights_file=None,
                     source_vocab_to_int=None,target_vocab_to_word=None,
                     source_sequence_lenth=20,target_sequence_lenth=25,
                     source_vocab_length=229,target_vocab_length=358,
                     encoder_Bi_LSTM_units_numbers=32,
                     decoder_LSTM_units_numbers=128):
    source_sequence_lenth = source_sequence_lenth #源序列长度不影响模型运行
    target_sequence_lenth = target_sequence_lenth #目标序列长度不影响模型运行
    source_vocab_length = source_vocab_length #源序列词典含有单词个数，若修改需要重新训练模型
    target_vocab_length = target_vocab_length #目标序列词典含有单词个数，若修改需要重新训练模型

    seq2seq_model = Attention_seq2seq(source_vocab_length=source_vocab_length,
                                      target_vocab_length=target_vocab_length)

    model = seq2seq_model.create_model(source_sequence_lenth=source_sequence_lenth,
                                       target_sequence_lenth=target_sequence_lenth,
                                       encoder_Bi_LSTM_units_numbers=encoder_Bi_LSTM_units_numbers,
                                       decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)

    model.load_weights(model_weights_file)

    translate_tool = Translation(model=model,
                                 source_vocab_to_int=source_vocab_to_int,
                                 target_vocab_to_word=target_vocab_to_word,
                                 source_sequence_lenth=source_sequence_lenth,
                                 decoder_LSTM_units_numbers=decoder_LSTM_units_numbers)
    return  translate_tool
