import numpy as np
import pickle
import tqdm
import os

max_source_sentence_length = 20
max_target_sentence_length = 25

def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    对文本句子进行数字编码

    @param sentence: 一个完整的句子，str类型
    @param map_dict: 单词到数字的映射，dict
    @param max_length: 句子的最大长度
    @param is_target: 是否为目标语句。在这里要区分目标句子与源句子，因为对于目标句子（即翻译后的句子）我们需要在句子最后增加<EOS>
    """

    # 用<PAD>填充整个序列
    text_to_idx = []
    # unk index
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果是输入源文本
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 否则，对于输出目标文本需要做<EOS>的填充最后
    else:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 如果超长需要截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 如果不够则增加<PAD>
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx

# English source data
with open(os.path.join("data","small_vocab_en"), "r", encoding="utf-8") as f:
    source_text = f.read()
# French target data
with open(os.path.join("data","small_vocab_fr"), "r", encoding="utf-8") as f:
    target_text = f.read()

view_sentence_range = (0, 10)

# 下面这是对原始文本按照空格分开，这样就可以查看原始文本中到底包含了多少个单词
print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

# 按照换行符将原始文本分割成句子
print("-"*5 + "English Text" + "-"*5)
sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

print()
print("-"*5 + "French Text" + "-"*5)
sentences = target_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
print('Max number of words in a sentence: {}'.format(np.max(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

# 构造英文词典
source_vocab = list(set(source_text.lower().split()))
# 构造法文词典
target_vocab = list(set(target_text.lower().split()))
print("The size of English vocab is : {}".format(len(source_vocab)))
print("The size of French vocab is : {}".format(len(target_vocab)))

# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符
# 构造英文映射字典
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}
# 构造法语映射词典
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}


# 对源句子进行转换 Tx = max_source_sentence_length
source_text_to_int = []
for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int,
                                          max_source_sentence_length,
                                          is_target=False))
# 对目标句子进行转换  Ty = max_target_sentence_length
target_text_to_int = []
for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int,
                                          max_target_sentence_length,
                                          is_target=True))

random_index = 77
print("-"*5 + "English example" + "-"*5)
print(source_text.split("\n")[random_index])
print(source_text_to_int[random_index])
print()
print("-"*5 + "French example" + "-"*5)
print(target_text.split("\n")[random_index])
print(target_text_to_int[random_index])


X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)

print("\nDATA shape:")
print("X_shape:\t", X.shape)
print("Y_shape:\t", Y.shape)

# 创建存储数据的文件夹
if not os.path.exists("preparing_resources"):
    os.mkdir("preparing_resources")
if not os.path.exists("tmp"):
    os.makedirs(os.path.join("tmp","checkpoints"))

# 存储预处理文件
np.savez(os.path.join("preparing_resources","prepared_data.npz"), X=X, Y=Y)
# 存储字典
with open(os.path.join("preparing_resources","en_vocab_to_int.pickle"), 'wb') as f:
    pickle.dump(source_vocab_to_int, f)
with open(os.path.join("preparing_resources","fa_vocab_to_int.pickle"), 'wb') as f:
    pickle.dump(target_vocab_to_int, f)
print("The size of source dict is : {}".format(len(source_vocab_to_int)))
print("The size of target dict is : {}".format(len(target_vocab_to_int)))