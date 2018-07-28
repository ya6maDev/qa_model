# 必要なライブラリをインポート
from __future__ import print_function

import sys

import MeCab
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# word2vecモデルパス
WORD2VEC_PATH = './train_data/20170201/entity_vector/entity_vector.model.bin'

# 訓練するサンプルの数
NUM_SAMPLES = 10000

# データ・セットの最初から10000目の単語までを取り出す
MAX_WORDS = 10000

# 区切り文字
SEPALETER = '\t'

"""
ファイルに記載されているQAデータを質問と回答を分解して返す関数
@return input_texts, target_texts
"""
def get_file_data(data_path):
    input_texts = []
    target_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
        input_text, target_text = line.split(SEPALETER)
        target_text = target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts


"""
"""
def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).rstrip()


"""
word2vecの学習済み日本語モデルを読み込み返す
"""
def get_word2vec_model():
    embeddings_model = KeyedVectors.load_word2vec_format(
        WORD2VEC_PATH, binary=True)
    return embeddings_model


"""
Tokenizerを生成して、返す関数
"""
def get_tokenizer(text_list):
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts([tokenize(t) for t in text_list])

    return tokenizer


"""
シーケンスを取得して、返す関数
"""
def get_sequences(text_list):
    tokenizer = get_tokenizer(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)
    return sequences


"""
word_indexを取得して、返す関数
"""
def get_word_index(text_list):
    tokenizer = get_tokenizer(text_list)
    word_index = tokenizer.word_index
    return word_index

"""
(index, word)の辞書型オブジェクトを返す
"""
def get_index_word(token_index):
    reverse_word_index = dict(
        (i, word) for word, i in token_index.items())

    return reverse_word_index

"""
"""
def get_one_hot_encording(text_list):
    tokenizer = get_tokenizer(text_list)

    one_hot_result = tokenizer.texts_to_matrix(text_list, mode='binary')

    return one_hot_result


"""

"""
def get_embedding_matrix(word_index):
    num_words = len(word_index)
    embeddings_model = get_word2vec_model()

    embedding_matrix = np.zeros((num_words+1, 200))
    for word, i in word_index.items():
        if word in embeddings_model.index2word:
            embedding_matrix[i] = embeddings_model[word]

    return embedding_matrix
