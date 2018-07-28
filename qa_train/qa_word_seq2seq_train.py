"""
QAモデルを生成する

参考としているソースコード
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""
# 必要なライブラリをインポート
from __future__ import print_function

import sys

import keras.callbacks as callbacks
import MeCab
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import LSTM, Dense, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import qa_utils as utils

# Batch size for training.
BATCH_SIZE = 64
# Number of epochs to train for.
EPOCHS = 2
# Latent dimensionality of the encoding space.
HIDDEN_UNITS = 256
# Number of samples to train on.
MAX_VOCAB_SIZE = 10000
# Path to the data txt file on disk.
DATA_PATH = 'train_data/qa.txt'
# モデルのパス
MODEL_PATH = './model'
# モデルファイル名
MODEL_FILE = MODEL_PATH + '/' + 's2s_qa' + '_epoch_' + str(EPOCHS) + '.h5'
# ログ
LOG_PATH = './log'

# 訓練用のQAテキストファイルから、QAデータを取得する
input_texts, target_texts = utils.get_file_data(DATA_PATH)

# 質問文で使用されているテキストを単語ごとに分解したリストを取得する
input_word_index = utils.get_word_index(input_texts)

# 回答文で使用されているテキストを単語ごとに分解したリストを取得する
target_word_index = utils.get_word_index(target_texts)

num_encoder_tokens = len(input_word_index) + 1
num_decoder_tokens = len(target_word_index) + 1
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# エンコーダー
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length), dtype='float32')

# デコーダー
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(utils.get_word_index([input_text])):
        encoder_input_data[i, t] = 1.
    for t, word in enumerate(utils.get_word_index([target_text])):
        decoder_input_data[i, t, target_word_index[word]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_word_index[word]] = 1.

# word2vec学習済みモデルを読み込む
embeddings_model = KeyedVectors.load_word2vec_format(
    utils.WORD2VEC_PATH, binary=True)
shared_Embedding = embeddings_model.get_keras_embedding()

# 入力シーケンスを定義して処理する
encoder_inputs = Input(shape=(None, ), name="encoder_inputs")
encoder = LSTM(HIDDEN_UNITS, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder(shared_Embedding(encoder_inputs))
# 'encoder_outputs'を破棄し、状態を保持します。
encoder_states = [state_h, state_c]

# 'encoder_states'を初期状態としてデコーダを設定する
decoder_inputs = Input(
    shape=(None, num_decoder_tokens), name="decoder_inputs")
# 完全な出力シーケンスを返すようにデコーダを設定し、同様に内部状態を返します。
# 私たちはトレーニングモデルの状態を戻しますが、推論でそれらを使用します。
decoder_lstm = LSTM(HIDDEN_UNITS, return_sequences=True,
                    return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens,
                      activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# ターンするモデルを定義する
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 訓練を開始する
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# コールバックを定義
tb_cb = callbacks.TensorBoard(log_dir=LOG_PATH,
                              histogram_freq=0,
                              batch_size=BATCH_SIZE,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)

#earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

#cbks = [tb_cb, earlystopping]
cbks = [tb_cb]

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=cbks,
                    validation_split=0.2)
# Save model
model.save(MODEL_FILE)

# 次：推論モード（サンプリング）。
# ドリルは次のとおりです。
# 1）は、入力を符号化し、初期デコーダ状態を検索する
# 2）は、この初期状態のデコーダの1つのステップを実行すると "シーケンスの開始"トークンをターゲットとして使用します。
# 出力は次のターゲットトークンになります
# 3）現在のターゲットトークンと現在の状態で繰り返します

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(HIDDEN_UNITS,))
decoder_state_input_c = Input(shape=(HIDDEN_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# シーケンスをデコードするための逆引きトークンインデックス
input_index_word = utils.get_index_word(input_word_index)
target_index_word = utils.get_index_word(target_word_index)

def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).rstrip()

def decode_sequence(input_seq):
    # 入力を状態ベクトルとして符号化する。
    states_value = encoder_model.predict(input_seq)

    # 長さ1の空のターゲットシーケンスを生成します。
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # ターゲット文字列の最初の文字に開始文字を設定。
    # target_seq[0, 0, target_word_index[utils.SEPALETER]] = 1.

    # 一連のシーケンスのサンプリングループ（簡略化のため、ここではサイズ1のバッチを想定しています）。
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # トークンのサンプル
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_index_word[sampled_token_index]
        decoded_sentence += sampled_word

        # 出口条件：最大長を打つか、停止文字を見つける。
        if (sampled_word == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # ターゲットシーケンス（長さ1）を更新します。
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 状態を更新する
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(10):
    # 1つのシーケンス（トレーニングセットの一部）デコードを試してみる
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    print('TensorBoard command : tensorboard --logdir=./logs')
