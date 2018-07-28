"""
モテSE QAモデルから予測結果を返す

"""

# 必要なライブラリをインポート
from __future__ import print_function

import sys

import numpy as np
from gensim.models import KeyedVectors
from keras.layers import LSTM, Dense, Input
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import qa_utils as utils

HIDDEN_UNITS = 256

# Path to the data txt file on disk.
DATA_PATH = 'train_data/qa.txt'

MODEL_FILE = './model/s2s_qa_epoch_200.h5'

def get_encoder_input_data(input_texts, target_texts):

    # 質問文で使用されているテキストを単語ごとに分解したリストを取得する
    input_word_index = utils.get_word_index(input_texts)

    # 回答文で使用されているテキストを単語ごとに分解したリストを取得する
    target_word_index = utils.get_word_index(target_texts)

    num_encoder_tokens = len(input_word_index) + 1
    num_decoder_tokens = len(target_word_index) + 1
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

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

    return encoder_input_data


"""
エンコーダーモデル、デコーダーモデルを生成する
"""


def create_model(input_texts, target_texts):
    # 質問文で使用されているテキストを単語ごとに分解したリストを取得する
    input_word_index = utils.get_word_index(input_texts)

    # 回答文で使用されているテキストを単語ごとに分解したリストを取得する
    target_word_index = utils.get_word_index(target_texts)

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
        shape=(None, len(target_word_index) + 1), name="decoder_inputs")
    # 完全な出力シーケンスを返すようにデコーダを設定し、同様に内部状態を返します。
    # 私たちはトレーニングモデルの状態を戻しますが、推論でそれらを使用します。
    decoder_lstm = LSTM(HIDDEN_UNITS, return_sequences=True,
                        return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(len(target_word_index) + 1,
                        activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # ターンするモデルを定義する
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model = load_model(MODEL_FILE)

    # モデルをコンパイルする
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    model.summary()

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
    return encoder_model, decoder_model


def decode_sequence(input_seq, target_texts, encoder_model, decoder_model):

    # 回答文で使用されているテキストを単語ごとに分解したリストを取得する
    target_word_index = utils.get_word_index(target_texts)

    # 入力を状態ベクトルとして符号化する。
    states_value = encoder_model.predict(input_seq)

    # 長さ1の空のターゲットシーケンスを生成します。
    target_seq = np.zeros((1, 1, len(target_word_index) + 1))

    target_index_word = utils.get_index_word(target_word_index)

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
                len(decoded_sentence) > max([len(txt) for txt in target_texts])):
            stop_condition = True

        # ターゲットシーケンス（長さ1）を更新します。
        target_seq = np.zeros((1, 1, len(target_word_index) + 1))
        target_seq[0, 0, sampled_token_index] = 1.

        # 状態を更新する
        states_value = [h, c]

    return decoded_sentence


def main():
    # 訓練用のQAテキストファイルから、QAデータを取得する
    _, target_texts = utils.get_file_data(DATA_PATH)

    input_texts = [sys.argv[1]]

    # エンコーダーモデル、デコーダーモデルを生成する。
    encoder_model, decoder_model = create_model(input_texts, target_texts)

    # エンコーダーモデルに入力するデータを取得する
    encoder_input_data = get_encoder_input_data(input_texts, target_texts)

    # 予測結果を取得する
    decoded_sentence = decode_sequence(
        encoder_input_data[0: 1], target_texts, encoder_model, decoder_model)

    print('-')
    print('Input sentence:', input_texts[0])
    print('Decoded sentence:', decoded_sentence)


if __name__ == "__main__":
    main()
