#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
"""
モテSE QAモデルから予測結果を返す

"""

# 必要なライブラリをインポート
from __future__ import print_function

import sys

import numpy as np
from keras.layers import LSTM, Dense, Input
from keras.models import Model, load_model

import mote_qa_utils as utils

MODEL_FILE = './model/s2s_mote_qa_epoch_200.h5'


def get_encoder_input_data(input_texts, target_texts):

    # 文字列のリストを作成する
    input_characters = utils.get_characters(input_texts)
    target_characters = utils.get_characters(target_texts)

    # 質問文、回答文のインデックスを辞書型で作成する
    input_token_index = utils.get_char_index(input_characters)
    target_token_index = utils.get_char_index(target_characters)

    # エンコーダー
    encoder_input_data = np.zeros(
        (len(input_texts), max([len(txt) for txt in input_texts]), len(input_characters)), dtype='float32')

    # デコーダー
    decoder_input_data = np.zeros(
        (len(input_texts), max([len(txt) for txt in target_texts]), len(target_characters)), dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max([len(txt) for txt in target_texts]), len(target_characters)), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data


"""
エンコーダーモデル、デコーダーモデルを生成する
"""
def create_model(input_texts, target_texts):
    # 文字列のリストを作成する
    input_characters = utils.get_characters(input_texts)
    target_characters = utils.get_characters(target_texts)

    # 入力シーケンスを定義して処理する
    encoder_inputs = Input(shape=(None, len(input_characters)))
    encoder = LSTM(utils.latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # 'encoder_outputs'を破棄し、状態を保持します。
    encoder_states = [state_h, state_c]

    # 'encoder_states'を初期状態としてデコーダを設定する
    decoder_inputs = Input(shape=(None, len(target_characters)))
    # 完全な出力シーケンスを返すようにデコーダを設定し、同様に内部状態を返します。
    # 私たちはトレーニングモデルの状態を戻しますが、推論でそれらを使用します。
    decoder_lstm = LSTM(
        utils.latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(len(target_characters), activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # ターンするモデルを定義する
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model = load_model(MODEL_FILE)

    # モデルをコンパイルする
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # 次：推論モード（サンプリング）。
    # ドリルは次のとおりです。
    # 1）は、入力を符号化し、初期デコーダ状態を検索する
    # 2）は、この初期状態のデコーダの1つのステップを実行すると "シーケンスの開始"トークンをターゲットとして使用します。
    # 出力は次のターゲットトークンになります
    # 3）現在のターゲットトークンと現在の状態で繰り返します

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(utils.latent_dim,))
    decoder_state_input_c = Input(shape=(utils.latent_dim,))
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

    target_characters = utils.get_characters(target_texts)
    target_token_index = utils.get_char_index(target_characters)
    # 入力を状態ベクトルとして符号化する。
    states_value = encoder_model.predict(input_seq)

    # 長さ1の空のターゲットシーケンスを生成します。
    target_seq = np.zeros((1, 1, len(target_characters)))
    # ターゲット文字列の最初の文字に開始文字を設定。
    target_seq[0, 0, target_token_index[utils.sepaleter]] = 1.

    reverse_target_char_index = utils.get_reverse_char_index(
        target_token_index)

    # 一連のシーケンスのサンプリングループ（簡略化のため、ここではサイズ1のバッチを想定しています）。
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # トークンのサンプル
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 出口条件：最大長を打つか、停止文字を見つける。
        if (sampled_char == '\n' or
                len(decoded_sentence) > max([len(txt) for txt in target_texts])):
            stop_condition = True

        # ターゲットシーケンス（長さ1）を更新します。
        target_seq = np.zeros((1, 1, len(target_characters)))
        target_seq[0, 0, sampled_token_index] = 1.

        # 状態を更新する
        states_value = [h, c]

    return decoded_sentence


def main():
    print('mainが実行されました')


if __name__ == "__main__":
    main()
