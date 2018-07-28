#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#

# 必要なライブラリをインポート
from flask import Flask, jsonify
from keras.backend import clear_session

import mote_qa_utils as utils
import predict

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

"""
リクエストで受け取った質問に対する回答をjson形式で返す
"""


@app.route('/qa/api/v1.0/answer/<question>', methods=['GET'])
def get_answer(question=None):

    # セッションをクリアする
    clear_session()

    # 訓練用のQAテキストファイルから、QAデータを取得する
    _, target_texts = utils.get_file_data()

    input_texts = [question]

    # エンコーダーモデル、デコーダーモデルを生成する。
    encoder_model, decoder_model = predict.create_model(
        input_texts, target_texts)

    # エンコーダーモデルに入力するデータを取得する
    encoder_input_data = predict.get_encoder_input_data(
        input_texts, target_texts)

    # 予測結果を取得する
    decoded_sentence = predict.decode_sequence(
        encoder_input_data[0: 1], target_texts, encoder_model, decoder_model)

    # json形式で質問と回答を返す
    return jsonify({'question': question, 'answer': decoded_sentence})


if __name__ == '__main__':
    app.run(debug=True)
