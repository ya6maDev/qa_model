# 必要なライブラリをインポート
import os

from flask import Flask
from flask import jsonify
from flask_cors import CORS
from keras.backend import clear_session

from common.seq2seq import Seq2seq
import common.qa_utils as utils

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    return 'Hello World!'


"""
リクエストで受け取った質問に対する回答をjson形式で返す
"""


@app.route('/qa/reply/<question>', methods=['GET'])
def reply(question=None):
    # セッションをクリアする
    clear_session()

    # インスタンスを生成する。
    qa = Seq2seq()

    # ルートパスを取得する
    root_path = utils.get_root_path()

    # 訓練用のQAテキストファイルから、QAデータを取得する
    data_path = os.path.join(root_path, 'train_data/qa.txt')
    _, target_texts = utils.get_file_data(data_path)

    input_texts = [question]

    # モデルを作成する
    model, encoder_model, decoder_model = qa.create_model(input_texts, target_texts)

    # モデルを読み込む
    model_path = root_path + '/model/s2s_qa_epoch_100.h5'
    model = qa.load_model(model_path)

    reply = qa.predict(1, encoder_model, decoder_model, input_texts)

    # json形式で質問と回答を返す
    return jsonify({'question': question, 'answer': reply})


if __name__ == '__main__':
    app.run(debug=True)
