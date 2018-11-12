import os

import numpy as np

from common.seq2seq import Seq2seq
import common.qa_utils as utils


def main():
    random_state = 42

    # ルートパスを取得する
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    np.random.seed(random_state)

    qa = Seq2seq()

    # 訓練用のQAテキストファイルから、QAデータを取得する
    data_path = root_path + '/train_data/qa.txt'
    input_texts, target_texts = utils.get_file_data(data_path)

    # モデルを作成する
    model, encoder_model, decoder_model = qa.create_model(input_texts, target_texts)

    # 訓練を開始する。
    history = qa.fit(model)

    # モデルを試す
    qa.predict(10, encoder_model, decoder_model, input_texts)


if __name__ == '__main__':
    main()
