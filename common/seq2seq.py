# 必要なライブラリをインポート
import os

import keras.callbacks as callbacks
import numpy as np
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model

import common.qa_utils as utils


class Seq2seq:
    """
    初期化をする。
    """

    def __init__(self):

        # ルートパスを取得する
        root_path = utils.get_root_path()

        self.model = None
        self.log_path = os.path.join(root_path, 'log')
        self.model_path = os.path.join(root_path, 'model')

        self.hidden_units = 256
        self.batch_size = 128
        self.epochs = 100

        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None

        self.input_word_index = None
        self.target_word_index = None

        self.num_encoder_tokens = None
        self.num_decoder_tokens = None

        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None

    """
    モデルを作成する。
    """

    def create_model(self, input_texts, target_texts):

        # 質問文で使用されているテキストを単語ごとに分解したリストを取得する
        self.input_word_index = utils.get_word_index(input_texts)

        # 回答文で使用されているテキストを単語ごとに分解したリストを取得する
        self.target_word_index = utils.get_word_index(target_texts)

        self.num_encoder_tokens = len(self.input_word_index) + 1
        self.num_decoder_tokens = len(self.target_word_index) + 1
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        # エンコーダー
        self.encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length), dtype='float32')

        # デコーダー
        self.decoder_input_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
                                           dtype='float32')
        self.decoder_target_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
                                            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, word in enumerate(utils.get_word_index([input_text])):
                self.encoder_input_data[i, t] = 1.
            for t, word in enumerate(utils.get_word_index([target_text])):
                self.decoder_input_data[i, t, self.target_word_index[word]] = 1.
                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_word_index[word]] = 1.

        # word2vec学習済みモデルを読み込む
        embeddings_model = KeyedVectors.load_word2vec_format(
            utils.WORD2VEC_PATH, binary=True)
        shared_Embedding = embeddings_model.get_keras_embedding()

        # 入力シーケンスを定義して処理する
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        encoder = Dropout(0.3)(encoder_inputs)
        encoder = LSTM(self.hidden_units, return_state=True, name="encoder_lstm")
        encoder_outputs, state_h, state_c = encoder(shared_Embedding(encoder_inputs))
        # 'encoder_outputs'を破棄し、状態を保持します。
        encoder_states = [state_h, state_c]

        # 'encoder_states'を初期状態としてデコーダを設定する
        decoder_inputs = Input(
            shape=(None, self.num_decoder_tokens), name="decoder_inputs")
        # 完全な出力シーケンスを返すようにデコーダを設定し、同様に内部状態を返します。
        # 私たちはトレーニングモデルの状態を戻しますが、推論でそれらを使用します。
        decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.hidden_units,))
        decoder_state_input_c = Input(shape=(self.hidden_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

    def fit(self, model):
        # 訓練を開始する
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # コールバックを定義
        tb_cb = callbacks.TensorBoard(log_dir=self.log_path,
                                      histogram_freq=0,
                                      batch_size=self.batch_size,
                                      write_graph=True,
                                      write_grads=False,
                                      write_images=False,
                                      embeddings_freq=0,
                                      embeddings_layer_names=None,
                                      embeddings_metadata=None)

        earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        cbks = [tb_cb, earlystopping]

        history = model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=cbks,
                            shuffle=True,
                            validation_split=0.2)
        # Save model
        model.save(self.model_path + '/' + 's2s_qa' + '_epoch_' + str(self.epochs) + '.h5')

        return history

    def predict(self, count, encoder_model, decoder_model, input_texts):

        for seq_index in range(count):
            # 1つのシーケンス（トレーニングセットの一部）デコードを試してみる
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]

            # シーケンスをデコードするための逆引きトークンインデックス
            input_index_word = utils.get_index_word(self.input_word_index)
            target_index_word = utils.get_index_word(self.target_word_index)

            # 入力を状態ベクトルとして符号化する。
            states_value = encoder_model.predict(input_seq)

            # 長さ1の空のターゲットシーケンスを生成します。
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))

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
                        len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True

                # ターゲットシーケンス（長さ1）を更新します。
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                # 状態を更新する
                states_value = [h, c]

            print('-')
            print('Input sentence:', input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)

        return decoded_sentence

    def load_model(self, model_path):
        model = load_model(model_path)
        return model
