# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 10:40 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from gensim.models import Word2Vec


def text_process(X_train, X_test):
    tokenizer = Tokenizer(num_words=224253)  # 元素个数
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    # 编码
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # 填充/截取为相同的长度
    X_train = pad_sequences(X_train, maxlen=256, padding='pre', truncating='pre')
    X_test = pad_sequences(X_test, maxlen=256, padding='pre', truncating='pre')

    word_index = tokenizer.word_index
    nb_words = len(word_index) + 1
    return X_train, X_test, word_index, nb_words


def w2v(data, embed_size=64):
    w2v_model = Word2Vec(
        sentences=data['tagid'].tolist(),
        vector_size=embed_size,
        window=1,
        min_count=1,
        epochs=10,
        hs=1
    )
    return w2v_model


def embedding_matrix_func(data, X_train, X_test):
    X_train, X_test, word_index, nb_words = text_process(X_train, X_test)
    w2v_model = w2v(data)
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        try:
            embedding_vector = w2v_model.wv.get_vector(word)
        except KeyError:
            continue
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def my_model(embedding_matrix, nb_words):
    embedding_input = Input(shape=(256,), dtype='int32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(
        nb_words,
        embed_size,
        input_length=256,
        weights=[embedding_matrix],
        trainable=False
    )
    embed = embedder(embedding_input)
    l = GRU(128, return_sequences=True)(embed)
    flat = BatchNormalization()(l)
    drop = Dropout(0.2)(flat)
    l2 = GRU(256)(drop)
    output = Dense(1, activation='sigmoid')(l2)
    model = Model(inputs=embedding_input, outputs=output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


def cv(X_train, y_train, X_test):
    # 五折交叉验证
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    train_pre = np.zeros([len(X_train), 1])
    test_predictions = np.zeros([len(X_test), 1])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n{}".format(fold_ + 1))
        model = my_model()
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
        bst_model_path = "./{}.h5".format(fold_)
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

        X_tra, X_val = X_train[trn_idx], X_train[val_idx]
        y_tra, y_val = y_train[trn_idx], y_train[val_idx]

        model.fit(X_tra, y_tra,
                  validation_data=(X_val, y_val),
                  epochs=100, batch_size=1000, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint])

        model.load_weights(bst_model_path)
        train_pre[val_idx] = model.predict(X_val)
        test_predictions += model.predict(X_test) / folds.n_splits

embed_size = 64
