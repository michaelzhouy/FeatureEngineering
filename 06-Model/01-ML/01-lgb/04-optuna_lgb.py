# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 3:01 下午
# @Author  : Michael Zhouy
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb


if __name__ == "__main__":
    data, target = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.3)
    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        verbose_eval=100,
        early_stopping_rounds=100
    )
    pred = model.predict(x_valid, num_iteration=model.best_iteration)
    pred = np.where(pred > 0.5, 1, 0)
    acc = accuracy_score(y_valid, pred)

    best_params = model.params
    print('Best params: ', best_params)
    print('Accuracy:    ', acc)
    print('Params: ')
    for key, value in best_params.items():
        print('{}: {}'.format(key, value))
