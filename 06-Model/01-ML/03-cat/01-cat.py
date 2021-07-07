# -*- coding:utf-8 -*-
# Time   : 2021/6/26 19:35
# Author : Michael_Zhouy
from catboost import CatBoostRegressor


def train(x_train, y_train, x_valid, y_valid, test, cat_cols, iterations=22000, depth=10):
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=2200,
        task_type="GPU"
    )
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid),
              early_stopping_rounds=500,
              verbose=500,
              cat_features=cat_cols)

    result = model.predict(test)
    return result