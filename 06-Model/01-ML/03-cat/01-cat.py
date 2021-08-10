# -*- coding:utf-8 -*-
# Time   : 2021/6/26 19:35
# Author : Michael_Zhouy
from catboost import CatBoostRegressor


def cat_train(x_train, y_train, x_valid, y_valid, cat_cols):
    model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='GPU',
        devices='1,2,3',
        random_seed=2021
    )
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=150,
        verbose=200,
        cat_features=cat_cols
    )

    valid_preds = model.predict(x_valid)
    return valid_preds
