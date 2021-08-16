# -*- coding:utf-8 -*-
# Time   : 2021/6/26 19:35
# Author : Michael_Zhouy
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool


def cat_train(x_train, y_train, x_valid, y_valid, cat_cols):
    params = {
        'iterations': 100000,
        'learning_rate': 0.05,
        'depth': 10,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'devices': '3:4:5:6:7',
        'random_seed': 2021
    }
    model = CatBoostRegressor(**params)
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=150,
        verbose=200,
        cat_features=cat_cols
    )

    valid_preds = model.predict(x_valid)
    return valid_preds


def cat_imp(model, X, y):
    df_imp = pd.DataFrame()
    df_imp['feats'] = model.feature_names_
    # model.feature_importances_
    df_imp['imp'] = model.get_feature_importance(Pool(X, y), type='LossFunctionChange')
    df_imp.sort_values('imp', ascending=False, inplace=True)
    df_imp['norm_imp'] = df_imp['imp'] / df_imp['imp'].sum()
    df_imp['cum_imp'] = np.cumsum(df_imp['norm_imp'])
    return df_imp
