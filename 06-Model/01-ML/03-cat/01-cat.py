# -*- coding:utf-8 -*-
# Time   : 2021/6/26 19:35
# Author : Michael_Zhouy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, CatBoostClassifier
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
    train_data = Pool(x_train, y_train, cat_features=cat_cols)
    valid_data = Pool(x_valid, y_valid, cat_features=cat_cols)
    model.fit(
        train_data,
        eval_set=valid_data,
        early_stopping_rounds=150,
        verbose=200
    )

    valid_preds = model.predict(x_valid)
    return valid_preds


def cat_imp(model, valid_data):
    df_imp = pd.DataFrame()
    df_imp['feats'] = model.feature_names_
    # model.feature_importances_
    df_imp['imp'] = model.get_feature_importance(valid_data, type='LossFunctionChange')
    df_imp.sort_values('imp', ascending=False, inplace=True)
    df_imp['norm_imp'] = df_imp['imp'] / df_imp['imp'].sum()
    df_imp['cum_imp'] = np.cumsum(df_imp['norm_imp'])
    return df_imp


def cat_cv(X_train, y_train, X_test, cat_cols):
    params = {
        'iterations': 100000,
        'learning_rate': 0.1,
        'depth': 10,
        'l2_leaf_reg': 3,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'devices': '3',
        'random_seed': 2021
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    train_pre = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n{}".format(fold_ + 1))

        X_tra, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_tra, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

        tra_data = Pool(X_tra, y_tra, cat_features=cat_cols)
        val_data = Pool(X_val, y_val, cat_features=cat_cols)

        cbt_model = CatBoostClassifier(**params)
        cbt_model.fit(
            tra_data,
            eval_set=val_data,
            early_stopping_rounds=150,
            verbose=50
        )

        train_pre[val_idx] = cbt_model.predict_proba(X_val)[:, 1]
        test_predictions += cbt_model.predict_proba(X_test)[:, 1] / folds.n_splits
    return test_predictions
