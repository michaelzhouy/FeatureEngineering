# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 7:33 下午
# @Author  : Michael Zhouy
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import plot_importance





def xgb_plot_imp(model):
    plot_importance(model)
    plt.show()
    # get_score()返回特征重要性，是一个字典
    feature_importance_dict = model.get_score(importance_type='gain')
    return sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)


def xgb_train(train, valid, features):
    params = {
        'eta': 0.05,
        'max_depth': 15,
        'subsample': 0.6,
        'eval_metric': 'rmse',
        'reg_alpha': 10,
        'reg_lambda': 30,
        'nthread': 30,
        'min_child_weight': 17,
        'tree_method': 'gpu_hist',
        'gpu_id': 1,
        'random_state': 2021
    }

    dtrain = xgb.DMatrix(train[features], train['label'])
    dvalid = xgb.DMatrix(valid[features], valid['label'])
    evals = [(dtrain, 'train'), (dvalid, 'eval')]
    xgb_model = xgb.train(
        params,
        dtrain,
        4000,
        evals=evals,
        verbose_eval=200,
        early_stopping_rounds=150
    )
    preds = xgb_model.predict(dvalid, xgb_model.best_ntree_limit)
    return preds, xgb_model


def xgb_cross_valid(train, test, features):
    params = {
        'eta': 0.05,
        'max_depth': 15,
        'subsample': 0.6,
        'eval_metric': 'rmse',
        'reg_alpha': 10,
        'reg_lambda': 30,
        'nthread': 30,
        'min_child_weight': 17,
        'tree_method': 'gpu_hist',
        'gpu_id': 1,
        'random_state': 2021
    }

    folds = KFold(n_splits=5, shuffle=True, random_state=2021)
    MSEs = []

    X = train[features]
    y = train['orders_3h_15h']

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        start_time = time.time()
        print('Training on fold {}'.format(fold_n + 1))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=150,
            verbose=200
        )
        val = model.predict(X_valid)
        mse_ = mean_squared_error(y_valid, val)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time.time() - start_time))))
        if fold_n == 0:
            result = model.predict(test[features]) / 5
        else:
            result += model.predict(test[features]) / 5

    return result


def xgb_sklearn_cross_valid(train, test, features):
    params = {
        'eta': 0.05,
        'max_depth': 15,
        'subsample': 0.6,
        'eval_metric': 'rmse',
        'n_estimators': 3000,
        'reg_alpha': 10,
        'reg_lambda': 30,
        'nthread': 30,
        'min_child_weight': 17,
        'tree_method': 'gpu_hist',
        'gpu_id': 1,
        'random_state': 2021
    }

    folds = KFold(n_splits=5, shuffle=True, random_state=2021)
    MSEs = []

    X = train[features]
    y = train['label']

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        start_time = time.time()
        print('Training on fold {}'.format(fold_n + 1))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric='rmse',
            verbose=200
        )
        val = model.predict(X.iloc[valid_index])
        mse_ = mean_squared_error(y.iloc[valid_index], val)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time.time() - start_time))))
        if fold_n == 0:
            result = model.predict(test[features]) / 5
        else:
            result += model.predict(test[features]) / 5

    return result
