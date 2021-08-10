# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 7:35 下午
# @Author  : Michael Zhouy
import time
import datetime
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from xgboost.dask import DaskDMatrix
import xgboost as xgb
from dask import array as da

train = pd.DataFrame()
test = pd.DataFrame()
features = []

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
    'gpu_id': 7,
    'random_state': 2021
}

folds = KFold(n_splits=5, shuffle=True, random_state=2021)

X = train[features]
y = train['label']
X_test = test[features]
y_test = test['label']
MSEs = []


def train():
    X_test_da = da.from_array(X_test.values, chunks=(1000, len(features)))
    y_test_da = da.from_array(y_test, chunks=(1000,))
    dtest = DaskDMatrix(client, X_test_da, y_test_da)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        start_time = time.time()
        print('Training on fold {}'.format(fold_n + 1))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

        X_train_da = da.from_array(X_train, chunks=(1000, len(features)))
        X_valid_da = da.from_array(X_valid, chunks=(1000, len(features)))
        y_train_da = da.from_array(y_train, chunks=(1000,))
        y_valid_da = da.from_array(y_valid, chunks=(1000,))

        dtrain = DaskDMatrix(client, X_train_da, y_train_da)
        dvalid = DaskDMatrix(client, X_valid_da, y_valid_da)

        evals = [(dtrain, 'train'), (dvalid, 'valid')]

        model = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=3000,
            evals=evals,
            verbose_eval=200,
            early_stopping_rounds=150
        )

        bst = model['booster']
        history = model['history']

        valid_pred = xgb.dask.predict(client, bst, dvalid)
        mse_ = mean_squared_error(y_valid, valid_pred)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time.time() - start_time))))
        # print('Evaluation history:', history)

        if fold_n == 0:
            result = xgb.dask.predict(client, bst, dtest) / 5
        else:
            result += xgb.dask.predict(client, bst, dtest) / 5

    return result


with LocalCUDACluster(n_workers=2, threads_per_worker=2, CUDA_VISIBLE_DEVICES='2, 3') as cluster:
    with Client(cluster) as client:
        result = train()
