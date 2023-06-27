# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 7:35 下午
# @Author  : Michael Zhouy
"""
1. Xgboost与CPU版本没有区别, 只是multi GPU需要用到dask和dask_cuda两个包
pip install dask
pip install dask_cuda
2. 坑: 遇到 ImportError: cannot import name 'dumps_msgpack' from 'distributed.protocol.core' , 版本有问题, 需指定安装版本, 命令: pip install dask==2021.4.1, 即可解决以下报错(我已解决, 之前的报错忘记截图了)
       pip install dask==2021.4.1
"""
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
n_splits = 5
folds = KFold(n_splits=n_splits, shuffle=True, random_state=2021)

X = train[features]
y = train['label']
X_test = test[features]
y_test = test['label']
MSEs = []


def train():
    # 将DataFrame转为da, chunks参数, 第1个为行数, 第2个为特征个数
    X_test_da = da.from_array(X_test, chunks=(1000, len(features)))
    # y_test的chunks参数, 第1个需要与X_test的第1个相等, 第2个不传
    y_test_da = da.from_array(y_test, chunks=(1000,))
    # 类似xgb.DMatrix()
    dtest = DaskDMatrix(client, X_test_da, y_test_da)

    # 交叉验证
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        start_time = time.time()
        print('Training on fold {}'.format(fold_n + 1))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

        # 同上
        X_train_da = da.from_array(X_train, chunks=(1000, len(features)))
        X_valid_da = da.from_array(X_valid, chunks=(1000, len(features)))
        y_train_da = da.from_array(y_train, chunks=(1000,))
        y_valid_da = da.from_array(y_valid, chunks=(1000,))

        dtrain = DaskDMatrix(client, X_train_da, y_train_da)
        dvalid = DaskDMatrix(client, X_valid_da, y_valid_da)

        evals = [(dtrain, 'train'), (dvalid, 'valid')]

        # 与CPU, 单GPU不同, 需xgb.dask.train(), 其他相同
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

        # 与CPU, 单GPU不同, 需xgb.dask.predict(), 并且需要传入client, bst, 其他相同
        valid_pred = xgb.dask.predict(client, bst, dvalid)
        mse_ = mean_squared_error(y_valid, valid_pred)
        print('MSE: {}'.format(mse_))
        MSEs.append(mse_)
        print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time.time() - start_time))))
        # print('Evaluation history:', history)

        if fold_n == 0:
            result = xgb.dask.predict(client, bst, dtest) / n_splits
        else:
            result += xgb.dask.predict(client, bst, dtest) / n_splits

    return result


# n_workers参数为使用多少个GPU
# threads_per_worker参数为每个GPU配置都是线程
# CUDA_VISIBLE_DEVICES参数为指定使用哪几个GPU
with LocalCUDACluster(n_workers=2, threads_per_worker=2, CUDA_VISIBLE_DEVICES='2, 3') as cluster:
    with Client(cluster) as client:
        result = train()
