# -*- coding:utf-8 -*-
# Time   : 2021/6/26 19:01
# Email  : 15602409303@163.com
# Author : Zhou Yang
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import RFE, RFECV
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

train_X = ''
train_y = ''
test_X = ''

lgb_model = lgb.LGBMRegressor(eval_metric='mae', random_state=666)

rfecv = RFECV(
    estimator=lgb_model,
    cv=5,
    step=1,  # 每步删除的特征个数
    scoring='neg_mean_absolute_error',  # 打分函数
    n_jobs=-1
)
rfecv.fit(train_X, train_y)

print(rfecv.n_features_)  # 选中的特征个数
print(rfecv.ranking_)  # 特征排名

feats = list(np.array(train_X.columns)[rfecv.support_])  # 选中的特征
print(feats)

# RFE
selector = RFE(estimator=lgb_model, n_features_to_select=4, step=1)
# 与RFECV不同，此处RFE函数需要用户定义选择的变量数量，此处设置为选择4个最好的变量，每一步我们仅删除一个变量

selector = selector.fit(train_X, train_y)  # 在训练集上训练

transformed_train = train_X[:, selector.support_]  # 转换训练集
assert np.array_equal(transformed_train, train_X[:, [0, 5, 6, 7]])  # 选择了第一个，第六个，第七个及第八个变量

transformed_test = train_X[:, selector.support_]  # 转换训练集
assert np.array_equal(transformed_test, test_X[:, [0, 5, 6, 7]])  # 选择了第一个，第六个，第七个及第八个变量


# RFECV
clf = ExtraTreesRegressor(n_estimators=25)
selector = RFECV(estimator=clf, step=1, cv=5) # 使用5折交叉验证
# 每一步我们仅删除一个变量
selector = selector.fit(train_X, train_y)

transformed_train = train_X[:, selector.support_]  # 转换训练集
assert np.array_equal(transformed_train, train_X)  # 选择了所有的变量

transformed_test = test_X[:, selector.support_]  # 转换训练集
assert np.array_equal(transformed_test, test_X)  # 选择了所有的变量


# 前向特征选择
forward_model = SequentialFeatureSelector(
    RandomForestRegressor(),
    forward=True,
    verbose=2,
    cv=5,
    n_jobs=-1,
    scoring='r2'
)
lgb_model.fit(train_X, train_y)
print(lgb_model.k_feature_idx_)
print(lgb_model.k_feature_names_)


# 后向特征选择
backward_model = SequentialFeatureSelector(
    RandomForestRegressor(),
    k_features=10,
    forward=False,
    verbose=2,
    cv=5,
    n_jobs=-1,
    scoring='r2'
)
backward_model.fit(train_X, train_y)
print(backward_model.k_feature_idx_)
print(train_X.columns[list(backward_model.k_feature_idx_)])


#
emodel = ExhaustiveFeatureSelector(
    RandomForestRegressor(),
    min_features=1,
    max_features=5,
    scoring='r2',
    n_jobs=-1
)
miniData = train_X[train_X.columns[list(backward_model.k_feature_idx_)]]
emodel.fit(miniData, train_y)
print(emodel.best_idx_)
print(miniData.columns[list(emodel.best_idx_)])
