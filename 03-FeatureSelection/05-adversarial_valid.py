# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 11:51 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


X_train = pd.DataFrame()
X_test = pd.DataFrame()

features = X_train.columns.tolist()
X_train['target'] = 1
X_test['target'] = 0
train_test = pd.concat([X_train, X_test], axis=0, ignore_index=True)
train1, test1 = train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
train_y = train1['target'].values
test_y = test1['target'].values
del train1['target'], test1['target']

if 'target' in features:
    features.remove('target')

adversarial_result = pd.DataFrame(index=train1.columns, columns=['roc'])
for i in tqdm(features):
    clf = lgb.LGBMClassifier(
        random_state=47,
        max_depth=2,
        metric='auc',
        n_estimators=1000,
        importance_type='gain'
    )
    clf.fit(
        np.array(train1[i]).reshape(-1, 1),
        train_y,
        eval_set=[(np.array(test1[i]).reshape(-1, 1), test_y)],
        early_stopping_rounds=200,
        verbose=0)
    temp_pred = clf.predict_proba(np.array(test1[i]).reshape(-1, 1))[:, 1]
    roc1 = roc_auc_score(test_y, temp_pred)
    adversarial_result.loc[i, 'roc'] = roc1

adversarial_result.sort_values('roc', ascending=False)
