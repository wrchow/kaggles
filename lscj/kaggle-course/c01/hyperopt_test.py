# -*- coding: utf-8 -*-
from hyperopt import fmin, hp, rand, pyll
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd


iris = load_iris()
X = iris.data
y = iris.target

# scale
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
svc1 = SVC(random_state=0)
svc1.fit(X_std, y)
prediction1 = svc1.predict(X_std)
print('SVC 初始化系数')
print('正确率: ' + str(accuracy_score(y, prediction1)))
print('pivot_table: ')
print(pd.pivot_table(pd.DataFrame({'oldclass':y, 'fitclass':prediction1}), index=['oldclass'],
columns=['fitclass'], aggfunc=len))
print('---------------------------------------------------------')


# search parameter
def percept(args):
 global X_std, y
 svc = SVC(C=args['C'], gamma=args['gamma'], random_state=0)
 svc.fit(X_std, y)
 y_pred = svc.predict(X_std)
 return(accuracy_score(y, y_pred))
space = {'C':hp.uniform('C', 2**-5, 2**5), 'gamma':hp.uniform('gamma', 2**-5, 2**5)}
#print(pyll.stochastic.sample(space))
best = fmin(percept, space, algo=rand.suggest, max_evals=100)
print('hyperopt 寻参')
print('最优参数' + str(best))
print('正确率' + str(percept(best)))


# result
svc2 = SVC(C=best['C'], gamma=best['gamma'], random_state=0)
svc2.fit(X_std, y)
prediction2 = svc2.predict(X_std)
print('pivot_table: ')
print(pd.pivot_table(pd.DataFrame({'oldclass':y, 'fitclass':prediction2}), index=['oldclass'],
columns=['fitclass'], aggfunc=len))


"""
# 结果
SVC 初始化系数
正确率: 0.973333333333
pivot_table:
fitclass 0 1 2
oldclass
0 50.0 NaN NaN
1 NaN 48.0 2.0
2 NaN 2.0 48.0
---------------------------------------------------------
hyperopt 寻参
最优参数{'gamma': 0.5409741446902947, 'C': 6.656174558885269}
正确率 0.986666666667
pivot_table:
fitclass 0 1 2
oldclass
0 50.0 NaN NaN
1 NaN 48.0 2.0
2 NaN NaN 50.0
# SVC 在计算鸢尾花数据集上使用原始参数就已经能够达到很好的效果，参数优化也效果
提升不大
"""


