#! -*- coding:utf-8 -*-
import xgboost as xgb
import numpy as np

# read in data
dtrain = xgb.DMatrix('./Affairs.txt.train.libsvm')
dtest = xgb.DMatrix('./Affairs.txt.test.libsvm')

print("Train dataset contains {0} rows and {1}
columns".format(dtrain.num_row(), dtrain.num_col()))

print("Test dataset contains {0} rows and {1}
columns".format(dtest.num_row(), dtest.num_col()))

print("Train possible labels: ")
print(np.unique(dtrain.get_label()))
print("\nTest possible labels: ")
print(np.unique(dtest.get_label()))


# specify parameters via map
params = {'max_depth':2, 'eta':1, 'silent':1,
'objective':'binary:logistic' }
num_rounds = 5
bst = xgb.train(params, dtrain, num_rounds)


watchlist = [(dtest,'test'), (dtrain,'train')] # native interface only

bst = xgb.train(params, dtrain, num_rounds, watchlist)


# make prediction
preds = bst.predict(dtest)
print preds

labels = dtest.get_label()
preds = preds > 0.276 # threshold
correct = 0

for i in range(len(preds)):
 if (labels[i] == preds[i]):
 correct += 1


print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-correct/len(preds)))


"""

"""
