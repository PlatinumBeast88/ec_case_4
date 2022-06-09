import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

train_data = pd.read_excel("job_satisfaction_train.xlsx")
train_data_copy = train_data

def encoder(text_to_enc):
    enc = LabelEncoder()
    text_to_enc['gender'] = enc.fit_transform(text_to_enc['gender'])
    return text_to_enc


encoder(train_data)
train_data = train_data.drop(columns=['marital', 'gender', 'reside', 'multline',
                              'callid', 'owntv', 'ownfax', 'response',
                                      'voice', 'pager', 'owncd'])

tr_data = train_data.loc[0:5399]
train_data.dropna(subset=['jobsat'])
test_data = train_data.loc[5400:6401]

tr_data_y = tr_data['jobsat']
tr_data_x = tr_data.drop(columns=['jobsat'])
test_data_x = test_data.drop(columns=['jobsat'])


# model = xgboost.XGBRegressor()
# model.fit(tr_data_x, tr_data_y)
#
# predictions = model.predict(test_data_x)
#
copy = train_data
copy = copy.drop(columns=['age', 'address', 'income', 'car', 'carcat', 'employ', 'retire', 'empcat'])

corr = copy.corr(method = 'spearman')
sb.heatmap(corr, annot=True)
plt.show()
X_subtrain = train_data.loc[0:4000]
X_subtrain = X_subtrain.drop(columns='jobsat')
X_val = train_data.loc[4001:5399]
y_subtrain = tr_data_y.loc[0:4000]
X_val = X_val.drop(columns='jobsat')


# print(X_val)

clf = xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500,
                           silent=True, objective='reg:linear', nthread=-1, gamma=0,
                           min_child_weight=1, max_delta_step=0, subsample=1,
                           colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                           reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                           seed=0, missing=1)
clf.fit(tr_data_x, tr_data_y)
predictions = clf.predict(test_data_x)
print(tr_data_x, tr_data_y, test_data_x)

zhizha = pd.DataFrame({"jobsat": predictions})

zhizha.to_excel("predictions.xlsx", index=False)
