
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import data_loader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = data_loader.train
validation = data_loader.validation
test = data_loader.test


df = pd.DataFrame(train[0]).iloc[:, :1000]
df['digit'] = train[1]
df_test = pd.DataFrame(test[0]).iloc[:, :100]
df_test['digit'] = test[1]

X = df.drop('digit', axis =1)
y = df['digit']
X_test = df_test.drop('digit', axis = 1)
y_test = df_test['digit']

svc = SVC()
svc.fit(X,y)

prediction = svc.predict(X_test)
score = accuracy_score(prediction, y_test)
print(score)
