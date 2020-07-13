import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train.isna().head()
test.isna().head()
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)

train = train.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

scaler = MinMaxScaler()
X_Scaled = scaler.fit_transform(X)
kmeans = KMeans(n_init=30, n_clusters=2, max_iter=30000, algorithm='auto')
kmeans.fit(X_Scaled)
KMeans(algorithm='auto', copy_x=True, init='k-means++',max_iter=30000, n_clusters=2, n_init=30, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))