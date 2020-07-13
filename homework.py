import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv('weatherAUS_ch.csv')
df.fillna(0,inplace=True)
df.head()
x = df.drop(columns=['RainTomorrow'])
x.head()
y = df['RainTomorrow'].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1,stratify=y)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
print("holdout method")
print(knn.predict(x_test[0:10]))
print(knn.score(x_test,y_test))
print("k-fold method")
knn_cv = KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(knn_cv, x, y, cv=10)
print(cv_scores)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))
