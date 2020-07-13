import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import seaborn as sn
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

# ignore warning
warnings.filterwarnings("ignore")

# load data from csv
heart = pd.read_csv('heart.csv')

# divide train and test
X_train, X_test, y_train, y_test = train_test_split(heart.drop(['target'], axis=1), heart['target'])

# decision tree
print('Decision tree')
score_dt = []
dTree_gini = DecisionTreeClassifier(criterion='gini', random_state=0)
dTree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)

# learing classifier
dTree_gini.fit(X_train, y_train)
dTree_entropy.fit(X_train, y_train)

X = heart.drop(['target'], axis=1)
y = heart['target']

# train model with cv of 10
cv_scores_gini = cross_val_score(dTree_gini, X, y, cv=10)
cv_scores_entropy = cross_val_score(dTree_entropy, X, y, cv=10)
cv_predict_gini = cross_val_predict(dTree_gini,X,y,cv=10)
cv_predict_entropy = cross_val_predict(dTree_entropy,X,y,cv=10)

score_avg_gini = np.mean(cv_scores_gini)
score_avg_entropy = np.mean(cv_scores_entropy)

score_dt.append(score_avg_gini)
score_dt.append(score_avg_entropy)
print('gini average:', score_avg_gini)
print('entropy average:', score_avg_entropy)

predict_dt = []
predict_dt.append(confusion_matrix(y,cv_predict_entropy))
predict_dt.append(confusion_matrix(y,cv_predict_gini))
predict_dt = np.mean(predict_dt,axis=0)
df_dt = pd.DataFrame(predict_dt,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('decision tree')
sn.heatmap(df_dt, annot=True,fmt='.1f')
plt.show()
#2d bars decision tree
bar_y = score_dt
bar_x = ['Gini', 'Entropy']
plt.bar(bar_x, bar_y, color="blue")
plt.ylabel('Average')
plt.show()


# logistic regression
score_lr = []
predict_lr = []
print('\nlogistic regression')
solvers = ['liblinear', 'lbfgs', 'sag']
max_iters = [50, 100, 200]

for i in range(3):
    score_lr.append([])
    predict_lr.append([])
    for j in range(3):
        logisticRegr = LogisticRegression(solver=solvers[i], max_iter=max_iters[j])

        # learning classifier
        logisticRegr.fit(X_train, y_train)
        scores = cross_val_score(logisticRegr, X, y, cv=10)
        #predict classifier
        predict = cross_val_predict(logisticRegr,X,y,cv=10)
        score = np.mean(scores)
        predict_lr[i].append(confusion_matrix(y,predict))
        score_lr[i].append(score)
        print('solver:', solvers[i], ' max_iter:', max_iters[j], ' score:', score)

#get average Confusion matrix of logistic regression
predict_lr = np.mean(predict_lr,axis=0)
predict_lr = np.mean(predict_lr,axis=0)
df_cm = pd.DataFrame(predict_lr,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('logistic regression')
sn.heatmap(df_cm, annot=True,fmt='.1f')
plt.show()
# 3d bar
score_lr = np.array(score_lr)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(solvers)
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(max_iters)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = score_lr
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz)
ax1.set_xlabel('solver')
ax1.set_ylabel('max_iter')
ax1.set_zlabel('score')
ax1.set_title('logistic regression')

plt.show()

# SVM
print('\nSVM')
predict_svm_liblinear = []
predict_svm_poly = []
predict_svm_rbf = []
predict_svm_sigmoid = []
score_svm_liblinear = []
score_svm_poly = []
score_svm_rbf = []
score_svm_sigmoid = []
Cs = [0.1, 1.0, 10.0]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
gammas = [0.1, 10, 100]

for i in range(3):
    score_svm_liblinear.append([])
    score_svm_poly.append([])
    score_svm_rbf.append([])
    score_svm_sigmoid.append([])
    for j in range(4):
        for k in range(3):
            SVM = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel=kernels[j], gamma=gammas[k], C=Cs[i]))
            ])

            # Learning classifier
            SVM.fit(X_train, y_train)

            # Train model with cv of 10
            scores = cross_val_score(SVM, X, y, cv=10)
            score = np.mean(scores)
            # predict model
            predict = cross_val_predict(SVM,X,y,cv=10)
            if j == 0:
                score_svm_liblinear[i].append(score)
                predict_svm_liblinear.append(confusion_matrix(y,predict))
            elif j == 1:
                score_svm_poly[i].append(score)
                predict_svm_poly.append(confusion_matrix(y,predict))
            elif j == 2:
                score_svm_rbf[i].append(score)
                predict_svm_rbf.append(confusion_matrix(y,predict))
            elif j == 3:
                score_svm_sigmoid[i].append(score)
                predict_svm_sigmoid.append(confusion_matrix(y,predict))
            print('C:', Cs[i], ' kernel:', kernels[j], ' gamma:', gammas[k], ' score:', score)

#get average Confusion matrix of SVM with liblinear kernel
predict_svm_liblinear = np.mean(predict_svm_liblinear,axis=0)
df_cm = pd.DataFrame(predict_svm_liblinear,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('SVM with liblinear')
sn.heatmap(df_cm, annot=True,fmt='.1f')
plt.show()
# 3d bar with kernel linear
score_svm_liblinear = np.array(score_svm_liblinear)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(Cs)
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(gammas)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = score_svm_liblinear
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz)
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('score')
ax1.set_title('SVM with kernel kernel')

plt.show()

#get average Confusion matrix of SVM with poly kernel
predict_svm_poly = np.mean(predict_svm_poly,axis=0)
df_cm = pd.DataFrame(predict_svm_poly,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('SVM with poly')
sn.heatmap(df_cm, annot=True,fmt='.1f')
plt.show()

# 3d bar with kernel poly
score_svm_poly = np.array(score_svm_poly)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(Cs)
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(gammas)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = score_svm_poly
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz)
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('score')
ax1.set_title('SVM with kernel poly')

plt.show()
#get average Confusion matrix of SVM with rbf kernel
predict_svm_rbf = np.mean(predict_svm_rbf,axis=0)
df_cm = pd.DataFrame(predict_svm_rbf,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('SVM with rbf')
sn.heatmap(df_cm, annot=True,fmt='.1f')
plt.show()

# 3d bar with kernel rbf
score_svm_rbf = np.array(score_svm_rbf)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(Cs)
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(gammas)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = score_svm_rbf
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz)
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('score')
ax1.set_title('SVM with kernel rbf')

plt.show()

#get average Confusion matrix of SVM with sigmoid kernel
predict_svm_sigmoid = np.mean(predict_svm_sigmoid,axis=0)
df_cm = pd.DataFrame(predict_svm_sigmoid,index=['false','true'],columns=['false','true'])
plt.figure(figsize=(10,7))
plt.title('SVM with sigmoid')
sn.heatmap(df_cm, annot=True,fmt='.1f')
plt.show()

# 3d bar with kernel sigmoid
score_svm_sigmoid = np.array(score_svm_sigmoid)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax1 = fig.add_subplot(111, projection='3d')

xlabels = np.array(Cs)
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(gammas)
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = score_svm_sigmoid
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.w_xaxis.set_ticks(xpos + dx / 2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy / 2.)
ax1.w_yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz)
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('score')
ax1.set_title('SVM with kernel sigmoid')

plt.show()