from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
warnings.filterwarnings("ignore")

mnist = fetch_mldata("MNIST original")
#split datas -> using 10% of data, train, test, ensemble_test
use_img, none_img, use_lbl, none_lbl = train_test_split(mnist.data, mnist.target, test_size=6.3/7, random_state=0)
train_img, test_img, train_lbl, test_lbl = train_test_split(use_img, use_lbl,test_size=5/7, random_state=0)
test_img, etest_img, test_lbl, etest_lbl = train_test_split(test_img, test_lbl, random_state=1)

#logistic regression for get highest score's params
score_lr = []
solvers = ['liblinear', 'lbfgs', 'sag']
max_iters = [50, 100, 200]
high_solver = ''
high_maxiter = 0
high_score_lr = 0

for i in range(3):
    score_lr.append([])
    for j in range(3):
        logisticRegr = LogisticRegression(solver=solvers[i], max_iter=max_iters[j])

        # learning classifier
        logisticRegr.fit(train_img, train_lbl)
        scores = logisticRegr.score(test_img,test_lbl)
        score = np.mean(scores)
        score_lr[i].append(score)
        if score>high_score_lr:
            high_solver = solvers[i]
            high_maxiter = max_iters[i]
            high_score_lr = score

#SVM for get highest score's params of kernel linear & rbf
predict_svm_liblinear = []
predict_svm_rbf = []
score_svm_liblinear = []
score_svm_rbf = []
SVC_linear_score = 0
SVC_linear_gamma = 0
SVC_linear_C = 0
SVC_rbf_score = 0
SVC_rbf_gamma = 0
SVC_rbf_C = 0
Cs = [0.1, 1.0, 10.0]
kernels = ['linear','rbf']
gammas = [0.1, 10, 100]

for i in range(3):
    score_svm_liblinear.append([])
    score_svm_rbf.append([])
    for j in range(2):
        for k in range(3):
            SVM = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel=kernels[j], gamma=gammas[k], C=Cs[i]))
            ])

            # Learning classifier
            SVM.fit(train_img, train_lbl)

            # Train model with cv of 10
            scores = SVM.score(test_img, test_lbl)
            score = np.mean(scores)
            # predict model
            if j == 0:
                score_svm_liblinear[i].append(score)
                if score>SVC_linear_score :
                    SVC_linear_C = Cs[i]
                    SVC_linear_gamma = gammas[k]
                    SVC_linear_score = score
            elif j == 1:
                score_svm_rbf[i].append(score)
                if score>SVC_rbf_score :
                    SVC_rbf_C = Cs[i]
                    SVC_rbf_gamma = gammas[k]
                    SVC_rbf_score = score

#make 3 classifiers with which shows highest score's params
lr_clf = LogisticRegression(solver=high_solver, max_iter=high_maxiter)
SVM_linear_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel='linear', gamma=SVC_linear_gamma, C=SVC_linear_C,probability=True))])
SVM_rbf_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel='rbf', gamma=SVC_rbf_gamma, C=SVC_rbf_C, probability=True))])

#make voting classifiers for ensemble test
vo_clf = VotingClassifier(estimators=[('lr',lr_clf),('SVM_linear',SVM_linear_clf),('SVM_rbf',SVM_rbf_clf)], voting='soft')
vo_clf.fit(train_img,train_lbl)
pred = vo_clf.predict(etest_img)
print(format(accuracy_score(etest_lbl,pred)))
#show confusion matrix of ensemble test
cpred = confusion_matrix(etest_lbl,pred)
df_cm = pd.DataFrame(cpred,index=[0,1,2,3,4,5,6,7,8,9],columns=[0,1,2,3,4,5,6,7,8,9])
plt.figure(figsize=(10,7))
plt.title('ensemble test')
sn.heatmap(df_cm, annot=True,fmt='d')
plt.show()

#compare with other 3 classifer
print("logistic regression : ", high_score_lr)
print("SVC with linear kernel : ", SVC_linear_score)
print("SVC with rbf kernel : ", SVC_rbf_score)
print("ensemble test : ", format(accuracy_score(etest_lbl,pred)))