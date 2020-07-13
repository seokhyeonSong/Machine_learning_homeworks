import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

df = pd.read_csv('mouse.csv')

cluster = [2,3,4,5,6]
iter = [50,100,200,300]
for i in range(len(cluster)):
    for j in range(len(iter)):
        kmeans = KMeans(n_clusters=cluster[i],max_iter=iter[j])
        y = kmeans.fit_predict(df.values)
        kmeans.fit(df.values)
        plt.xlabel('x')
        plt.ylabel('y')
        centers = pd.DataFrame(kmeans.cluster_centers_, columns = ['X', 'Y'])
        center_x = centers['X']
        center_y = centers['Y']
        plt.title("cluster : "+ str(cluster[i]) + " max_iter : " + str(iter[j]))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c = y, alpha = 0.5)
        plt.scatter(center_x, center_y, s = 50, marker = 'D', c = 'r')
        plt.show()

eps = [0.001, 0.002, 0.005,0.01,0.02,0.05,0.1,0.2,0.5]
sample = [3,5,10,15,20,30,50,100]
for i in range(len(eps)):
    for j in range(len(sample)):
        db_default = DBSCAN(eps = eps[i], min_samples = sample[j])
        y_pred = db_default.fit_predict(df.values)
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c = y_pred, alpha = 0.5)
        plt.title('eps : '+str(eps[i]) + ' min_samples : ' + str(sample[j]))
        plt.show()