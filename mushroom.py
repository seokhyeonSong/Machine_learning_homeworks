import math

import numpy as np
import pandas as pd
import sys
import pprint
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows',9000)
pd.set_option('display.max_columns',9000)

data = pd.read_csv('mushrooms.csv')
label = LabelEncoder()
encoded = pd.DataFrame(index = range(len(data)), columns = data.columns)
for i in range(len(data.columns)):
    encoded.iloc[:, i] = label.fit_transform(data.iloc[:, i])

target = encoded['class']
encoded = encoded.drop('class', axis = 1, inplace = False)



Eps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
min_sample = [3, 5, 10, 15, 20, 30, 50, 100]

for k in range(len(Eps)):
    for s in range(len(min_sample)):
        dbscan = DBSCAN(eps=Eps[k], min_samples=min_sample[s])
        clusters = dbscan.fit_predict(encoded)
        print(clusters)

        '''
        column = []
        for i in range(max(clusters) + 1):
            column.append(str(i))

        clusters_mat = pd.DataFrame()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(encoded)
        # normalized = normalize(scaled)
        # normalized = pd.DataFrame(normalized)
        oenum = 0;
        opnum = 0;
        for i in range(len(data)):
            if data.iloc[i, 0] == 'e':
                oenum += 1
            else:
                opnum += 1
        for i in range(max(clusters)+1):
            cluster = []
            for j in range(int(len(clusters))):
                if clusters[j] == i:
                    cluster.append(j)
            clusters_mat = pd.concat([clusters_mat, pd.DataFrame(pd.Series(cluster, name = column[i], dtype = int))], axis = 1, sort = True)
        enum = 0
        pnum = 0
        purity = 0
        if not clusters_mat.empty:
            for i in range(int(max(clusters_mat))+1):
                enum = 0
                pnum = 0
                for j in range(len(clusters_mat)):
                    if math.isnan(clusters_mat.iloc[j,i]):
                        break
                    check = data.iloc[int(clusters_mat.iloc[j,i]),0]
                    if check=='e':
                        enum += 1
                    else :
                        pnum += 1
                oenum -= enum
                opnum -= pnum
                if enum > pnum :
                    purity += enum
                else:
                    purity += pnum
        if oenum > opnum:
            purity += oenum
        else:
            purity += opnum
        purity = purity / len(data)
        print("in case of eps is ",Eps[k],"/ min_sample is ",min_sample[s], "/ purity is %.4f"%purity)
sys.stdout = open('output.txt','w')
'''