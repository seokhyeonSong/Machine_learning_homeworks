import pandas as pd
from kmodes.kmodes import KModes

def kmode(data,k):
    import numpy as np
    import pandas as pd
    import math
    import random
    import sys

    mode = {} #first random centroid
    cluster = {} #clustering
    modevector = {} #mode vector after first random centroid
    randomode = {} #first random centroid which has 1 index dataframe
    #make empty dataframe
    for i in range(k):
        cluster[i] = pd.DataFrame(columns=data.columns.values)
        modevector[i] = pd.DataFrame(columns=data.columns.values)
        randomode[i] = pd.DataFrame(columns=data.columns.values)
    count = 0
    small = 0
    smallindex = 0
    stop = True
    mode = data.sample(n=k) # first random centroid
    #divide dataframe into each dataframe
    for l in range(k):
        randomode[l] = mode.iloc[l]
    #while true
    for u in range(100000):
        #reset cluster
        cluster={}
        for i in range(k):
            cluster[i] = pd.DataFrame(columns=data.columns.values)
        # if it is not first run
        if u != 0:
            for l in range(k):
                randomode[l] = modevector[l].iloc[0]
        #divide data to each clusters
        for p in range(len(data)):
            small = 0
            #calculate distance for each centroid
            for i in range(k):
                count = 0
                #compare each columns
                for j in range(len(data.columns)):
                    if randomode[i].iloc[j] == data.iloc[p,j]:
                        count = count + 1
                #get the minimum distance for each data
                if small < count :
                    small = count
                    smallindex = i
            cluster[smallindex] = cluster[smallindex].append(data.iloc[p])

        #get mode vector
        for b in range(k):
            modevector[b] = cluster[b].mode(axis=0)
        stop = True
        #for every data
        for b in range(k):
            #except the unknown data ?
            for c in range(len(modevector[b].columns)):
                if randomode[b].iloc[c] == '?':
                    continue
                elif modevector[b].iloc[0,c] == '?':
                    continue
                #if any data is different from mode vector and centroid re-run
                elif (randomode[b].iloc[c] != modevector[b].iloc[0,c]) :
                    stop = False
        #if every data is same with mode vector and centroid
        if stop==True:
            return cluster
def purity(cluster,target,k): # for only mushrooms.csv
    enum = 0 #edible number
    pnum = 0 #posionous number
    purity = list()
    result = list()
    #fill empty list
    for i in range(k):
        result.append("hi")
        purity.append(0)
    #calculate the cluster's predict
    for i in range(k):
        for j in range(len(cluster[i])):
            if target.iloc[cluster[i].iloc[j].name] == 'e':
                enum = enum + 1
            else:
                pnum = pnum + 1
        if enum > pnum:
            result[i] = 'e'
        else:
            result[i] = 'p'
    #calculate purity
    for i in range(k):
        for j in range(len(cluster[i])):
            if target.iloc[cluster[i].iloc[j].name] == result[i]:
                purity[i] = purity[i] + 1

    return sum(purity)/len(target)

data = pd.read_csv("mushrooms.csv")
target = data['class']
data = data.drop('class',axis=1,inplace=False)


k=2
cluster = kmode(data,k)
purity = purity(cluster,target,k)

km = KModes(n_clusters=2)
km_cluster = km.fit_predict(data)
new_km_cluster = list()
for i in range(len(km_cluster)):
    if km_cluster[i] == 0:
        new_km_cluster.append('e')
    else :
        new_km_cluster.append('p')
kmpurity = 0
for i in range(len(new_km_cluster)):
    if new_km_cluster[i] == target[i]:
        kmpurity = kmpurity + 1
kmpurity = kmpurity / len(new_km_cluster)
#from 109~121 is to use kmodes library as user customized for me
#my library purity
print("my library's purity is ",purity)
#library purity
print("Library's purity is ", kmpurity)