import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

data = pd.DataFrame(data=([1,3,4],
                          [2,3,4,5],
                          [1,2,3,5],
                          [2,5],
                          [1,2,3,5],
                          [2,4,5],
                          [3,4,5]))
num_records = len(data)
records = []
for i in range(0, num_records):
    records.append([str(data.values[i,j]) for j in range(0, data.iloc[i])])
min_support = 1.4
min_confidence = 0.2
min_lift = 3
min_length = 2
association_rules = apriori(records, min_support=1.4,min_confidence=0.20, min_lift=3, min_length=2)
association_results = list(association_rules)
print(len(association_results))
print(association_results[0])
