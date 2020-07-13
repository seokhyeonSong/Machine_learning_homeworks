import pandas as pd
import numpy as np
data = np.array([[0.9,0.07,0.03],[0.15,0.8,0.05],[0.35,0.15,0.5]])
order = np.array(['bull','bear','stagnant','bull','stagnant','bear','bear','bull'])
cal = 1
for i in range(len(order)-1) :
    if order[i+1] == 'bear':
        set2 = 2
    elif order[i+1] == 'bull':
        set2 = 1
    else:
        set2 = 3
    if order[i] == 'bear':
        set1 = 2
    elif order[i] == 'bull':
        set1 = 1
    else:
        set1 = 3
    cal = cal * data[set1-1,set2-1]
print(cal)