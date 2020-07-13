import pandas as pd
import numpy as np
data = np.array([[0.6,0.15,0.05,0.2],[0.05,0.8,0.1,0.05],[0.05,0.15,0.5,0.3],[0.2,0.15,0.15,0.5]])
order = np.array(['rest','eat','study','study','walk','rest'])
cal = 1
for i in range(len(order)-1) :
    if order[i+1] == 'study':
        set2 = 1
    elif order[i+1] == 'rest':
        set2 = 2
    elif order[i+1] == 'walk':
        set2 = 3
    else :
        set2=4
    if order[i] == 'study':
        set1 = 1
    elif order[i] == 'rest':
        set1 = 2
    elif order[i] == 'walk':
        set1 = 3
    else:
        set1 = 4
    cal = cal * data[set1-1,set2-1]
print(cal)