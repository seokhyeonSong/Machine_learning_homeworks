import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('play_golf.csv')
data.loc[20] = ['null','sunny','cool','high',False]
encoder = LabelEncoder()
encoded = pd.DataFrame(index = range(len(data)), columns = data.columns)
for i in range(len(data.columns)):
    encoded.iloc[:, i] = encoder.fit_transform(data.iloc[:, i])

target = encoded['play Golf']
target = target.drop(20,0)
data = encoded.drop('play Golf', axis=1, inplace=False)
pre = encoded.iloc[20,1:]
data = data.drop(20,0)
no = data[encoded['play Golf'] == 0]
new_pre = pd.DataFrame(data=pre)
no_col1 = no['Outlook'].value_counts()
no_col2 = no['Temp'].value_counts()
no_col3 = no['Humidity'].value_counts()
no_col4 = no['Wind'].value_counts()
no_val1 = no_col1.loc[new_pre.loc['Outlook']]
no_val2 = no_col2.loc[new_pre.loc['Temp']]
no_val3 = no_col3.loc[new_pre.loc['Humidity']]
no_val4 = no_col4.loc[new_pre.loc['Wind']]

yes = data[encoded['play Golf'] == 2]
yes_col1 = yes['Outlook'].value_counts()
yes_col2 = yes['Temp'].value_counts()
yes_col3 = yes['Humidity'].value_counts()
yes_col4 = yes['Wind'].value_counts()
yes_val1 = yes_col1.loc[new_pre.loc['Outlook']]
yes_val2 = yes_col2.loc[new_pre.loc['Temp']]
yes_val3 = yes_col3.loc[new_pre.loc['Humidity']]
yes_val4 = yes_col4.loc[new_pre.loc['Wind']]

yescal = yes_val1.iloc[0]/len(yes) * yes_val2.iloc[0]/len(yes) * yes_val3.iloc[0]/len(yes) * yes_val4.iloc[0]/len(yes)*len(yes)/len(encoded)
nocal = no_val1.iloc[0]/len(no) * no_val2.iloc[0]/len(no) * no_val3.iloc[0]/len(no) * no_val4.iloc[0]/len(no)*len(no)/len(encoded)

print(yescal)
print(nocal)

if yescal>nocal:
    print("play golf")
else :
    print("no play golf")