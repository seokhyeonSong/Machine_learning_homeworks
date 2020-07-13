import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('go_university.csv')
data.loc[20] = ['null','university','university','girl']
encoder = LabelEncoder()
encoded = pd.DataFrame(index = range(len(data)), columns = data.columns)
for i in range(len(data.columns)):
    encoded.iloc[:, i] = encoder.fit_transform(data.iloc[:, i])

target = encoded['university']
target = target.drop(20,0)
data = encoded.drop('university', axis=1, inplace=False)
pre = encoded.iloc[20,1:]
data = data.drop(20,0)

new_pre = pd.DataFrame(data=pre)
no = data[encoded['university'] == 1]
no_col1 = no['dad'].value_counts()
no_col2 = no['mom'].value_counts()
no_col3 = no['child'].value_counts()
no_val1 = no_col1.loc[new_pre.loc['dad']]
no_val2 = no_col2.loc[new_pre.loc['mom']]
no_val3 = no_col3.loc[new_pre.loc['child']]


yes = data[encoded['university'] == 0]
yes_col1 = yes['mom'].value_counts()
yes_col2 = yes['dad'].value_counts()
yes_col3 = yes['child'].value_counts()
yes_val1 = yes_col1.loc[new_pre.loc['mom']]
yes_val2 = yes_col2.loc[new_pre.loc['dad']]
yes_val3 = yes_col3.loc[new_pre.loc['child']]

yescal = yes_val1.iloc[0]/len(yes) * yes_val2.iloc[0]/len(yes) * yes_val3.iloc[0]/len(yes) *len(yes)/len(encoded)
nocal = no_val1.iloc[0]/len(no) * no_val2.iloc[0]/len(no) * no_val3.iloc[0]/len(no) *len(no)/len(encoded)

if yescal>nocal:
    print("go university")
else :
    print("no go university")
