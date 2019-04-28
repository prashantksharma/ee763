import pandas as pd

train = pd.read_csv('train.csv',names=['a','b','c','d','e','f'])
print(pd.unique(train['f']))
print(len(train['f']))
train = train[train.f !='Car']
train = train[train.f !='Cart']
train = train[train.f !='Skater']
print(len(train))
print(pd.unique(train['f']))
train.to_csv('train_labels.csv',index=False)
