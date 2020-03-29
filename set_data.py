import pandas as pd
import numpy as np
import math

save_path = "./data_set/"
origin_path = "./titanic/"

df = pd.read_csv(origin_path+"train.csv")

df = df[['PassengerId', 'Age', 'SibSp', 'Parch', 'Pclass', 'Survived', 'Sex']]

df.fillna(60, inplace = True)
df['Acquaintance'] = (df['SibSp'] > 0) | (df['Parch'] > 0)
df.drop(['SibSp', 'Parch'], axis='columns', inplace=True)
df['Age'] = df["Age"].apply(lambda x: float(int(math.ceil(x))))
df['Acquaintance'] = df["Acquaintance"].apply(lambda x: float(x))
df['Pclass'] = df["Pclass"].apply(lambda x: float(x))
df['Survived'] = df["Survived"].apply(lambda x: float(x))
df['Sex'] = df["Sex"].apply(lambda x: float(1) if x == 'male' else float(0))

cabin = pd.read_csv(origin_path+"train.csv")
cabin = cabin['Cabin']
cabin.fillna('H', inplace = True)
cabin = cabin.apply(lambda x: float(ord(x[:1])-65))
df['Cabin'] = cabin

df.to_csv(save_path+"ch_train2.csv", index=False)

#########################################################################

df = pd.read_csv(origin_path+"test.csv")

df = df[['PassengerId', 'Age', 'SibSp', 'Parch', 'Pclass', 'Sex', 'Cabin']]

df.fillna(60, inplace = True)
df['Acquaintance'] = (df['SibSp'] > 0) | (df['Parch'] > 0)
df.drop(['SibSp', 'Parch'], axis='columns', inplace=True)
df['Age'] = df["Age"].apply(lambda x: float(int(math.ceil(x))))
df['Acquaintance'] = df["Acquaintance"].apply(lambda x: float(x))
df['Pclass'] = df["Pclass"].apply(lambda x: float(x))
df['Sex'] = df["Sex"].apply(lambda x: float(1) if x == 'male' else float(0))

cabin = pd.read_csv(origin_path+"test.csv")
cabin = cabin['Cabin']
cabin.fillna('H', inplace = True)
cabin = cabin.apply(lambda x: float(ord(x[:1])-65))
df['Cabin'] = cabin

df.to_csv(save_path+"ch_test2.csv", index=False)