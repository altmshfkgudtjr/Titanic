import pandas as pd
import numpy as np

train_path = './titanic/train.csv'
'''
PassengerId(int)	: 의미 X
Survived(int)		: 정답
Pclass(int)			: 좌석 클래스 -> 지표
Name(string)		: 의미 있을까? -> 보류
Sex(string)			: 성별 -> 지표
Age(float)			: 나이 -> 지표
SibSp(int)			: 함께 탑승한 형제 및 배우자의 수 -> 지표 (존재유무가 지표가 될 수 있을 것 같다.)
Parch(int)			: 함께 탑승한 부모 및 자녀수 -> 지표 (존재유무가 지표가 될 수 있을 것 같다.)
Ticket(string)		: 티켓번호 
Fare(float)			: 티켓요금
Cabin(string)		: 선실 번호
Embarked(string)	: 탑승한 곳
'''
df = pd.read_csv(train_path)
#df.info()	# 데이터 정보확인

# 생존못한사람 + 필요없는 컬럽 제외
# df = df.loc[df['Survived'] == 0,:].sort_values(by='Survived', ascending=False)[df.columns.difference(['PassengerId', 'Name'])]

print("<타이타닉 분석>")
print("전체인원 :", len(df))
print("생존/죽음 :", df['Survived'].sum(),"/",len(df)-df['Survived'].sum())
print("\n")
print("전체남성 :", len(df.loc[df['Sex'] == 'male']))
print("생존/죽음 :", len(df.loc[df['Sex'] == 'male'].loc[df['Survived'] == 1]),"/",len(df.loc[df['Sex'] == 'male'].loc[df['Survived'] == 0]))
print("생존확률 :", len(df.loc[df['Sex'] == 'male'].loc[df['Survived'] == 1]) / len(df))
print("\n")
print("전체여성 :", len(df.loc[df['Sex'] == 'female']))
print("생존/죽음 :", len(df.loc[df['Sex'] == 'female'].loc[df['Survived'] == 1]),"/",len(df.loc[df['Sex'] == 'female'].loc[df['Survived'] == 0]))
print("생존확률 :", len(df.loc[df['Sex'] == 'female'].loc[df['Survived'] == 1]) / len(df))
print("\n")
print("전체 좌석 클래스 :", len(df['Pclass']))
print("1등급 생존/죽음 :", len(df.loc[df['Pclass'] == 1].loc[df['Survived'] == 1]),"/",len(df.loc[df['Pclass'] == 1].loc[df['Survived'] == 0]))
print("2등급 생존/죽음 :", len(df.loc[df['Pclass'] == 2].loc[df['Survived'] == 1]),"/",len(df.loc[df['Pclass'] == 2].loc[df['Survived'] == 0]))
print("3등급 생존/죽음 :", len(df.loc[df['Pclass'] == 3].loc[df['Survived'] == 1]),"/",len(df.loc[df['Pclass'] == 3].loc[df['Survived'] == 0]))
print("\n")
print("선실번호 데이터 있음 :", len(df.loc[df['Cabin'].notnull()]))
print("생존/죽음 :", len(df.loc[df['Cabin'].notnull()].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].notnull()].loc[df['Survived'] == 0]))
print("A등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('A', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('A', na=False)].loc[df['Survived'] == 0]))
print("B등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('B', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('B', na=False)].loc[df['Survived'] == 0]))
print("C등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('C', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('C', na=False)].loc[df['Survived'] == 0]))
print("D등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('D', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('D', na=False)].loc[df['Survived'] == 0]))
print("E등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('E', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('E', na=False)].loc[df['Survived'] == 0]))
print("F등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('F', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('F', na=False)].loc[df['Survived'] == 0]))
print("G등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('G', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('G', na=False)].loc[df['Survived'] == 0]))
print("T등급 생존/죽음 :", len(df.loc[df['Cabin'].str.startswith('T', na=False)].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].str.startswith('T', na=False)].loc[df['Survived'] == 0]))
print("\n")
print("선실번호 데이터 없음 :", len(df.loc[df['Cabin'].isnull()]))
print("생존/죽음 :", len(df.loc[df['Cabin'].isnull()].loc[df['Survived'] == 1]),"/",len(df.loc[df['Cabin'].isnull()].loc[df['Survived'] == 0]))
print("\n")
print("나이 데이터가 있음 :", len(df[df['Age'].notnull()]))
print("0 ~ 10세 생존/죽음:", len(df.loc[(0 <= df['Age']) & (df['Age'] < 10)].loc[df['Survived'] == 1]),"/",len(df.loc[(0 <= df['Age']) & (df['Age'] < 10)].loc[df['Survived'] == 0]))
print("10 ~ 20세 생존/죽음:", len(df.loc[(10 <= df['Age']) & (df['Age'] < 20)].loc[df['Survived'] == 1]),"/",len(df.loc[(10 <= df['Age']) & (df['Age'] < 20)].loc[df['Survived'] == 0]))
print("20 ~ 30세 생존/죽음:", len(df.loc[(20 <= df['Age']) & (df['Age'] < 30)].loc[df['Survived'] == 1]),"/",len(df.loc[(20 <= df['Age']) & (df['Age'] < 30)].loc[df['Survived'] == 0]))
print("30 ~ 40세 생존/죽음:", len(df.loc[(30 <= df['Age']) & (df['Age'] < 40)].loc[df['Survived'] == 1]),"/",len(df.loc[(30 <= df['Age']) & (df['Age'] < 40)].loc[df['Survived'] == 0]))
print("40 ~ 50세 생존/죽음:", len(df.loc[(40 <= df['Age']) & (df['Age'] < 50)].loc[df['Survived'] == 1]),"/",len(df.loc[(40 <= df['Age']) & (df['Age'] < 50)].loc[df['Survived'] == 0]))
print("50 ~ 60세 생존/죽음:", len(df.loc[(50 <= df['Age']) & (df['Age'] < 60)].loc[df['Survived'] == 1]),"/",len(df.loc[(50 <= df['Age']) & (df['Age'] < 60)].loc[df['Survived'] == 0]))
print("60 ~ 70세 생존/죽음:", len(df.loc[(60 <= df['Age']) & (df['Age'] < 70)].loc[df['Survived'] == 1]),"/",len(df.loc[(60 <= df['Age']) & (df['Age'] < 70)].loc[df['Survived'] == 0]))
print("70 ~ 80세 생존/죽음:", len(df.loc[(70 <= df['Age']) & (df['Age'] < 80)].loc[df['Survived'] == 1]),"/",len(df.loc[(70 <= df['Age']) & (df['Age'] < 80)].loc[df['Survived'] == 0]))
print("80 ~ 90세 생존/죽음:", len(df.loc[(80 <= df['Age']) & (df['Age'] < 90)].loc[df['Survived'] == 1]),"/",len(df.loc[(80 <= df['Age']) & (df['Age'] < 90)].loc[df['Survived'] == 0]))
print("생존/죽음 :", len(df[df['Age'].notnull()].loc[df['Survived'] == 1]),"/",len(df[df['Age'].notnull()].loc[df['Survived'] == 0]))
print("생존확률 :", df[df['Age'].notnull()]['Survived'].sum() / df['Age'].notnull().sum())
print("\n")
print("나이 데이터가 없음 :", df['Age'].isnull().sum())
print("생존/죽음 :", len(df[df['Age'].isnull()].loc[df['Survived'] == 1]),"/",len(df[df['Age'].isnull()].loc[df['Survived'] == 0]))
print("생존확률 :", df[df['Age'].isnull()]['Survived'].sum() / df['Age'].isnull().sum())
print("\n")
print("같이 탄 지인 존재 :", len(df[(df['SibSp'] > 0) | (df['Parch'] > 0)]))
print("생존/죽음 :", len(df[(df['SibSp'] > 0) | (df['Parch'] > 0)].loc[df['Survived'] == 1]),"/",len(df[(df['SibSp'] > 0) | (df['Parch'] > 0)].loc[df['Survived'] == 0]))
print("생존확률 :", len(df[(df['SibSp'] > 0) | (df['Parch'] > 0)].loc[df['Survived'] == 1])/len(df[(df['SibSp'] > 0) | (df['Parch'] > 0)]))
print("\n")
print("같이 탄 지인 존재 안함 :", len(df[(df['SibSp'] == 0) & (df['Parch'] == 0)]))
print("생존/죽음 :", len(df[(df['SibSp'] == 0) & (df['Parch'] == 0)].loc[df['Survived'] == 1]),"/",len(df[(df['SibSp'] == 0) & (df['Parch'] == 0)].loc[df['Survived'] == 0]))
print("생존확률 :", len(df[(df['SibSp'] == 0) & (df['Parch'] == 0)].loc[df['Survived'] == 1])/len(df[(df['SibSp'] == 0) & (df['Parch'] == 0)]))
print("\n")
print("탑승한 곳 존재 :", len(df[df['Embarked'].notnull()]))
print("C 생존/죽음:", len(df[df['Embarked'] == 'C'].loc[df['Survived'] == 1]),"/",len(df[df['Embarked'] == 'C'].loc[df['Survived'] == 0]))
print("Q 생존/죽음:", len(df[df['Embarked'] == 'Q'].loc[df['Survived'] == 1]),"/",len(df[df['Embarked'] == 'Q'].loc[df['Survived'] == 0]))
print("S 생존/죽음:", len(df[df['Embarked'] == 'S'].loc[df['Survived'] == 1]),"/",len(df[df['Embarked'] == 'S'].loc[df['Survived'] == 0]))
print("\n")
print("탑승한 곳 데이터 없음 :", len(df[df['Embarked'].isnull()]))
print("생존/죽음 :", len(df[df['Embarked'].isnull()].loc[df['Survived'] == 1]),"/",len(df[df['Embarked'].isnull()].loc[df['Survived'] == 0]))
print("생존확률 :", len(df[df['Embarked'].isnull()].loc[df['Survived'] == 1])/len(df[df['Embarked'].isnull()]))
print("\n")
print("가격 분류")
print("0 ~ 50 생존/죽음", len(df[df['Fare'] < 50].loc[df['Survived'] == 1]),"/",len(df[df['Fare'] < 50].loc[df['Survived'] == 0]))
print("50 ~ 512 생존/죽음", len(df[df['Fare'] >= 50].loc[df['Survived'] == 1]),"/",len(df[df['Fare'] >= 50].loc[df['Survived'] == 0]))


