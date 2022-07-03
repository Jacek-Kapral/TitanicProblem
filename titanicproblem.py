import pandas as pd
train = pd.read_csv("./titanic/train.csv")
train = train.drop(['Cabin'], 1, inplace=False)
train = train.dropna()
y = train['Survived']
X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
X = pd.get_dummies(train)

test = pd.read_csv("test.csv")
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
test.fillna(2, inplace=True)
test = pd.get_dummies(test)