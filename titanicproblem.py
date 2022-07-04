import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


dane_tytanica = pd.read_csv('train.csv')
print(dane_tytanica.head())

#usuniecie bezuzytecznych danych oraz uzupelnienie danych brakujacych

dane_tytanica = dane_tytanica.drop(columns='Cabin', axis=1)

dane_tytanica['Age'].fillna(dane_tytanica['Age'].mean(), inplace=True)

dane_tytanica['Embarked'].fillna(dane_tytanica['Embarked'].mode()[0], inplace=True)

# print(dane_tytanica.isnull().sum())
# print(dane_tytanica.describe())
# print(dane_tytanica['Survived'].value_counts())

dane_tytanica.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)

print(dane_tytanica.head())

#dalsze filtrowanie danych
ZestawX = dane_tytanica.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
ZestawY = dane_tytanica['Survived']

X_trening, X_test, Y_trening, Y_test = train_test_split(ZestawX, ZestawY, test_size=0.2, random_state=2)

model_treningowy = LogisticRegression()
model_treningowy.fit(X_trening, Y_trening)

X_predykcja = model_treningowy.predict(X_trening)
print(X_predykcja)

trafnosc_predykcji = accuracy_score(Y_trening, X_predykcja)
print("Trafność predykcji w oparciu o dane treningowe:", trafnosc_predykcji)

trafnosc_danych_testowych = accuracy_score(Y_test, X_predykcja)
print("Trafność predykcji w oparciu o dane testowe:", trafnosc_danych_testowych)