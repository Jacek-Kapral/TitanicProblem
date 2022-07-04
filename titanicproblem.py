import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

daneTytanica = pd.read_csv('train.csv')
print(daneTytanica)

sea.heatmap(daneTytanica.corr(), cmap="Oranges")
plt.show()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class UzupelniaczWieku(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        uzupelniacz = SimpleImputer(strategy="mean")
        X['Age'] = uzupelniacz.fit_transform(X[['Age']])
        return X
from sklearn.preprocessing import OneHotEncoder

class CechyPasazera(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        enkodowanie = OneHotEncoder()
        macierz = enkodowanie.fit_transform(X[['Embarked']]).toarray()
        nazwy_kolumn = ["C", "S", "Q", "N"]

        for i in range(len(macierz.T)):
            X[nazwy_kolumn[i]] = macierz.T[i]

        macierz = enkodowanie.fit_transform(X, [['Sex']]).toarray()

        nazwy_kolumn = ["Female", "Male"]

        for i in range(len(macierz.T)):
            X[nazwy_kolumn[i]] = macierz.T[i]
        return X

class PomijaczCech(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "N", "Sex"], axis=1, errors="ignore")

from sklearn.pipeline import Pipeline

przefiltrowanedane = Pipeline([("uzupelniaczwieku", UzupelniaczWieku()),
                     ("cechypasazera", CechyPasazera()),
                     ("pomijaczcech", PomijaczCech())])

zestaw_treningowy = przefiltrowanedane.fit_transform()