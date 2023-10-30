import numpy as num
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the csv file
wine = pd.read_csv("winequality-red.csv")

print(wine.head())

print(wine.shape)

import matplotlib.pyplot as plt
import seaborn as sns

#barplot
plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='alcohol', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='fixed acidity', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='citric acid', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='residual sugar', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='chlorides', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='free sulfur dioxide', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='total sulfur dioxide', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='density', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='pH', data=wine)
plt.show()

plt.figure(figsize = (5,5))
sns.barplot(x='quality', y='sulphates', data=wine)
plt.show()

#heatmap
plt.figure(figsize = (12,5))
sns.heatmap(wine.corr(), annot=True)
plt.show()

#model training
X=wine.drop(['quality'], axis=1)
Y=wine['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
wine['quality'].unique()
wine['quality']=[1 if x>=7 else 0 for x in wine['quality']]
wine['quality'].value_counts()
sns.countplot(wine['quality'])

#handling imbalanced dataset
from imblearn.over_sampling import SMOTE

X_res, Y_res = SMOTE().fit_resample(X,Y)
Y_res.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_res,Y_res,test_size=0.2, random_state=2)

from sklearn.metrics import accuracy_score

#RandomForest (evaluation)

from sklearn.ensemble import RandomForestClassifier
rando = RandomForestClassifier()
rando.fit(X_train, Y_train)
X_test_predict4 = rando.predict(X_test)
accuracy_score(Y_test, X_test_predict4)

input_data = (7.1,0.6,0.124,0.3,0.082,5.0,5.9,0.8597,3.16,2.65,10.0)

input_data_as_numpy_array = num.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = rando.predict(input_data_reshaped)

if (prediction[0]==1):
    print('Good quality')
else:
    print('Bad quality')

from flask import Flask, render_template, request
import pickle
import os

pickle.dump(rando, open("WineQuality.pkl", "wb"))