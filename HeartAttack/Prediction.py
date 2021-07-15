import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import xgboost as xgb

heart = pd.read_csv('heart.csv')

x = heart.iloc[:, 1:-1].values
y = heart.iloc[:, -1].values
x,y

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

print("Training data:", x_train.shape, y_train.shape)
print("Testing data:", x_test.shape, y_test.shape)

def Feature_Scalling():
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  x_train, x_test

def Logistic_Regression():
  model = LogisticRegression()
  model.fit(x_train, y_train)
  predicted = model.predict(x_test)
  conf = confusion_matrix(y_test, predicted)

  print("Matrix: \n", conf)
  print("Accuracy:", accuracy_score(y_test, predicted)*100, "%")

def Gaussian_Naive_Bayes():
  model = GaussianNB()
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  print("Accuracy:", accuracy_score(y_test, predicted)*100, "%")

def Bernoulli_Naive_Bayes():
  model = BernoulliNB()
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  print("Accuracy:", accuracy_score(y_test, predicted)*100, "%")

def Support_Vector_Machine():
  model = SVC()
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  print("Accuracy:", accuracy_score(y_test, predicted)*100, "%")

def Random_Forest():
  model = RandomForestRegressor(n_estimators= 100, random_state= 0)
  model.fit(x_train, y_train)
  predicted = model.predict(x_test)

  print("Accuracy:", accuracy_score(y_test, predicted.round())*100, "%")

def K_Nearest_Neighbours():
  model = KNeighborsClassifier(n_neighbors= 1)
  model.fit(x_train, y_train)
  predicted = model.predict(x_test)

  print(confusion_matrix(y_test, predicted))
  print("Accuracy:", accuracy_score(y_test, predicted.round())*100, "%")

def Optimizing_the_KNN():
  model = KNeighborsClassifier(n_neighbors=7)
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  from sklearn.metrics import classification_report, confusion_matrix

  print(confusion_matrix(y_test, predicted))
  print("Accuracy: ", accuracy_score(y_test, predicted.round())*100, "%")

def X_Grandient_Boosting():
  model = xgb.XGBClassifier(use_label_encoder=False)
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  cm = confusion_matrix(y_test, predicted)
  print(cm)
  print("Accuracy:", accuracy_score(y_test, predicted)*100, "%")