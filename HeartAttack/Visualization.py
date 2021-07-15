import seaborn as sns
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

heart = pd.read_csv('heart.csv')

def counterTable():
  print("Rows:", heart.shape[0], "| Columns:",heart.shape[1])

heart[heart.duplicated()]
heart.drop_duplicates(inplace=True)
heart.describe()

def counterSex():
  x = (heart.sex.value_counts())  
  print(x[0], x[1])
  p = sns.countplot(data = heart, x = "sex")
  plt.show()

def counterFbs():
  x = (heart.fbs.value_counts())
  print(x)
  p = sns.countplot(data = heart, x = "fbs")
  plt.show()

def counterAge():
  plt.figure(figsize=(10,10))
  sns.displot(heart.age, color="#27a2a8", label="age", kde=True)
  plt.legend()

def cBloodPressure():
  plt.figure(figsize=(20, 20))
  sns.displot(heart.trtbps, color="#40de4b", label="Blood Pressure", kde=True)
  plt.legend()

def cChol():
  plt.figure(figsize=(20,20))
  sns.displot(heart[heart['output'] == 0]["chol"], color="#40de4b", kde=True,)
  sns.displot(heart[heart['output'] == 1]["chol"], color="#de8f40", kde=True)
  plt.title('Cholestrol vs Age')
  plt.show()
