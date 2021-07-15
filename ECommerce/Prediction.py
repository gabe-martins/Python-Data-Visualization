import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import json 
import os
import matplotlib.pyplot as plt
import nltk

import tensorflow as tf

from urllib.request import urlopen
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

for dirname, _, filenames in os.walk('e-commerce/'):
  for filename in filenames:
    print(os.path.join(dirname, filename))

raw_path = "/e-commerce/"

customers = pd.read_csv(raw_path + 'olist_customers_dataset.csv')
geolocalization = pd.read_csv(raw_path + 'olist_geolocation_dataset.csv')
order_items = pd.read_csv(raw_path + 'olist_order_items_dataset.csv')
order_payments = pd.read_csv(raw_path +'olist_order_payments_dataset.csv')
order_reviews = pd.read_csv(raw_path +'olist_order_reviews_dataset.csv')
orders=pd.read_csv(raw_path + 'olist_orders_dataset.csv')
products_dataset=pd.read_csv(raw_path + 'olist_products_dataset.csv')
sellers_dataset=pd.read_csv(raw_path + 'olist_sellers_dataset.csv')
translation=pd.read_csv(raw_path + 'product_category_name_translation.csv')

# Separando base para teste e treino
def split_base():
  x_treino, x_teste, y_treino, y_teste = train_test_split(x[['freight_value']], order_items.price, test_size = 0.7, random_state = 5)
  print(x_treino.shape)

# Treinando modelo
regr = LinearRegression()
regr.fit(x_treino, y_treino)
pred_teste = regr.predict(x_teste)

plt.scatter(regr.predict(x_treino), regr.predict(x_treino) - y_treino, c = 'b', s=40, alpha=0.5)
plt.scatter(regr.predict(x_teste), regr.predict(x_teste) - y_teste, c = 'g', s=40, alpha=0.5)
plt.show()


###
#TensorFlow
###
train_x, test_x = np.asarray(train_test_split(order_items[['freight_value']], test_size=0.6, random_state=5 ))
train_y, test_y = np.asarray(train_test_split(order_items[['price']], test_size=0.6, random_state=5))
n_samples = train_x.shape[0]

regr_2 = LinearRegression()
regr_2.fit(train_x, train_y)
pred_test = regr_2.predict(test_x)

#
plt.scatter(regr_2.predict(train_x), regr_2.predict(train_x) - train_y, c = 'b', s=40, alpha=0.5)
plt.scatter(regr_2.predict(test_y), regr_2.predict(test_y) - test_y, c = 'g', s=40, alpha=0.5)
plt.show()