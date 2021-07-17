import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import json 
import os
import matplotlib.pyplot as plt
import nltk
import statistics  as sts
# %matplotlib inline 

from urllib.request import urlopen
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

for dirname, _, filenames in os.walk('/content/drive/MyDrive/datasets/e-commerce/'):
  for filename in filenames:
    print(os.path.join(dirname, filename))

raw_path = "/content/drive/MyDrive/datasets/e-commerce/"

customers = pd.read_csv(raw_path + 'olist_customers_dataset.csv')
geolocalization = pd.read_csv(raw_path + 'olist_geolocation_dataset.csv')
order_items = pd.read_csv(raw_path + 'olist_order_items_dataset.csv')
order_payments = pd.read_csv(raw_path +'olist_order_payments_dataset.csv')
order_reviews = pd.read_csv(raw_path +'olist_order_reviews_dataset.csv')
orders=pd.read_csv(raw_path + 'olist_orders_dataset.csv')
products_dataset=pd.read_csv(raw_path + 'olist_products_dataset.csv')
sellers_dataset=pd.read_csv(raw_path + 'olist_sellers_dataset.csv')
translation=pd.read_csv(raw_path + 'product_category_name_translation.csv')

#Removendo not_defined
order_payments.loc[order_payments['payment_type'] == 'not_defined', 'payment_type'] = 'credit_card'
order_payments.payment_type.unique()

x1 = order_payments.payment_type.value_counts()
x = (order_payments.payment_type)
y = round(order_payments.groupby('payment_type')['payment_value'].sum(), 2)
y2 = order_payments.groupby('payment_type').size().sort_values()

print(y)
fig = plt.figure(figsize=(6,6))
sns.set_style('darkgrid')
sns.barplot(y.index, y.values)
plt.show()

order_payments['payment_type'].value_counts().plot(
    kind='pie',
    autopct="%1.1f%%",
    fontsize=(12),
    figsize=(6,6)
)

plt.show()

top_category = products_dataset['product_category_name'].value_counts().sort_values(ascending=False)[:10]

fig = plt.figure(figsize=(8,6))
sns.set_style('darkgrid')
sns.barplot(y=top_category.index, x=top_category.values)
plt.show()

round(order_items['freight_value'].corr(order_items['price']), 2)

order_items.plot(
    kind = 'scatter',
    x = 'price',
    y = 'freight_value'
)

plt.show()

x = order_items.drop('price', axis = 1 )
y = order_items.price