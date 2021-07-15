import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import json 
import os
import matplotlib.pyplot as plt
import nltk

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

order_payments.payment_type.unique()

def counter_order_pay():
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

def counter_category():
  top_category = products_dataset['product_category_name'].value_counts().sort_values(ascending=False)[:10]

  fig = plt.figure(figsize=(8,6))
  sns.set_style('darkgrid')
  sns.barplot(y=top_category.index, x=top_category.values)
  plt.show()

#Price to freight 
order_items.plot(
    kind = 'scatter',
    x = 'price',
    y = 'freight_value'
)

plt.show()