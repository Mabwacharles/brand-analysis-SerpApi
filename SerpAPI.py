#!/usr/bin/env python
# coding: utf-8

# # Environment Setup
# Import the required libraries

# In[79]:


import re
import json
import time
import pandas as pd
import plotly.express as px
from serpapi import GoogleSearch
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')


# ## Ebay data extracted by the serpAPI:
# ## Ebay tool

# In[15]:


api_key =  "serp_api_key"
engine_search = "ebay"

# Ebay products:
products = [
    {"name":"engine", "query":"sony"},
    {"name":"engine", "query":"samsung"},
    {"name":"engine", "query":"tcl"},
    {"name":"engine", "query":"lg"},
    {"name":"engine", "query":"hisense"}
    #{"name":"engine", "query":"name_of_product"}   
]

ebay_data = pd.DataFrame([])

for product in products:
    params = {
        product['name']:engine_search, 
        "_nkw": product['query'],
        "api_key": api_key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results['organic_results']
    ebay_data = ebay_data.append(pd.json_normalize(organic_results), ignore_index = True)


# ## Required Ebay data:

# In[18]:


ebay_data = ebay_data[["title", "condition", "shipping", "reviews", "top_rated", "price.extracted", "extensions"]]
ebay_data['market'] = 'Ebay'


# ## Walmart tool:

# In[30]:


api_key =  "serp_api_key"
engine_search = "walmart"

# Walmart products:
products = [
    {"name":"engine", "query":"sony"},
    {"name":"engine", "query":"samsung"},
    {"name":"engine", "query":"tcl"},
    {"name":"engine", "query":"lg"},
    {"name":"engine", "query":"hisense"}   
]


walmart_data = pd.DataFrame([])

for product in products:
    params = {
        product['name']:engine_search, 
        "query": product['query'],
        "api_key": api_key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results['organic_results']
    walmart_data = walmart_data.append(pd.json_normalize(organic_results), ignore_index = True)


# ## Required Walmart data:

# In[31]:


walmart_data = walmart_data[["title", "rating", "reviews", "two_day_shipping", "out_of_stock", "primary_offer.offer_price"]]
walmart_data['market'] = 'Walmart'


# # Cleaning Data

# **1. From the title of the product extract the title of the brand of interest**

# In[24]:


# List of brands
L = ['Samsung', 'Hisense','TCL','Sony']
pat = '|'.join(r"\b{}\b".format(x) for x in L)

# Extract the brand names, making all characters small for merging
ebay_data['brand'] = ebay_data['title'].str.findall(pat, flags=re.I).str.join(' ')
ebay_data = ebay_data.apply(lambda x: x.astype(str).str.lower())
ebay_data['brand'] = ebay_data['brand'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1', regex=True)


# **2. From the items_sold, extract the minimum integer value of the brand items sold as of the date of data extraction**

# In[25]:


# Define the function to remove the punctuation
ebay_data['extensions'] = ebay_data['extensions'].str.replace('[^\w\s]','',  regex=True)
ebay_data['extensions'] = ebay_data.extensions.str.extract('(\d+)')


# **3. Drop all the missing values: this will/may not help in our analysis.**

# In[26]:


# Drop missing values
ebay_data.dropna()

# Drop title variable
ebay_data.drop('title', inplace=True, axis=1)


# ## Walmart  data cleaning

# In[33]:


#  Extract ist of brands in walmart from Walmart product titles:
L = ['Samsung', 'Hisense','TCL','Sony']
pat = '|'.join(r"\b{}\b".format(x) for x in L)

# Extract the brand names, making all characters small for merging
walmart_data['brand'] = walmart_data['title'].str.findall(pat, flags=re.I).str.join(' ')
walmart_data = walmart_data.apply(lambda x: x.astype(str).str.lower())
walmart_data['brand'] = walmart_data['brand'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1', regex=True)


# In[35]:


# Drop missing values
walmart_data.dropna()

# Drop title variable
walmart_data.drop('title', inplace=True, axis=1)


# # Brand Analysis

# **1. Agregate the number of products for each brand and market

# In[54]:


ebay_brands = ebay_data[['brand', 'market']].reset_index()
walmart_brands = walmart_data[['brand', 'market']].reset_index()
market_brands = pd.concat([ebay_brands, walmart_brands], ignore_index=True)
count_series = market_brands.groupby(['brand', 'market']).size()
count_series = count_series.to_frame(name = 'size').reset_index()
count_series['brand'] = count_series['brand'].replace(r'^\s*$', 'lg', regex=True)
count_series.drop(count_series.tail(1).index,inplace=True) 

# Plot
fig = px.bar(count_series, x="brand", y="size", color="market")
fig.show()


# # Ebay Brand compared to the items sold 

# In[58]:


# Brand sales in Ebay
sales = ebay_data[['brand', 'extensions']].reset_index()
sales["brand"] = sales["brand"].replace("nan", "samsung", regex=True)
sales["extensions"] = pd.to_numeric(sales["extensions"])

sales = sales.groupby(['brand'])['extensions'].sum().reset_index()
sales['brand'] = sales['brand'].replace(r'^\s*$', 'lg', regex=True)

fig = px.bar(sales, x="brand", y="extensions")
fig.show()


# # Shipping cost effect on the sales of a brand in Ebay?
# **Group the data by shipping status and the number of sales:**

# In[72]:


ebay_data['shipping'] = ebay_data['shipping'].replace("nan", "with shipping cost", regex=True)
ebay_data['shipping'] = ebay_data['shipping'].str.replace('[^\w\s]','',  regex=True)

ebay_data.loc[ebay_data.shipping.str.contains('(\d+)')  == True, 'shipping'] = 'with shipping cost'

# Required data
shipped_sales = ebay_data[['brand', 'shipping', 'extensions']].reset_index()
shipped_sales['extensions'] = pd.to_numeric(shipped_sales['extensions'])


# Aggregate the data by the grouping
shipped_sales = shipped_sales.groupby(['shipping','brand'])['extensions'].sum().reset_index()
shipped_sales["extensions"] = pd.to_numeric(shipped_sales ["extensions"])
shipped_sales['brand'] = shipped_sales['brand'].replace(r'^\s*$', 'lg', regex=True)
shipped_sales = shipped_sales.drop(labels=[5,7,13], axis=0)

# Visual
fig = px.bar(shipped_sales, x="brand", y="extensions", color="shipping")
fig.show()


# # Correlation in Ebay sales

# **The question we would like to answer: is there a correlation between item prices, the reviews and the items sold?**
# ## Extract data:

# In[80]:


cor = ebay_data[['price.extracted', 'extensions', 'reviews']].reset_index()
# Drop data whose values are not known
cor_data = cor_data[~cor.reviews.str.contains("nan")]
col = cor.columns.drop('index')
cor[col] = cor [col].apply(pd.to_numeric, errors='coerce')


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(cor.corr(), dtype=np.bool))
heatmap = sns.heatmap(cor.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')


# In[74]:





# # Scatter plot
# ## 1. Correlation between prices & items sold

# In[48]:


fig = px.scatter(cor_data, x="price.extracted", y="extensions", trendline="ols")
fig.show()


# ## 2. Correlation between reviews & items sold

# In[49]:


fig = px.scatter(cor_data, x="reviews", y="extensions" , trendline="ols")
fig.show()


# ## 3. Correlation between prices & reviews

# In[50]:


fig = px.scatter(cor_data, x="reviews", y="price.extracted", trendline="ols")
fig.show()


# ## General Reviews and Prices

# In[51]:


ebay_rv = ebay_data[['brand', 'price.extracted', 'reviews', 'market']]
walmart_rv = walmart_data[['brand', 'primary_offer.offer_price', 'reviews', 'market']]
walmart_rv = walmart_rv.rename({'primary_offer.offer_price': 'price.extracted',}, axis=1)
rvw = pd.concat([walmart_rv, ebay_rv], ignore_index=True)
rvw['price.extracted'] = rvw['price.extracted'].replace("nan", 99, regex=True)
rvw['reviews'] = rvw['reviews'].replace("nan", 99, regex=True)
rvw['price.extracted'] = pd.to_numeric(rvw['price.extracted'])
rvw['reviews'] = pd.to_numeric(rvw['reviews'])
rvw['brand'] = rvw['brand'].replace(r'^\s*$', 'lg', regex=True)


# In[52]:


fig = px.scatter(rvw, x="price.extracted", y="reviews",  color="market", symbol="brand", trendline="ols", trendline_scope="overall")
fig.show()

