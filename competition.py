#!/usr/bin/env python
# coding: utf-8

# # Solving Machathon 1.0 Competetion

# In[89]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Data from drive since i am using google colab

# In[90]:


data = pd.read_csv('/content/drive/MyDrive/X_train1.csv')
data.head(5)


# Features are crammed up in a single string in the cuisine and restaurant columns. So we should split the whole string into an array of features in order to get their dummy values latter

# In[91]:


def translate_feature(s): #splittin the string values
  s = s.split("'")
  s.remove('[')
  s.remove(']')
  s = [x for x in s if x.strip()]
  return s

data['cuisine'] = data['cuisine'].apply(lambda x: translate_feature(x) if not ('array(' in x) else np.nan)
data['restaurant_features'] = data['restaurant_features'].apply(lambda x: translate_feature(x) if not ('array(' in x) else np.nan)


# Removing data that were entered incorrectly

# In[92]:


data = data.dropna(axis = 0)
data.tail(10)


# Creatting a Set object to get all uniques features

# In[93]:


cuisine_features = {'set_init'}
restaurant_features = {'set_init'}

def bag_of_cuisines(s, bag):
  for i in s:
    bag.add(i)

  return s

data['cuisine'] = data['cuisine'].apply(lambda x: bag_of_cuisines(x, cuisine_features))
data['restaurant_features'] = data['restaurant_features'].apply(lambda x: bag_of_cuisines(x, restaurant_features))

cuisine_features.remove('set_init')
restaurant_features.remove('set_init')


# In[94]:


cuisine_features = sorted((list(cuisine_features)))


# In[115]:


restaurant_features = sorted((list(restaurant_features)))


# Getting the dummies of the features to insert them into the machine learning model

# In[97]:


cuisine_dummies = []
restaurant_dummies = []

def manual_dummies(bag, dummies, features):
  dummies.append([1 if i in bag else 0 for i in features])

data['cuisine'].apply(lambda x: manual_dummies(x, cuisine_dummies, cuisine_features))
data[cuisine_features] = cuisine_dummies
data.rename(columns={'Grill': 'Cuisine Grill'}, inplace=True)

data['restaurant_features'].apply(lambda x: manual_dummies(x, restaurant_dummies, restaurant_features))
data[restaurant_features] = restaurant_dummies
data.rename(columns={'Grill': 'Restaurant Feature Grill'}, inplace=True)


# Now we get the dummies of the other categorical data we have

# In[98]:


data.drop(columns=['cuisine', 'restaurant_features'], inplace=True)

y = data['Ratings']
data.drop(columns=['Ratings'], inplace=True)


# In[99]:


area = data['area']
area_dummies = pd.get_dummies(area)

data.drop(columns=['area'], inplace=True)
data[area_dummies.columns] = area_dummies


# Changing discount values from 0 and 1 to strings in order to create columns from

# In[100]:


data['discounted'] = data['discounted'].apply(lambda x: 'Got Discount' if x else "Didn't Get Discount")
data.head(15)


# In[101]:


discount = data['discounted']
discount_dummies = pd.get_dummies(discount)

data.drop(columns=['discounted'], inplace=True)
data[discount_dummies.columns] = discount_dummies


# In[102]:


data.head()


# Getting our data ready to insert it into our model

# In[103]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[104]:


data['Ratings'] = y
data.head()


# Normalizing the data

# In[105]:


scaler = StandardScaler()
data.iloc[:, 2:-1] = scaler.fit_transform(data.iloc[:, 2:-1])


# In[106]:


data.head()


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:, 2:-1].values, data.iloc[:, -1].values, test_size = 0.2, random_state = 0)


# We will use decision tree classification since the ratings are from 1-5

# In[108]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# calculating F1 score for accuracy

# In[114]:


from sklearn.metrics import f1_score
f1_score(y_test, model.predict(X_test), average='micro')


# In[ ]:




