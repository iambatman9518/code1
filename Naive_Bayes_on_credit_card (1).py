#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df=pd.read_csv("C:\\users\\nitin\\Desktop\\Machine Learning\\creditcard.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna()


# In[10]:


x=df.drop(['Class'],axis=1)


# In[11]:


y=df[['Class']]


# In[12]:


y.head()


# In[13]:


print(x.shape,y.shape)


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[15]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression as LR


# In[17]:


nb=GaussianNB()
nb.fit(x_train,y_train)


# In[18]:


predTrain=nb.predict(x_train)
from sklearn.metrics import roc_auc_score as roc, classification_report
print(classification_report(y_train,predTrain))
print("Accuracy: ",roc(y_train,predTrain))


# In[19]:


test_data = np.array([[0.1, -0.2, 1.5, 0.6, -0.7, 0.9, -0.1, 0.8, -1.2, 1.3, 
                       -0.4, 0.5, 0.7, -0.8, 0.2, 1.0, -0.9, 0.3, 1.2, -1.5, 
                       0.6, -0.3, 0.4, -0.6, 0.8, -1.1, 1.1, 0.5, 20000, 500]])

# Reshape the data to match the model input format
test_data = test_data.reshape(1, -1)

# Use the trained Naive Bayes model to predict
prediction = nb.predict(test_data)

# Print prediction (0 = non-fraud, 1 = fraud)
print(f"Prediction: {prediction[0]}")

