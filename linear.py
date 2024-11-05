#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


df=pd.read_csv("C:\\Users\\nitin\\Downloads\\Salary_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df=df.drop(['Unnamed: 0'],axis=1)


# In[7]:


df.shape


# In[8]:


x=df['YearsExperience']
y=df['Salary']


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=3)


# In[12]:


x_train.shape


# In[16]:


y_test.shape


# In[21]:


x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)


# In[19]:


from sklearn.linear_model import LinearRegression 
LR = LinearRegression()
LR.fit(x_train,y_train)


# In[22]:


y_pred=LR.predict(x_test)


# In[26]:


y_test = np.array(y_test).reshape(-1, 1)


# In[28]:


from sklearn.metrics import mean_squared_error, r2_score

# Predictions
y_pred = LR.predict(x_train)

# Evaluation
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# In[30]:


import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='blue',label='Actual point')
plt.plot(x_train,y_pred,color='red',label='Regression_Line')


# In[ ]:


#poly


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = np.array(x_train).reshape(-1, 1)

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_train)

model = LinearRegression()
model.fit(x_poly, y_train)

y_pred = model.predict(x_poly)

plt.scatter(x_train, y_train, color='blue', label='Actual Data')

sorted_indices = np.argsort(x_train.flatten())
plt.plot(x_train[sorted_indices], y_pred[sorted_indices], color='red', label=f'Polynomial Regression (degree {degree})')

# Adding labels and title
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.title(f'Polynomial Regression (degree {degree})')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[ ]:




