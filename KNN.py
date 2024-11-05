#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\nitin\\Downloads\\knn_sample_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


x=df.drop(['Class'],axis=1)
y=df['Class']


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)


# In[10]:


y_pred=knn.predict(x_train)


# In[11]:


y_pred


# In[14]:


from sklearn.metrics import accuracy_score
print("accuracy: ",accuracy_score(y_train,y_pred))


# In[16]:


import seaborn as sns
sns.scatterplot(x='Height', y='Weight', hue='Class', data=df, palette='coolwarm')
plt.title('Height vs. Weight by Class')
plt.show()


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:\\Users\\nitin\\Downloads\\knn_sample_dataset.csv")

# Separate the features (Height, Weight, Age) and the target (Class)
X = df[['Height', 'Weight', 'Age']]
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
accuracy = knn.score(X_test_scaled, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')


# In[27]:


# New data point: Height=5.7, Weight=69, Age=26
new_data = [[5.7, 69, 26]]

# Standardize the new data point using the same scaler
new_data_scaled = scaler.transform(new_data)

# Predict the class using the trained KNN model
predicted_class = knn.predict(new_data_scaled)

print(f'The predicted class for the new data point is: {predicted_class[0]}')


# In[ ]:




