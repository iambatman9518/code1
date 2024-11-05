#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from apyori import apriori


# In[3]:


df=pd.read_csv("C:\\users\\nitin\\Downloads\\archive (4)\\Market_Basket_Optimisation.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


record=[]
for i in range(0,7500):
    record.append([str(df.values[i,j]) for j in range(0,20)])


# In[7]:


association_rules=apriori(record,min_support=0.0045,min_confidence=0.20,min_lift=3,min_length=2)
association_result=list(association_rules)


# In[8]:


print(len(association_result))


# In[9]:


print(association_result[0])


# In[12]:


for item in association_result:
    pair=item[0]
    items=[x for x in pair]
    print("rule: "+ items[0] +"->"+ items[1])
    print("support: "+str(item[1]))
    print("confidence: "+str(item[2][0][2]))
    print("Lift: "+str(item[2][0][3]))
    print("............................................")


# In[ ]:




