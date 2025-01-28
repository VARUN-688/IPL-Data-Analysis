#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("matches.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.date=pd.to_datetime(df.date)


# In[7]:


df.dtypes


# In[8]:


df.season.value_counts()


# In[9]:





# In[9]:


df.season.unique()


# In[10]:


df.isnull().sum()


# In[11]:


df.city.unique()


# In[12]:


df.loc[df.city.isnull()]


# In[13]:


df.city.fillna("Dubai",inplace=True)


# In[14]:


df.city.isnull().sum()


# In[15]:


df.team1.unique()


# In[16]:


df.team1=df.team1.apply(lambda x:"".join(w[0] for w in x.split()))


# In[17]:


df.team1.unique()


# In[18]:


df.team2=df.team2.apply(lambda x:"".join(w[0] for w in x.split()))


# In[19]:


df.toss_winner=df.toss_winner.apply(lambda x:"".join(w[0] for w in x.split()))


# In[20]:


df.winner=df.winner.apply(lambda x:"".join(w[0] for w in x.split()) if pd.notna(x) else x)


# In[21]:


for x in ['team1','team2','toss_winner','winner']:
    print(df.loc[:,f'{x}'].unique())


# In[23]:


df['venue'] = df['venue'].str.replace('.', '', regex=True)
df['venue'] = df['venue'].apply(lambda x: x.split(',')[0])

df.venue.value_counts()


# In[24]:


df.to_csv("matches_cleaned_data.csv")


# In[ ]:




