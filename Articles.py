#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd


# In[127]:


df = pd.read_csv('articles.csv', encoding='iso-8859-1')


# In[128]:


df = df.drop(columns=['Id', 'Article.Banner.Image'])


# In[129]:


df.head()


# In[130]:


from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer


# In[131]:


vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# In[132]:


import re
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def lower_column(text):
    return text.lower()


# In[133]:


df['Full_Article']= df['Full_Article'].apply(clean_text)
df['Article.Description']= df['Article.Description'].apply(clean_text)
df['Outlets']= df['Outlets'].apply(clean_text)
df['Heading']= df['Heading'].apply(clean_text)
df['Article_Type'] = df['Article_Type'].apply(clean_text)
df['Tonality'] = df['Tonality'].apply(clean_text)


# In[173]:


df = df.iloc[:-2]
df.head()


# In[174]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# X = vectorizer.encode(df['Full_Article'].tolist())


# In[175]:


X = tokenizer.encode(df['Heading'].tolist(), return_tensors="pt")


# In[176]:


columns = ['Full_Article', 'Article.Description', 'Outlets', 'Heading', 'Tonality']


# In[200]:


import numpy as np
new_df = pd.DataFrame()
X = []


# In[201]:


for col in columns:
    lst = tokenizer.encode(df[col].tolist())
    lst = lst[:-2]
    X.append(lst)


# In[202]:


X = np.array(X)
shp = X.shape
print(shp)
X = np.reshape(X, (shp[1], shp[0]))


# In[203]:


y = df['Article_Type'].factorize()[0]


# In[204]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[205]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)


# In[208]:


model = MLPClassifier()
model.fit(X_train, y_train)


# In[210]:


y_pred = model.predict(X_test)


# In[211]:


accuracy = np.mean(y_pred == y_test)


# In[212]:


print(accuracy)


# In[ ]:




