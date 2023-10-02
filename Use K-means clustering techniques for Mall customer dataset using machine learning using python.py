#!/usr/bin/env python
# coding: utf-8

# Use K-means clustering techniques for Mall customer dataset using machine learning using python

# In[8]:


pip install seaborn


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv("C:/Users/shreya/Downloads/Mall_Customers.csv")
df.head()


# In[8]:


# statistical info
df.describe()


# In[9]:


# datatype info
df.info()


# In[11]:


# Plot the count of each category in the 'Genre' column
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Count of each Genre')
plt.xticks(rotation=45)
plt.show()


# In[12]:


plt.figure(figsize=(10, 6))
sns.distplot(df['Age'], kde=True, bins=20) 
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of Age')
plt.show()


# In[13]:


plt.figure(figsize=(10, 6))
sns.distplot(df['Annual Income (k$)'], kde=True, bins=20)  
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.title('Distribution of Annual Income (k$)')
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
sns.distplot(df['Spending Score (1-100)'], kde=True, bins=20) 
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Density')
plt.title('Distribution of Spending Score (1-100)')
plt.show()


# In[15]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[16]:


df.head()


# In[17]:


df1 = df[['Annual Income (k$)', 'Spending Score (1-100)']]
df1.head()


# In[19]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.show()


# In[20]:


from sklearn.cluster import KMeans
errors = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df1)
    errors.append(kmeans.inertia_) 


# In[21]:


# plot the results for elbow method
plt.figure(figsize=(13,6))
plt.plot(range(1,11), errors)
plt.plot(range(1,11), errors, linewidth=3, color='red', marker='8')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1,11,1))
plt.show()


# In[22]:


km = KMeans(n_clusters=5)
km.fit(df1)
y = km.predict(df1)
df1['Label'] = y
df1.head()


# In[23]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1, hue='Label', s=50, palette=['red', 'green', 'brown', 'blue', 'orange'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.legend(title='Label')
plt.show()


# In[24]:


# cluster on 3 features
df2 = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
df2.head()


# In[25]:


errors = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df2)
    errors.append(kmeans.inertia_)


# In[26]:


# plot the results for elbow method
plt.figure(figsize=(13,6))
plt.plot(range(1,11), errors)
plt.plot(range(1,11), errors, linewidth=3, color='red', marker='8')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1,11,1))
plt.show()


# In[28]:


from sklearn.cluster import KMeans
import pandas as pd

# Assuming 'df2' contains the features for clustering

# Initialize the KMeans model with 5 clusters
km = KMeans(n_clusters=5)

# Fit the KMeans model to the data and get cluster labels
y = km.fit_predict(df2)

# Assign the cluster labels to the DataFrame
df2['Label'] = y

# Display the first few rows of the DataFrame with cluster labels
print(df2.head())


# In[29]:


# 3d scatter plot
fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df2['Age'][df2['Label']==0], df2['Annual Income (k$)'][df2['Label']==0], df2['Spending Score (1-100)'][df2['Label']==0], c='red', s=50)

ax.scatter(df2['Age'][df2['Label']==1], df2['Annual Income (k$)'][df2['Label']==1], df2['Spending Score (1-100)'][df2['Label']==1], c='green', s=50)

ax.scatter(df2['Age'][df2['Label']==2], df2['Annual Income (k$)'][df2['Label']==2], df2['Spending Score (1-100)'][df2['Label']==2], c='blue', s=50)

ax.scatter(df2['Age'][df2['Label']==3], df2['Annual Income (k$)'][df2['Label']==3], df2['Spending Score (1-100)'][df2['Label']==3], c='brown', s=50)

ax.scatter(df2['Age'][df2['Label']==4], df2['Annual Income (k$)'][df2['Label']==4], df2['Spending Score (1-100)'][df2['Label']==4], c='orange', s=50)

ax.view_init(30, 190)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.show()


# In[ ]:




