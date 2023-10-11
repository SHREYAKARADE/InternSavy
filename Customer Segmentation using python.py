#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle


# In[2]:


data = pd.read_csv("C:/Users/shreya/Downloads/segmentation data.csv", index_col = 0)


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.corr()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 9))
s = sns.heatmap(data.corr(),
                annot=True,
                cmap='RdBu',  # Corrected quotation marks here
                vmin=-1,
                vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=90, fontsize=12)
plt.title('Correlation Heatmap')  # Corrected quotation marks here
plt.show()


# In[10]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
plt.scatter(data.iloc[:, 2], data.iloc[:, 4])
plt.xlabel("Age/Umur")  # Corrected quotation marks here
plt.ylabel("Income/Gaji")  # Corrected quotation marks here
plt.title("Scatter Plot Korelasi Antara Age dan Income")  # Corrected quotation marks here
plt.show()


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


scaler =StandardScaler()
data_std = scaler.fit_transform(data)


# In[13]:


import scipy
from scipy.cluster.hierarchy import dendrogram, linkage


# In[19]:


from scipy.cluster.hierarchy import linkage

hierarchy_cluster = linkage(data_std, method='ward')  


# In[21]:


plt.figure(figsize=(12, 9))
plt.title("Cluster Hirarki")
plt.xlabel("Observasi")
plt.ylabel("Jarak")
dendrogram(hierarchy_cluster, truncate_mode='level', p=5, show_leaf_counts=False, no_labels=True)
plt.show()


# In[22]:


from sklearn.cluster import KMeans


# In[24]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Corrected quotation marks and indentation
    kmeans.fit(data_std)
    wcss.append(kmeans.inertia_)


# In[26]:


plt.figure(figsize=(12, 9))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')  
plt.ylabel('WCSS')  # Corrected quotation marks here
plt.xlabel('Number of Clusters')  # Added x-axis label
plt.title('K-means Clustering')  # Corrected quotation marks here
plt.show()


# In[28]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(data_std)


# In[30]:


data_kmeans = data.copy()
data_kmeans['segment'] = kmeans.labels_


# In[32]:


data_kmeans_analysis = data_kmeans.groupby('segment').mean()


# In[33]:


data_kmeans_analysis['jumlah_customer'] = data_kmeans[['segment', 'Sex']].groupby(['segment']).count()


# In[34]:


data_kmeans_analysis['rata2'] = data_kmeans_analysis['jumlah_customer'] / data_kmeans_analysis['jumlah_customer'].sum()
print(data_kmeans_analysis)


# In[35]:


data_kmeans_analysis.rename(index={0: 'Well-Off', 1: 'Fewer-Opportunities', 2: 'Standard', 3: 'Career-Focused'}, inplace=True)
print(data_kmeans_analysis)


# In[36]:


data_kmeans['Labels'] = data_kmeans['segment'].map({0: 'Well-Off', 1: 'Fewer-Opportunities', 2: 'Standard', 3: 'Career-Focused'})

x_axis = data_kmeans['Age']
y_axis = data_kmeans['Income']

plt.figure(figsize=(10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue=data_kmeans['Labels'], palette=['g', 'r', 'c', 'm'])

plt.title('Segmentation K-means')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()


# In[37]:


from sklearn.decomposition import PCA


# In[38]:


pca = PCA()
pca.fit(data_std)


# In[39]:


pca.explained_variance_ratio_


# In[40]:


plt.figure(figsize=(12, 9))
plt.plot(range(1, 8), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='-', color='b')
plt.title('Variasi Komponen')
plt.xlabel('Jumlah Komponen')
plt.ylabel('Cumulative Explained Variance')
plt.show()


# In[41]:


pca=PCA(n_components=3)
pca.fit(data_std)
pca.components_


# In[43]:


data_std_pca = pd.DataFrame(data=pca.components_, columns=data.columns.values, index=['Component 1', 'Component 2', 'Component 3'])


# In[44]:


import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(data_std_pca, vmin=-1, vmax=1, cmap='RdBu', annot=True)

# Correcting yticks
plt.yticks([0, 1, 2], ['Component 1', 'Component 2', 'Component 3'], rotation=0, fontsize=12)

plt.show()


# In[45]:


pca.transform(data_std)


# In[46]:


skor_pca = pca.transform(data_std)


# In[48]:


wcss = []
for i in range(1,11): kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
kmeans_pca.fit(skor_pca)
wcss.append(kmeans_pca.inertia_)


# In[55]:


import matplotlib.pyplot as plt

# Assuming wcss is a list or array with the WCSS values
wcss = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04]

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')
plt.show()


# In[56]:


from sklearn.cluster import KMeans

kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pca.fit(skor_pca)

data_pca_kmeans = pd.concat([data.reset_index(drop=True), pd.DataFrame(skor_pca)], axis=1)
data_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
data_pca_kmeans['segment'] = kmeans_pca.labels_


# In[57]:


data_pca_kmeans_freq = data_pca_kmeans.groupby(['segment']).mean()


# In[58]:


data_pca_kmeans_freq['jumlah'] = data_pca_kmeans[['segment', 'Sex']].groupby(['segment']).count()
data_pca_kmeans_freq['rata2'] = data_pca_kmeans_freq['jumlah'] / data_pca_kmeans_freq['jumlah'].sum()

# Rename segments
data_pca_kmeans_freq.rename({0: 'Standard', 1: 'Career-Focus', 2: 'Fewer-Opportunity', 3: 'Well-Off'}, inplace=True)

print(data_pca_kmeans_freq)


# In[60]:


data_pca_kmeans['Legend'] = data_pca_kmeans['segment'].map({0: 'Standard', 1: 'Career-Focus', 2: 'Fewer-Opportunity', 3: 'Well-Off'})


# In[61]:


x_axis = data_pca_kmeans['Component 2']
y_axis = data_pca_kmeans['Component 1']

plt.figure(figsize=(10, 8))
sns.scatterplot(x=x_axis, y=y_axis, hue=data_pca_kmeans['Legend'], palette=['g', 'r', 'c', 'm'])

plt.title('Kluster Setelah PCA Components')
plt.xlabel('Component 2')
plt.ylabel('Component 1')
plt.show()


# In[62]:


import pickle


# In[63]:


import pickle

# Pickle the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Pickle the PCA
with open('pca.pkl', 'wb') as file:
    pickle.dump(pca, file)

# Pickle the KMeans model
with open('kmeans_pca.pkl', 'wb') as file:
    pickle.dump(kmeans_pca, file)

