#!/usr/bin/env python
# coding: utf-8

# Problem Statment : This dataset is created for prediction of Graduate Admissions from an Indian perspective
# 

# The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are : 
#  
# 1. GRE Scores ( out of 340 ) 
# 2. TOEFL Scores ( out of 120 ) 
# 3. University Rating ( out of 5 ) 
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
# 5. Undergraduate GPA ( out of 10 ) 
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 ) is the Target Variable
# 

# Models
# 
# 1. Logistic Regression 
# 2. Random Forest 
# 3. Support Vector Machine 
# 
# This Model will help Beginners as well Intermediate ML Practitioners models will be introduced. Also measure the accuracy of models that are built by using Machine Learning, and further development will be assessed.

# 1. Load the Data and Analysis of Variables

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


# Read the Data and check the Column Names to check the Space in the Names
df=pd.read_csv("C:/Users/shreya/Downloads/Admission_Predict_Ver1.1.csv")
col_names=df.columns.tolist()
print("Column names:")
print(col_names)
print("\nSample Data:")
print(df.head())


# In[6]:


df=df.rename(columns={'Serial No.':'no','GRE Score':'gre','TOEFL Score':'toefl','University Rating':'rating','SOP':'sop','LOR ':'lor',
                           'CGPA':'gpa','Research':'research','Chance of Admit ':'chance'})


# In[7]:


df.dtypes


# In[8]:


print('Shape of the data:')
df.shape


# In[9]:


print('Missing values in columns:')
df.isnull().sum()


# Data Exploration
# 
# First of all, let us see the basic statistics of the data.
# 

# In[10]:


df.describe()


# 
# This shows some descriptive statistics on the data set. Notice, it only shows the statistics on the numerical columns. From here you can see the following statistics:
# 
# Row count, which aligns to what the shape attribute showed us.
# 
# * The mean, or average.
# * 50% aka Median it is very import to understand Skeweness of the Data 
# * The standard deviation, or how spread out the data is.
# * The minimum and maximum value of each column
# * The number of items that fall within the first, second, and third percentiles.
# 
# Second, let us analyze the distribution of subjects' chance of admit in the data. Let us see how rating affects chance of admission:

# In[11]:


df.groupby('rating').mean()


# Average chance of admission of subjects which applied to program with rating 1 is less than that of the subjects which applied to program with higher ratings.
# Now let us analyze the subjects with more than 82% of chance which is the third quartile of the chance data.

# In[12]:


df[df['chance']>0.82].groupby('chance').mean()


# Several observations:
# Average GPAs of those with higher level of chance to admit is greater than 9 where the data average is 8.57.
# Average LOR, SOP, GRE and TOEFL grades of those with higher level of chance to admit is greater than the data average.
# Research Shows us 75% Research Experience so there may be chances to Reject the Application in the next we take more than 83%
# 

# Data Visualization

# Histogram of Admission Chance : 
#     
# Let us visualize our data to get a much clearer picture of the data and the significant features.
# 

# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(df['chance'],bins=10,color="orange")
plt.title('Histogram of Admission Chance')
plt.xlabel('Admission Chance')
plt.ylabel('Frequency of Chance')
plt.show()


# Histogram shows us that 'chance of admission' column is well distributed in data.
# 
# Line Plot for Research Output and The Chance of Admission

# In[14]:


plt.figure(figsize=(12,8))
plt.plot(range(len(df[df['research']==1])), df[df['research']==1]['chance'], color='orange')
plt.plot(range(len(df[df['research']==0])), df[df['research']==0]['chance'], color='olive')
plt.show()


# The chance of admission depends a great deal on the subjects' research output; hence, research can be a good predictor in predicting the outcome.
# 
# Scatter Plot of University Rating and The Chance of Admission

# In[15]:


df.boxplot(column='chance',by='rating',grid=False,figsize=(12,8))
plt.title('The Chance of Admission for University Ratings')
plt.xlabel('University Rating')
plt.ylabel('Chance of Admission')
plt.show()


# As it can be seen from the boxplot chart, chance of admission is higher within the applicants of lower rated (5) universities.
# 
# Histogram of Numeric Variables

# In[16]:


df.hist(bins=10, figsize=(20,15))
plt.show()


# Data Preperation
# 
# As it  can be seen from Data Preprocessing section above, graduate admission data is only include numerical variables. Hence, only the following steps should be implemented before model devolopment:
# 
# The outcome variable is 'chance', and all other features are predictors.
# 'no' variable should be dropped from dataset since it only indicator of the instances (means ID Column)

# In[17]:


df.dtypes


# In[18]:


df.drop(['no'],axis=1,inplace=True)
var=df.columns.values.tolist()
y=df['chance']
x=[i for i in var if i not in ['chance']]
x=df[x]


# Data will be splitted using train_test_split module of scikitlearn library where splitting ratio is chosen as 20% for test data.

# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)


# Data normalization is important in order to represent data in comparable scales. 

# In[20]:


from sklearn.preprocessing import MinMaxScaler
xs=MinMaxScaler()
x_train[x_train.columns] = xs.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = xs.transform(x_test[x_test.columns])


#  Classification Models & Furher Analysis
# 
# In order to make classification models, outcome of the classification is defined as 'status' which takes value of True for the applicants that have more than 83% chance. Third quartile, 83%, is chosen as threshold since median of the chance data, 72%, is pretty high.

# In[21]:


import numpy as np
cy_train=[1 if chance > 0.83 else 0 for chance in y_train]
cy_train=np.array(cy_train)

cy_test=[1 if chance > 0.83 else 0 for chance in y_test]
cy_test=np.array(cy_test)



# Logistic Regression

# In[24]:


# Fitting logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, cy_train)


# Printing accuracy score & confusion matrix
from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(cy_test, lr.predict(x_test))))
print('--------------------------------------')
from sklearn.metrics import classification_report
print(classification_report(cy_test, lr.predict(x_test)))

cy = lr.predict(x_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
lr_confm = confusion_matrix(cy, cy_test)
sns.heatmap(lr_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Logistic Regression')
plt.show()


# Random Forest

# In[25]:


# Fitting random forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, cy_train)

# Printing accuracy score & confusion matrix
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(cy_test, rf.predict(x_test))))
print('--------------------------------------')
print(classification_report(cy_test, rf.predict(x_test)))

cy = rf.predict(x_test)
rf_confm = confusion_matrix(cy, cy_test)
sns.heatmap(rf_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.show()


# Support Vector Machine

# In[27]:


# Fitting support vector machine model
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, cy_train)

# Printing accuracy score & confusion matrix
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(cy_test, svc.predict(x_test))))
print('--------------------------------------')
print(classification_report(cy_test, svc.predict(x_test)))

cy = svc.predict(x_test)
svc_confm = confusion_matrix(cy, cy_test)
sns.heatmap(svc_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.show()


# How often does my classifier predict admitted students correctly? This measurement is called “recall” and a quick look at these diagrams can demonstrate that models ara close but logistic regression and support vector machine models are slightly better for this criteria. Out of 20 True admitted applicants, LR and SVM are correctly classified 18 of them which means the models are approximately have 90% of "recall".
# 
# How often does my classifier predict students that will be admitted correctly? This measurement is called “precision” and Random Forest performs slightly better (17 out of 18) than other classifiers.
# 
# All classifiers have the same accuracy score which is 96%.
# Feature Importance for Random Forest Model

# In order to represent the most important features which influence whether an applicant is admitted in college, random forest classifier is used.

# In[28]:


f_imp=pd.Series(rf.feature_importances_,index=x_train.columns).sort_values(ascending=False)
print(f_imp)


# In[ ]:




