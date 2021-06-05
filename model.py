#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# In[1]:


## Data Analysis Phase
## Main aim is to understand more about the data

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
## Display all columns of the dataframe

pd.pandas.set_option('display.max_columns', None)


# In[2]:


dataset = pd.read_csv('D:\ML project\env\data.csv')

## print shape of the dataset
print(dataset.shape)


# In[3]:


dataset.head()


# In[4]:


dataset.drop(['id', 'Unnamed: 32'], axis=1, inplace = True)


# In[5]:


## Here we will check the percentage of missing values in each feature
## Step - 1: make the list of features which have missing values

features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]

## Step - 2: Print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4), '%missing values')


# In[6]:


## NO missing values available in the dataset


# In[7]:


## LabelEncoding (Convert the value of M and N into 1 and 0)
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
dataset.iloc[:, 0] = labelEncoder_y.fit_transform(dataset.iloc[:, 0].values)


# In[8]:


dataset.head()


# In[9]:


continous_features = dataset.drop(['diagnosis'], axis=1)
continous_features.head()


# In[10]:


## lets analyze the continous values by creating histogram to understand the distribution
for feature in continous_features:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()


# In[11]:


## The data is not distributed Normally


# ## Outliers

# In[12]:


## Check and removing Outliers
for feature in continous_features:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[13]:


### There is so many outliers


# # Feature Selection

# ## Correlation

# In[14]:


corr = dataset.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot= True, annot_kws={'size':15}, cmap='GnBu')
plt.show()


# ### Data Preprocessing

# In[15]:


dataprocessed = dataset.drop(['diagnosis'], axis=1)


# In[16]:


dataprocessed.head()


# In[17]:


corr = dataprocessed.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot= True, annot_kws={'size':10}, cmap='GnBu')
plt.show()


# In[18]:


droplist = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 
           'concave points_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
           'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst']
dataprocessed = dataprocessed.drop(droplist, axis=1)


# In[19]:


dataprocessed.head()


# In[20]:


for feature in dataprocessed.columns:
    sns.displot(dataprocessed[feature])


# In[21]:


def outlierLimit(column):
    q1, q3 = np.nanpercentile(column, [25, 75])
    iqr = q3 - q1
    
    uplimit = q3 + 1.5*iqr
    lowlimit = q1 - 1.5*iqr
    return uplimit, lowlimit


# In[22]:


for column in dataprocessed.columns:
    if dataprocessed[column].dtype != 'object':
        uplimit, lowlimit = outlierLimit(dataprocessed[column])
        dataprocessed[column] = np.where((dataprocessed[column]>uplimit) | (dataprocessed[column]<lowlimit), np.nan, dataprocessed[column])


# In[23]:


dataprocessed.isnull().sum()


# In[24]:


## Now you can see we change outliers into Nan values


# In[25]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=4)
dataprocessed.iloc[:, :] = imputer.fit_transform(dataprocessed)


# In[26]:


dataprocessed.isnull().sum()


# In[27]:


dataprocessed.head()


# # Model Training and Testing

# In[28]:


y = dataset['diagnosis']
X = dataprocessed


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


def models(X_train, y_train):
    ## LogisticRegression 
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    ## DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=42, criterion='entropy')
    tree.fit(X_train, y_train)
    
    ##  Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators = 10)
    forest.fit(X_train, y_train)
    
    print('[0]LogisticRegression Accuracy: ', lr.score(X_train, y_train))
    print('[0]DecisionTreeClassifier Accuracy: ', tree.score(X_train, y_train))
    print('[0]Random Forest Accuracy: ', forest.score(X_train, y_train))
    return lr, tree, forest


# In[32]:


model = models(X_train, y_train)


# In[33]:


from sklearn.metrics import classification_report, accuracy_score, recall_score

for i in range(len(model)):
    print("Model",i)
    print(classification_report(y_test, model[i].predict(X_test)))
    print(accuracy_score(y_test, model[i].predict(X_test)))
    print(recall_score(y_test, model[i].predict(X_test)))


# In[34]:


# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# print('Recall: {}'.format(recall_score(y_test, y_pred)))


# In[39]:


##  Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators = 10)
forest.fit(X_train, y_train)


# In[40]:


import pickle 
model = pickle.dump(forest, open('model.pkl', 'wb'))


# In[ ]:




