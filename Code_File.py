#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES
# 

# In[2]:


import pandas as pd    

import numpy as np

import seaborn as sns       #Importin seabor library for interactive visualization

import matplotlib.pyplot as plt    #Importing pyplot interface using matplotlib

from sklearn.preprocessing import StandardScaler    #Importing StandardScaler using sklearn library

from sklearn.model_selection import train_test_split   #To split the data in training and testing part  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score   #To generate classification report and acuracy score 


# ---------

#  # Data Acquisition & Description

# In[3]:


# Importing Dataset.
credit_card = pd.read_csv("C:\Home\Data\BIA project\Model\creditcard_2023.csv")


# In[3]:


# Printing the number of rows and columns of a dataset as well as Dataset.
print('Shape of our Dataset -',credit_card.shape)
credit_card


# In[6]:


# Gives a summary of the dataset including column names, data types, non-null values, and memory usage.
credit_card.info()


# In[7]:


# Returns the data type of each feature of a dataset.
credit_card.dtypes


# In[8]:


# Displays the first five rows of a dataset.  
credit_card.head()


# In[9]:


# Displays the last five rows of  dataset.
credit_card.tail()


# In[10]:


# Returns a list of column names in the dataset.
credit_card.columns


# In[5]:


#Gives descriptive statistics of a dataset. 
credit_card.describe().round(3)


# In[44]:


print("********* Amount Lost due to fraud:************\n")
print("Total amount lost to fraud")
print(credit_card.Amount[credit_card.Class == 1].sum())
print("Mean amount per fraudulent transaction")
print(credit_card.Amount[credit_card.Class == 1].mean().round(4))
print("Compare to normal transactions:")
print("Total amount from normal transactions")
print(credit_card.Amount[credit_card.Class == 0].sum())
print("Mean amount per normal transactions")
print(credit_card.Amount[credit_card.Class == 0].mean().round(4))


# ### Observations
# ###### ⪼  We have 568630 Rows of observations having 31 columns.
# ###### ⪼  'Class' is our Output feature indicating whether the transaction is "Fraudulent" (1) or "Not Fraudulent" (0).
# ###### ⪼  "V1-V28"Anonymized features representing various transaction attributes.
# ###### ⪼  dtype(data type) of all the features looks perfect.

# -------------

# # DATA PREPROCESSING

# In[12]:


# Checking null values in the dataset
print(credit_card.isnull().sum())


# In[14]:


# Checking duplicate values in the dataset
credit_card.duplicated().any()


# ### Observations
# ##### ⪼ No missing values.
# ##### ⪼ No duplicates.

# ------------------

# # EDA (Exploratory Data Analysis)

# In[50]:


# Observing the Distribution of Feature V1
plt.figure(figsize=(9, 4.5))
sns.histplot(credit_card['V1'], bins=25, kde=True, color='darkblue')
plt.title('Distribution of Feature V1')
plt.xlabel('V1 Value')
plt.ylabel('Frequency')
plt.show()


# In[51]:


# Observing the Distribution of Feature V9
plt.figure(figsize=(9, 4.5))
sns.histplot(credit_card['V9'], bins=25, kde=True, color='green')
plt.title('Distribution of Feature V9')
plt.xlabel('V9 Value')
plt.ylabel('Frequency')
plt.show()


# In[54]:


# Observing the Distribution of Feature V17
plt.figure(figsize=(9, 4.5))
sns.histplot(credit_card['V17'], bins=25, kde=True, color='darkblue')
plt.title('Distribution of Feature V17')
plt.xlabel('V17 Value')
plt.ylabel('Frequency')
plt.show()


# In[53]:


# Observing the Distribution of Feature V26
plt.figure(figsize=(9, 4.5))
sns.histplot(credit_card['V26'], bins=25, kde=True, color='green')
plt.title('Distribution of Feature V26')
plt.xlabel('V26 Value')
plt.ylabel('Frequency')
plt.show()


# In[55]:


# Observing the Amount Disribution 
sns.kdeplot(data= credit_card['Amount'],color = 'blue', fill=True)
plt.title('Amount Distribution',size=14)
plt.show()


# In[57]:


# Observing the Disribution of the Feature 'Class'
colors = ['blue', 'green']
explode = [0.1, 0]
credit_card['Class'].value_counts().plot.pie(
    explode=explode,
    autopct='%3.1f%%',
    shadow=True,
    legend=True,
    startangle=45,
    colors=colors,  
    wedgeprops=dict(width=0.4) 
)

plt.title('Distribution of Class',size=14)
plt.show()


# In[68]:


# Observing the Disribution of the Feature 'Class'
plt.figure(figsize=(8, 6))
credit_card['Class'].value_counts().plot(kind='bar', color=['blue', 'green'])
plt.title('Distribution of Classes (0: Non-Fraudulent, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()


# In[59]:


# Pulling the highest correlated feature to the feature 'Class'
corrmat = credit_card.corr()
cols = corrmat.nlargest(15,'Class')['Class'].index
cols


# In[60]:


# Pulling the least correlated feature to the feature 'Class'
cols_negative = corrmat.nsmallest(15,'Class')['Class'].index
cols_negative


# In[61]:


# Joining the two above variables in one variable 'Credit_card'
Credit_card = []
for i in cols:
    Credit_card.append (i)
for j in cols_negative:
    Credit_card.append(j)

Credit_card


# In[66]:


# Observing the Correlation between features using a heatmap
corrmat = credit_card[Credit_card].corr()
sns.set(font_scale=1.15)
f, ax = plt.subplots(figsize=(12,12))
hm = sns.heatmap(corrmat,
                 cmap='PuBu',
                 cbar=True, 
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 7},
                 yticklabels=corrmat.columns,
                 xticklabels=corrmat.columns)


# -------------

# # SPLIT DATA INTO TEST AND TRAIN

# In[6]:


# Split the data into features (X) and target (y).
x = credit_card.drop(['id','Class'],axis=1)
y = credit_card.Class


# In[7]:


# Standardize the feature data (x)   
scaler = StandardScaler()
X = scaler.fit_transform(x)
print(X)


# In[8]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


# -------

# # MODEL SELECTION AND TRAINING

# ### i) RandomForestClassifier

# In[21]:


from sklearn.ensemble import RandomForestClassifier


# In[22]:


rf = RandomForestClassifier()


# In[23]:


rf.fit(X_train, y_train)


# In[24]:


y_pred_rf = rf.predict(X_test)


# In[25]:


print("Randon Forest Classifier")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_rf)*100,"%")


# In[26]:


RandomForestClassifier = accuracy_score(y_test, y_pred_rf)*100


# ### ii) Support Vector Machine (SVM)

# In[27]:


from sklearn.svm import SVC


# In[28]:


clf = SVC()


# In[29]:


clf.fit(X_train, y_train)


# In[30]:


y_pred_svm = clf.predict(X_test)


# In[31]:


print("Support Vector Machine")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_svm)*100,"%")


# In[32]:


Support_Vector_Machine = accuracy_score(y_test, y_pred_svm)*100


# ### iii) Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


reg = LogisticRegression()
reg.fit(X_train, y_train)


# In[35]:


reg.coef_


# In[36]:


reg.intercept_


# In[37]:


y_pred_reg =  reg.predict(X_test)


# In[38]:


print("Logistic Regression Model")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_reg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_reg))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_reg)*100,"%")


# In[39]:


Logistic_Regression = accuracy_score(y_test, y_pred_reg)*100


# ### iv) Gradient Boosting Classifier (XGBoost)

# In[14]:


get_ipython().system('pip3 install xgboost')


# In[40]:


from xgboost import XGBClassifier


# In[41]:


xgb = XGBClassifier()


# In[42]:


xgb.fit(X_train, y_train)


# In[43]:


y_pred_xgb = xgb.predict(X_test)


# In[44]:


print("XGBoost Model")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_xgb)*100,'%')


# In[45]:


XGBoost = accuracy_score(y_test, y_pred_xgb)*100


# ------

# # EVALUATION OF MODELS

# In[47]:


# Assuming you have accuracy values for four models
model_names = ['RandomForestClassifier', 'Support Vector Machine ', 'Logistic Regression', 'XGBoost']
accuracy_values = [RandomForestClassifier, Support_Vector_Machine, Logistic_Regression, XGBoost]  # Replace with your actual accuracy values

# Create a bar plot
bars = plt.bar(model_names, accuracy_values, color=['blue', 'lightblue', 'lightgreen', 'green'])

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Four Models')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Add values on top of each bar
for bar, value in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2f}', ha='center', va='bottom')

# Display the plot
plt.show()


# In[48]:


cm_1 = confusion_matrix(y_test, y_pred_rf)
cmn_1 = cm_1.astype('float') / cm_1.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cmn_1, annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# In[49]:


cm_2 = confusion_matrix(y_test, y_pred_svm)
cmn_2 = cm_2.astype('float') / cm_2.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cmn_2, annot=True, fmt='.2%', cmap='Greens')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# In[50]:


cm_3 = confusion_matrix(y_test, y_pred_reg)
cmn_3 = cm_3.astype('float') / cm_3.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cmn_3, annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# In[51]:


cm_4 = confusion_matrix(y_test, y_pred_xgb)
cmn_4 = cm_4.astype('float') / cm_4.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cmn_4, annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




