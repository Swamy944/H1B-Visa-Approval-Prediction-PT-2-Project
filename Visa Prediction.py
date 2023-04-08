#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[99]:


data = pd.read_csv('h1b_kaggle.csv')


# In[100]:


data


# In[101]:


data.shape


# In[102]:


data.head()


# In[103]:


data.tail()


# In[104]:


data.info()


# In[105]:


data.describe()


# In[106]:


print(data['YEAR'].unique())


# # Data Visualization

# In[107]:


# Number of H1B Visa applications year wise from 2011 to 2016
fig = plt.figure(figsize=(12,10), facecolor='white')
ax1 = fig.add_axes([0,0,.4,.4])

year = data['YEAR'].value_counts()   
ax1.bar(x=year.index, height=year[:])
ax1.set_title('Number of H-1B Applications Year Wise')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Applications')
plt.show()


# In[108]:


# Ploting the case status in a piechart
status = data['CASE_STATUS'].value_counts()
fig = plt.figure(figsize = (10,6))
plt.pie(labels=status.index[:-3], x=status[:-3],autopct='%.f%%', shadow=True, textprops={'fontsize':16})
plt.show()


# In[109]:


#Top 10 Job titles
fig = plt.figure(figsize = (12,10))
ax2 = fig.add_axes([0,0,.4,.4])
job_title = data['JOB_TITLE'].value_counts()[:10]

ax2.bar(x=job_title.index,height=job_title[:])
ax2.set_xticklabels(labels=job_title.index ,rotation=90)
ax2.set_title('Top 10 Desirable Job Titles')
ax2.set_xlabel('Job Titles')
ax2.set_ylabel('Number of Applicants')
plt.show()


# In[110]:


# Ploting companies who sent highgest H-1B visa applications
company = data['EMPLOYER_NAME'].value_counts()[:20]
plt.figure(figsize=(15,8))
plt.bar(x=company.index, height=company[:])
plt.title('Top 20 Companies who sent highest H-1B visa applications')
plt.xlabel('Company')
plt.ylabel('Number of Applicants')
plt.xticks(rotation=90)
plt.show()


# In[111]:


plt.figure(figsize=(12,7))
sns.set(style = 'whitegrid')
g = sns.countplot(x = 'FULL_TIME_POSITION',data = data)
plt.title('No of applications made for the full time position')
plt.ylabel('No of petitions made')
plt.show()


# In[112]:


top_emp = list(data['EMPLOYER_NAME'][data['YEAR'] >= 2015].groupby(data['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).index)

byempyear = data[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][data['EMPLOYER_NAME'].isin(top_emp)]

byempyear = byempyear.groupby([data['EMPLOYER_NAME'], data['YEAR']])


# In[113]:


plt.figure(figsize=(12,7))

markers=['o','v','^','<','>','d','s','p','*','h','x','D','o','v','^','<','>','d','s','p','*','h','x','D']

for company in top_emp:
    tmp = byempyear.count().loc[company]
    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[top_emp.index(company)])
plt.xlabel("Year")
plt.ylabel("Number of Applications")
plt.legend()
plt.title('Number of Applications of Top 10 Applicants')
plt.show()


# In[114]:


#Average salary of each company
plt.figure(figsize=(12,7))

for company in top_emp:
    tmp = byempyear.mean().loc[company]
    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[top_emp.index(company)])
plt.xlabel("Year")
plt.ylabel("Average Salary offered (USD)")
plt.legend()
plt.title('Average Salary of Top 10 Applicants')
plt.show()


# In[115]:


data.CASE_STATUS.value_counts()


# In[116]:


plt.figure(figsize=(10,7))
data.CASE_STATUS.value_counts().plot(kind='barh')
data.sort_values('CASE_STATUS')
plt.title("NUMBER OF APPLICATIONS")
plt.show()


# # Data Pre-processing

# In[117]:


data.isnull().any()


# In[118]:


data.isnull().sum()


# In[119]:


data.head(10)


# In[120]:


data = data.drop(['Unnamed: 0', 'EMPLOYER_NAME','JOB_TITLE','WORKSITE', 'lon','lat'], axis = 1)


# In[121]:


data


# In[122]:


data.mean(axis=1,skipna = True)


# In[123]:


data.isnull().any()


# In[124]:


data['CASE_STATUS'].fillna(data['CASE_STATUS'].mode().iloc[0],inplace=True)
data['SOC_NAME'].fillna(data['SOC_NAME'].mode().iloc[0],inplace=True)
data['FULL_TIME_POSITION'].fillna(data['FULL_TIME_POSITION'].mode().iloc[0],inplace=True)
data['YEAR'].fillna(data['YEAR'].mode().iloc[0],inplace=True)


# In[125]:


data['PREVAILING_WAGE'].fillna(data['PREVAILING_WAGE'].median(),inplace=True)


# In[126]:


data.isnull().any()


# In[127]:


data.isnull().sum()


# In[128]:


print(data['CASE_STATUS'].unique())
print(data['YEAR'].unique())
print(data['FULL_TIME_POSITION'].unique())


# # Label Encoding

# In[129]:


data['CASE_STATUS'] = data['CASE_STATUS'].map({'CERTIFIED': 0,'CERTIFIED-WITHDRAWN': 1,'DENIED': 2,'WITHDRAWN': 3,'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED': 4,'REJECTED': 5,'INVALIDATED': 6})


# In[130]:


data['FULL_TIME_POSITION'] = data['FULL_TIME_POSITION'].map({'N': 0, 'Y': 1})


# In[131]:


data.head()


# In[132]:


# For SOC_NAME Labelencoding is in different format.


# In[133]:


import sys
data['SOC_NAME1'] = 'others'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('chief','management')] = 'manager'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('computer','software')] = 'it'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('mechanical')] = 'mechanical'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('database')] = 'database'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('sales','market')] = 'scm'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('financial')] = 'finance'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('public','fundraising')] = 'pr'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('education','law')] = 'administrative'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('auditors','compliance')] = 'audit'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('distribution','logistics')] = 'scm'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('recruiters','human')] = 'hr'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('agricultural','farm')] = 'agri'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('construction','architectural')] = 'estate'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('forencsic','health')] = 'medical'
data['SOC_NAME1'][data['SOC_NAME'].str.contains('teachers')] = 'education'


# In[134]:


data.head(25)


# In[135]:


data['SOC_NAME1'].unique()


# In[136]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data['SOC_NAME1'])
data['SOC_N'] = le.transform(data['SOC_NAME1'])


# In[137]:


group =data.groupby('SOC_NAME1')
data2 = group.apply(lambda x: x['SOC_N'].unique())


# In[138]:


data2


# In[139]:


data.loc[data['CASE_STATUS'] == 2]


# In[140]:


data


# In[147]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("oh",OneHotEncoder(),[4])],remainder="passthrough")
x = ct.fit_transform(data)
x


# In[148]:


x = x[:,1:]
x


# In[149]:


ct2 = ColumnTransformer([("oh",OneHotEncoder(),[5])],remainder="passthrough")
x = ct2.fit_transform(data)
x


# In[150]:


x = x[:,1:]
x


# In[151]:


import joblib
joblib.dump(ct,'tranform')


# In[152]:


import joblib
joblib.dump(ct2,'tranform2')


# In[154]:


data


# In[156]:


data = data.drop(['SOC_NAME','SOC_NAME1'],axis=1)


# In[157]:


data


# # Correlation Table and Heatmap

# In[46]:


data.corr()


# In[47]:


sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})


# # Removal of outliers

# In[48]:


from scipy import stats
z = np.abs(stats.zscore(data))
z


# In[49]:


z.shape


# In[50]:


threshold=3
np.where(z>threshold)


# In[51]:


data_no_outliers=data[(z<=3).all(axis=1)]
data_no_outliers


# In[52]:


data_no_outliers.shape


# # Splitting the dataset into x and y

# In[53]:


selcols=["FULL_TIME_POSITION","PREVAILING_WAGE","YEAR","SOC_N"]
x=pd.DataFrame(data_no_outliers,columns=selcols)
y=pd.DataFrame(data_no_outliers,columns=['CASE_STATUS'])


# In[54]:


x


# In[55]:


y


# # Splitting data into train and test datasets

# In[56]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 42)

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Building

# LOGISTIC REGRESSION

# In[57]:


from sklearn.linear_model import LogisticRegression
lr =  LogisticRegression()
lr.fit(x_train,y_train.values.ravel())


# In[58]:


y_pred_lr = lr.predict(x_test)
print(y_pred_lr)


# In[59]:


print(y_test)


# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_lr,zero_division = 0))


# In[61]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_lr))


# Decision Tree 

# In[62]:


from sklearn import tree
model_dt = tree.DecisionTreeClassifier()
model_dt.fit(x_train, y_train)


# In[63]:


y_pred_dt = model_dt.predict(x_test)
print(y_pred_dt)


# In[64]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_dt,zero_division = 0))


# In[65]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_dt)


# Random Forest Classification

# In[66]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train.values.ravel())


# In[67]:


y_pred_rf =rf.predict(x_test)
print(y_pred_rf)


# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))


# In[69]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_rf)


# K-Nearest Neighbors

# In[70]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train.values.ravel())


# In[71]:


y_pred_KNN = KNN.predict(x_test)
print(y_pred_KNN)


# In[72]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_KNN,zero_division = 0))


# In[73]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_KNN)


# # comparing accuracies of four models

# In[74]:


#logistic regression
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_lr))


# In[75]:


#Decision Tree
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_dt))


# In[76]:


#Random Forest
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_rf))


# In[77]:


#KNN
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_KNN))


# # Conclusion

# At the end of our modelling step, we build four models: Logistic Regression, Decision Tree,Random Forest and KNN.We noticed that the Logistic Regression gave a better result when compared to others. Hence, it will be selected to deploy the web application using IBM cloud.

# # Saving The Model

# In[78]:


import pickle
pickle.dump(lr,open("Visa.pkl",'wb'))

