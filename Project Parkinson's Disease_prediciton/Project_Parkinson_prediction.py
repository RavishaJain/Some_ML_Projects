#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[43]:


os.getcwd() #To get the current working directory


# In[44]:


#os.chdir('Downloads\\')
df=pd.read_excel('Parkinsons.xlsx') #Load the data


# In[45]:


df.head() #5 enteries of the data


# In[46]:


df


# In[47]:


import ydata_profiling as pf
pf.ProfileReport(df) #Full profile of data


# In[48]:


df.describe() #Summary of data


# In[49]:


df.info #Info of the data


# In[50]:


len(df) #Length of data


# In[51]:


df.shape  #Shape of data 


# In[52]:


df.dtypes  #Datatype of each column


# In[53]:


df.isna().sum() #TO check null value in each column


# In[54]:


df.columns #NAme of Columns


# In[55]:


df['status'] 


# In[104]:


#Histogram of status wrt Frequencies
sns.histplot(df['status'],color='red')
plt.xlabel('Status')
plt.ylabel('Frequencies')
plt.show() 


# In[113]:


sns.barplot(x="status",y="NHR",data=df,palette='colorblind') #Barplot of Status wrt NHR
plt.show()


# In[117]:


sns.barplot(x='status',y='RPDE',data=df,palette='pastel')  #Barplot of status wrt RPDE
plt.show()


# In[59]:


#Distribution plot to check skewness
import warnings
warnings.filterwarnings('ignore')
rows=3
cols=7
fig, ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
col=df.columns
index=1
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]],ax=ax[i][j])
        index=index+1
        
plt.tight_layout()
plt.show()


# In[60]:


#Corelation matrix
corr = df.corr()
corr


# In[61]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10


# In[62]:


#Heat map
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='cubehelix',annot = True)
plt.show()


# In[63]:


df.drop(['name'],axis=1,inplace=True) #To remove name column as it is of no use


# In[64]:


X=df.drop(labels=['status'],axis=1) #Setting input
X.head()


# In[65]:


Y=df['status'] #Output which is categorical
Y.head()


# In[66]:


#Splitting training and testing data
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[67]:


display(train_x.shape) #Train_x shape
display(test_y.shape)  #Test_x shape
display(train_y.shape) #Train_y shape
display(test_y.shape)  #Test_y shape


# In[68]:


#Linear Regressio model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr=lr.fit(train_x, train_y)


# In[69]:


y_pred=lr.predict(train_x) #Prediction on training data


# In[70]:


from sklearn.metrics import accuracy_score, classification_report 
print("Model accuracy on train is: ", accuracy_score(train_y, y_pred)) #Accuracy score


# In[71]:


test_y_pred=lr.predict(test_x) #prediction on test data
print("Model accuracy on test is: ", accuracy_score(test_y, test_y_pred)) #Accuracy score


# In[72]:


from sklearn.metrics import confusion_matrix
print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred)) #To get Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred)) #To get Confusion matrix on test data
print('-'*50)
print('\nClassification Report Train is ') #Classification report on training data
print('%'*50)
print(classification_report (train_y, y_pred)) 
print('%'*50)
print('\nClassification Report Test is ') #Classification report on testing data
print('#'*50)
print(classification_report (test_y, test_y_pred))
print('#'*50)


# In[73]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(train_x,train_y)
y_pred1=rf.predict(train_x) #Prediction on training data

print("Model accuracy on train is: ", accuracy_score(train_y, y_pred1)) #Accuracy score


# In[74]:


test_y_pred1=rf.predict(test_x) #Prediction on test data


# In[75]:


print("Model accuracy on train is: ", accuracy_score(test_y,test_y_pred1)) #Accuracy score


# In[76]:


print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred1)) #To get Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred1))  #To get Confusion matrix on test data
print('-'*50)
print('\nClassification Report Train is ')
print('%'*50)
print(classification_report (train_y, y_pred1))  #Classification report on training data
print('%'*50)
print('\nClassification Report Test is ')
print('#'*50)
print(classification_report (test_y, test_y_pred1)) #Classification report on testing data
print('#'*50)


# In[77]:


print((test_y !=test_y_pred).sum(),'/',((test_y == test_y_pred1).sum()+(test_y != test_y_pred1).sum())) #to get wrong prediction


# In[78]:


from sklearn import metrics
print('KappaScore is: ', metrics.cohen_kappa_score(test_y,test_y_pred1)) #Kappa Score


# In[79]:


df1=pd.DataFrame(data=[test_y_pred1,test_y]) #Displaying the test and Predicted Values
(df1)


# In[80]:


df1.T #Transpose


# In[81]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt=dt.fit(train_x,train_y)
y_pred2=dt.predict(train_x) #Prediction on training data


# In[82]:


print("Model accuracy on train is: ", accuracy_score(train_y, y_pred2)) #Accuracy score


# In[83]:


test_y_pred2=dt.predict(test_x) #Prediction on testing data
print("Model accuracy on train is: ", accuracy_score(test_y, test_y_pred2)) #Accuracy score


# In[84]:


print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred2)) #To get Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred2)) #To get Confusion matrix on test data
print('-'*50)
print('\nClassification Report Train is ')
print('%'*50)
print(classification_report (train_y, y_pred2)) #Classification report on training data
print('%'*50)
print('\nClassification Report Test is ')
print('#'*50)
print(classification_report (test_y, test_y_pred2)) #Classification report on testing data
print('#'*50)


# In[85]:


print((test_y !=test_y_pred2).sum(),'/',((test_y == test_y_pred2).sum()+(test_y != test_y_pred2).sum()))#to get wrong prediction


# In[86]:


print('KappaScore is: ', metrics.cohen_kappa_score(test_y,test_y_pred2))  #Kappa Score


# In[87]:


#NAive Bayes 
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb=nb.fit(train_x,train_y)
y_pred3=nb.predict(train_x) #Prediction on training data
print("Model accuracy on train is: ", accuracy_score(train_y, y_pred3)) #Accuracy score


# In[88]:


test_y_pred3=nb.predict(test_x)  #Prediction on testing data
print("Model accuracy on train is: ", accuracy_score(test_y, test_y_pred3)) #Accuracy score


# In[89]:


print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred3)) #To get Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred3)) #To get Confusion matrix on test data
print('\nClassification Report Train is ')
print('%'*50)
print(classification_report (train_y, y_pred3)) #Classification report on training data
print('%'*50)
print('\nClassification Report Test is ')
print('#'*50)
print(classification_report (test_y, test_y_pred3)) #Classification report on test data
print('#'*50)


# In[90]:


print((test_y !=test_y_pred3).sum(),'/',((test_y == test_y_pred3).sum()+(test_y != test_y_pred3).sum()))#To get wrong prediction


# In[91]:


print('KappaScore is: ', metrics.cohen_kappa_score(test_y,test_y_pred3)) #Kappa Score


# In[92]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn=knn.fit(train_x,train_y)
y_pred4=knn.predict(train_x) #Prediction on training data
print("Model accuracy on train is: ", accuracy_score(train_y, y_pred4)) #Accuracy score


# In[93]:


test_y_pred4=knn.predict(test_x)
print("Model accuracy on train is: ", accuracy_score(test_y, test_y_pred4)) #Accuracy score


# In[94]:


print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred4)) #Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred4)) #Confusion matrix on test data
print('-'*50)
print('\nClassification Report Train is ')
print('%'*50)
print(classification_report (train_y, y_pred4)) #Classification Report on train data
print('%'*50)
print('\nClassification Report Test is ')
print('#'*50)
print(classification_report (test_y, test_y_pred4)) #Classification Report on test data
print('#'*50)


# In[95]:


print((test_y !=test_y_pred4).sum(),'/',((test_y == test_y_pred4).sum()+(test_y != test_y_pred4).sum()))#To get wrong prediction


# In[96]:


print('KappaScore is: ', metrics.cohen_kappa_score(test_y,test_y_pred4)) #Kappa Score


# In[97]:


#Support Vector Machine
from sklearn.svm import SVC
SVM = SVC(kernel='linear')
SVM=SVM.fit(train_x,train_y)
y_pred5=SVM.predict(train_x) #Prediction on training data
print("Model accuracy on train is: ", accuracy_score(train_y, y_pred5)) #Accuracy score
test_y_pred5=SVM.predict(test_x) #Prediction on test data
print("Model accuracy on train is: ", accuracy_score(test_y, test_y_pred5)) #Accuracy score


# In[98]:


print("confusion_matrix train is:\n ", confusion_matrix(train_y, y_pred5))  #Confusion matrix on train data
print('-'*50)
print("confusion_matrix test is:\n ", confusion_matrix(test_y, test_y_pred5)) #Confusion matrix on test data
print('-'*50)
print('\nClassification Report Train is ')
print('%'*50)
print(classification_report (train_y, y_pred5))  #Classification report on train data
print('%'*50)
print('\nClassification Report Test is ')
print('#'*50)
print(classification_report (test_y, test_y_pred5)) #Classification report on test data
print('#'*50)


# In[99]:


print("recall", metrics.recall_score(test_y, test_y_pred5))  #RecallScore


# In[100]:


print((test_y !=test_y_pred5).sum(),'/',((test_y == test_y_pred5).sum()+(test_y != test_y_pred5).sum()))#To get wrong prediction


# In[101]:


print('KappaScore is: ', metrics.cohen_kappa_score(test_y,test_y_pred5)) #Kappa Score


# In[102]:


import pickle  #To save into .pkl file
pickle.dump(SVM,open('deploy_SVM.pkl','wb'))
model=pickle.load(open('deploy_SVM.pkl','rb'))
model.predict (train_x)

