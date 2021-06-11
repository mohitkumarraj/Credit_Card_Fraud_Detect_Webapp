# Importing Required Libraries and Importing our Dataset into notebook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

os.chdir('PycharmProjects\Creditcard_Fraud')


data=pd.read_csv('creditcard.csv',na_values=['??','???'])

data.head(5)
data.tail()
fraud_trans = data.loc[data['Class']==1]
normal_trans=data[data['Class']==0]

data.keys()

data['V1'].describe()
# num=np.random.uniform(-4.83255893623954,9.38255843282114)

#to get rid of the exponential values 
# pd.set_option('display.float_format', lambda x:'%5'%x)

data['Class'].value_counts()

print('Number of Fraud transaction=',len(fraud_trans),"\nNumber of Normal transaction= ",len(normal_trans))

# plt.scatter(x='Time',y='Amount',data=data,c='Red')

sns.countplot(data['Class'])
corre=data.corr()

# mask = np.triu(np.ones_like(corre, dtype=float))

corre

sns.set(rc={'figure.figsize':(12,12)})

sns.heatmap(corre)


sns.lmplot(x='Time',y='Amount',data=data,hue='Class')

sns.lmplot(x='Time',y='Amount',data=fraud_trans,hue='Class')

fraud_trans['Amount'].value_counts().sort_index()

fraud_trans['Amount'].describe()


data.isna().sum() #no any null value 

x=data.drop('Class',axis=1,inplace=False)
y=data['Class']






# ============================================Random Forest Classifier=================================
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# 
# 
# 
# X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
# 
# rf_model=RandomForestClassifier()
# 
# rf_model.fit(X_train,y_train)
# 
# r_pred=rf_model.predict(X_test)
# 
# 
# print('Accuracy Score= ',accuracy_score(y_test,r_pred)*100,'%')
# rf_accuracy=(accuracy_score(y_test, r_pred)*100)
# print('\nConfusion matrix=\n',confusion_matrix(y_test,r_pred))
# print('\nclassification Report \n',classification_report(y_test, r_pred))
# pickle.dump(rf_model, open('r_model.pkl','wb'))
# 
# =============================================================================






















# ==========================================================================
# =====================   start Linear Model       ============================
# ==========================================================================

from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
print('X_train=',X_train.shape)
print('X_test=',X_test.shape)

print('y_train',y_train.shape)
print('y_test',y_test.shape)

#standardization scaling data
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)

classifier=linear_model.LogisticRegression()

classifier.fit(X_train, y_train)

pred_value =classifier.coef_

y_pred= np.array(classifier.predict(X_test))

y=np.array(y_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,y_pred))
ln_accuracy=accuracy_score(y_test, y_pred)*100
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(classifier, open('lg.pkl','wb'))

# ==========================================================================
# =====================   end Linear Model      ============================
# ==========================================================================







# ==========================================================================
# =====================    start Decision Tree classifier  ============================
# ==========================================================================




# =============================================================================
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#  
# X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
#  
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#  
# dclassifier=DecisionTreeClassifier(criterion='gini',random_state=(100),max_depth=(10),min_samples_leaf=5)
#  
# dclassifier.fit(X_train, y_train)
#  
# d_pred=dclassifier.predict(X_test)
#  
# print(accuracy_score(y_test,d_pred))
# d_accuracy=(accuracy_score(y_test, d_pred)*100)
# print(confusion_matrix(y_test,d_pred))
# print(classification_report(y_test, d_pred))
# pickle.dump(dclassifier, open('d_model.pkl','wb'))
# =============================================================================





ln_log=np.log(ln_accuracy)
dt_log=np.log(d_accuracy)

counts=[ln_accuracy,d_accuracy,rf_accuracy]
models=['Ln_regression','Decision_Tree Classifier','RandomForestClassification']
index=np.arange(len(models))
plt.ylabel("accuracy->")
plt.title("Comparison of accuracy of models")
plt.xlabel(models)
plt.bar(index,counts,color=['cyan','blue','Green'])





######################by taking log of accuracys============
ln_log=np.log(ln_accuracy)
dt_log=np.log(d_accuracy)

counts=[ln_log,dt_log]
models=['Ln_regression','Decision_Tree Classifier','RandomForestClassification']
index=np.arange(len(models))
plt.ylabel("accuracy->")
plt.title("Comparison of accuracy of models")
plt.xlabel(models)
plt.bar(index,counts,color=['red','blue'])


####accuracy of  both models are too close ================




# =============================================================================
# features=[45.55]
# 
# pc1 = [-2.31222654232630, 1.95199201064158, -1.60985073229769, 3.99790558754680, -0.52218786466776,
#          -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915,
#            3.20203320709635, -2.89990738849473, -0.59522188132461, -4.28925378244217, 0.38972412027449,
#             -1.14074717980657, -2.83005567450437, -0.01682246818083, 0.41695570503791, 0.12691055906147,
#             0.51723237086176, -0.03504936860530, -0.46521107618239, 0.32019819851453, 0.04451916747317, 0.17783979828440,
#             0.26114500256768, -0.14327587469892]
# 
# 
# features.append(pc1)
# features.append(999)
# 
# 
# 
# 
# print(fraud_trans)
# 
# first=fraud_trans.iloc[0,1:29]
# 
# print(fraud_trans.iloc[0:1,1:29])
# 
# 
# 
# 
# 
# f1=[fe for i in fraud_trans.iloc[0:1,1:29] fe.append(i)]
# 
# 
# 
# ###########
# features=[45]
# 
# 
# fraud_trans.head(10)
# 
# 
# r1=fraud_trans.iloc[2,1:29].tolist()
# 
# features=features+r1
# 
# 
# r1.append(54646)
# 
# =============================================================================

















# import random

# random_num = random.choice(data['V2'])

# col=[data.columns]

# col=col.tolist()

# pca=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']




# features=[]

# features.append(5555565651)
    
# # for i in pca:
# #     features.append(random.choice(data[i]))
    

# for i in pca:
#     features.append(random.choice(data[i]))


# features.append(567555.6)

# features_arra=np.array(features).reshape(1,-1)
# features=scaling.fit_transform(features_arra)

# print(features)

# # features.append(6546464654)


# tid = random.randint(345548616575,785641364312)

# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(
