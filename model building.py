# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:54:33 2023

@author: Admin
"""

#Import Libraries and Read the data
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import pipeline

df  =  pd.read_csv("G:/project2/solardata_preprocessed.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
# categorical to numeric
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['FaultType'] = labelencoder.fit_transform(df['FaultType']) 

X = df.drop('FaultType', axis = 1)
Y = df['FaultType']

# Normalization
from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(X)
print(scaled)

input_data = pd.DataFrame(scaled)
data = pd.concat([input_data, Y], axis=1)

#Split the Data into Training and Testing sets with test size as #30%
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, shuffle=True)

from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest
#No we Will select the top 5 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=5)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

from sklearn.svm import SVC
svc = SVC()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

model_pipeline = []
model_pipeline.append(reg)
model_pipeline.append(dtc)
model_pipeline.append(rfc)
model_pipeline.append(svc)
model_pipeline.append(knn)
model_pipeline.append(gnb)

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import ensemble

model_list = ['Logistic regression', 'Decision Tree classifier', 'Random Forest classifier', 'SVC', 'KNeighbors Classifier', 'Gaussian classifier']
acc_list = []
auc_list = []
cm_list = []


for model in model_pipeline:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_list.append(metrics.accuracy_score(y_test, y_pred))
    fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    auc_list.append(round(metrics.auc(fpr, tpr), 2))
    cm_list.append(confusion_matrix(y_test, y_pred))
    
result_df = pd.DataFrame({'Model':model_list, 'Accuracy':acc_list, 'AUC':auc_list})
print(result_df)

# From abve result, Random Forest classifier is the best model as it is having high accuracy.



model = RandomForestClassifier(max_depth=3, random_state=22)
model.fit(X_train, y_train)
print('Training Accuracy : ', metrics.accuracy_score(y_train, model.predict(X_train))*100)
print('Validation Accuracy : ', metrics.accuracy_score(y_test, model.predict(X_test))*100)


model = RandomForestClassifier(max_depth=30, random_state=22)
model.fit(X_train, y_train)
print('Training Accuracy : ', metrics.accuracy_score(y_train, model.predict(X_train))*100)
print('Validation Accuracy : ', metrics.accuracy_score(y_test, model.predict(X_test))*100)


rfmodel = RandomForestClassifier(max_depth=3, n_estimators=22, min_samples_split=4, max_leaf_nodes=5, random_state=22)
rfmodel.fit(X_train, y_train)
print('Training Accuracy : ', metrics.accuracy_score(y_train, model.predict(X_train))*100)
print('Validation Accuracy : ', metrics.accuracy_score(y_test, model.predict(X_test))*100)

import pickle
import joblib
filename = 'finalized_model.pkl'
joblib.dump(rfmodel, filename)
mdl = joblib.load('finalized_model.pkl')
print(mdl)
with open('finalized_model.pkl', 'wb') as model_file: 
    pickle.dump(rfmodel, model_file)
model = joblib.load('finalized_model.pkl')
