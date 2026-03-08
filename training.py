#Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib


#Training 
df=pd.read_csv("Titanic-Dataset.csv")
y=df["Survived"]
x=df.drop(["Survived","PassengerId","Name","Ticket","Embarked","Cabin"],axis=1)
x['Age']=x['Age'].fillna(x['Age'].mean())
x=pd.get_dummies(x,columns=['Sex'],dtype=int)
print(x)
x=x.drop(['Sex_male'],axis=1)
print(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model_lr=LogisticRegression()
model_dtc=DecisionTreeClassifier()
model_rfc=RandomForestClassifier()
model_lr.fit(xtrain,ytrain)
model_dtc.fit(xtrain,ytrain)
model_rfc.fit(xtrain,ytrain)


#Joblib
joblib.dump(model_lr,'LogisticRegression.pkl')
joblib.dump(model_dtc,'DecisionTree.pkl')
joblib.dump(model_rfc,'RandomForest.pkl')