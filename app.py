import joblib
import streamlit as st
#Streamlit
model_lr=joblib.load('LogisticRegression.pkl')
model_dtc=joblib.load('DecisionTree.pkl')
model_rfc=joblib.load('RandomForest.pkl')




Pclass=st.number_input("Enter passenger class")
Sex_female=st.number_input("Enter sex")
Age=st.number_input("Enter age")
Sibsp=st.number_input("Enter the number of siblings and spouses")
Parch=st.number_input("Enter the number of parents and children")
Fare=st.number_input("Enter fare")

ypred_lr=model_lr.predict([[Pclass,Age,Sibsp,Parch,Fare,Sex_female]])
ypred_dtc=model_dtc.predict([[Pclass,Age,Sibsp,Parch,Fare,Sex_female]])
ypred_rfc=model_rfc.predict([[Pclass,Age,Sibsp,Parch,Fare,Sex_female]])
print("Prediction is",ypred_lr)
print("Prediction is",ypred_dtc)
print("Prediction is",ypred_rfc)

dropdown=st.selectbox("Enter the supervised learning model",["Logistic Regression","RandomForest","DecisionTree"])
if st.button("Did this passenger survive?"):
    if(dropdown=='Logistic Regression'):
        if(ypred_lr==1):
            st.write("The passenger survived")
        else:
            st.write("The passenger did not survive")
    elif(dropdown=='DecisionTree'):
        if(ypred_dtc==1):
            st.write("The passenger survived")
        else:
            st.write("The passenger did not survive")
    elif(dropdown=='RandomForest'):
        if(ypred_rfc==1):
            st.write("The passenger survived")
        else:
            st.write("The passenger did not survive")
            