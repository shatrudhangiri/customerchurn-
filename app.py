import streamlit as st
import pandas as pd
import numpy as np
import pickle

## Load Model

model = pickle.load(open('churn_model1.pkl', 'rb'))

## Pgae Title

st.title("Customer Churn Prediction App")
st.write("Enter your customer information below")

## Now providing UI for user input

gender = st.selectbox("gender",['Male','Female'])
SeniorCitizen= st.selectbox("SeniroCitizen",[0,1])
tenure = st.slider('tenure', 1,72)
monthly_charges=st.number_input('MonthlyCharges')
total_charges= st.number_input('TotalCharges')
contract = st.selectbox("Contract Type", ["month-to-month",'One year','Two year'])

internet_service = st.selectbox("Internet Service",['DSL','Fibre optic','No'])

## Manual Encoding

gender = 1 if gender =="Male" else 0

contract_map= {
    "month-to-month":0,
    "One year":1,
    "Two year":2
    }

internet_map={
    'DSL':0,
    'Fiber optic':1,
    'No':2
}

contract = contract_map[contract]
internet_service= internet_map[internet_service]

##Predict Button

if st.button("Predict Churn"):
    features = np.array([[
        gender,
        SeniorCitizen,
        tenure,
        monthly_charges,
        total_charges,
        contract,
        internet_service
    ]])
    prediction = model.predict(features)  
    probability = model.predict_proba(features)  

    st.subheader("Prediction Result")

    if prediction[0]==1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")

    st.write(
        "Churn Probability:",
        round(probability[0][1]*100, 2),
    '%'
    )
