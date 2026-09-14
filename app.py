import tensorflow as tf
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import pandas as pd
#Load the model
model = tf.keras.models.load_model('churn_model.h5')

#Load the encoder and scaler
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
## streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox("Select Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Select Gender", ["Male", "Female"])
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Enter Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Enter Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])   
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)

#prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
}

#one-hot encode the categorical features
geography_encoded = onehot_encoder.transform([[input_data['Geography'][0]]]).toarray()
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

#label encode the gender feature
input_data['Gender'] = label_encoder.transform([input_data['Gender']])

# Convert input data to DataFrame   
input_df = pd.DataFrame(input_data)
input_df = pd.concat([input_df.drop('Geography', axis=1), geography_df], axis=1)   

#scale the input data
input_data_scaled = scaler.transform(input_df)

#Predict churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}.")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_probability:.2f}.") 
