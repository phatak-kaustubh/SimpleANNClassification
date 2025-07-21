from calendar import c
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# load trained model
model = tf.keras.models.load_model('model.h5')

# load all the encode and scalar
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoder_geo.pkl','rb')as file:
    onehot_encoder_geo=pickle.load(file)
with open('scalar.pkl','rb')as file:
    scalar=pickle.load(file)
    
st.title('Customer Churn Predictor')

#user input in UI
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,100)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance',min_value=0.0)
credit_score = st.number_input('Credit Score',min_value=0,max_value=1000)
num_of_products = st.slider('Number of Products',1,5)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active = st.selectbox('Is Active',[0,1])
estimated_salary = st.number_input('Estimated Salary',min_value=0.0)


#prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active],
    'EstimatedSalary':[estimated_salary]
})

#onehot encode for Geography
geo_encode = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encode = pd.DataFrame(geo_encode, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine the onehot encoded data with the input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encode], axis=1)

#Scale the input data
input_data = scalar.transform(input_data)

#predict the output
prediction = model.predict(input_data)
#display the output
prediction_probability = prediction[0][0]
st.write('The customer is likely to churn with a probability of', prediction_probability)

if prediction >0.5:
    st.write('The customer is not likely to churn')
else:
    st.write('The customer is likely to churn')