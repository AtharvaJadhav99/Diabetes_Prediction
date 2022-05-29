import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

pickle_in= open('diabetes.pkl','rb')
classifier= pickle.load(pickle_in)

st.header(""" Welcome to the Diabetes Prediction App """)
st.subheader(" THIS APP HELPS IN PREDICTING IF A PERSON HAS DIABETES OR NOT USING MACHINE LEARNING TECHNIQUES")

image = Image.open('diabetes.png')
st.image(image, use_column_width='auto')

st.sidebar.header('Inputs Selected by the User')

name = st.sidebar.text_input("Name:")
Pregnancies= st.sidebar.slider('Pregnancies', 0,10 ,3)
Glucose= st.sidebar.slider('Glucose', 0,199,120)
BloodPressure= st.sidebar.slider('Blood Pressure (mm Hg)', 0,122 ,69)
Insulin= st.sidebar.slider('Insulin level', 0,850 ,80)
BMI= st.sidebar.slider('BMI', 15,40 ,30)
Age= st.sidebar.slider('Age', 18,90 ,30)

st.subheader('Parameters Input by User')
data={
	'Pregnancies':Pregnancies,
	'Glucose':Glucose,
	'Blood Pressure (mm Hg)':BloodPressure,
      'Insulin level':Insulin,
	'BMI':BMI,
	'Age':Age}
features = pd.DataFrame(data, index=[0])
st.write(features)

     
st.subheader("""Prediction """)
prediction = classifier.predict([[Pregnancies, Glucose, BloodPressure,Insulin,BMI,Age]])
if(prediction == 0):
	st.write('""" Congratulation,',name,'you do not show any signs of having diabetes """)
else:
      st.write( """ Sorry,""",name,"""It seems that there is a high chance of you being Diabetic. It is highly advisible to contact your doctor to know more about the disease and the further course of treatment """)
      st.write(" To know more about the symptoms and causes of diabetes refer the following link: https://www.healthline.com/health/diabetes#symptoms")
      st.write(" To Book a Doctor's Appointment refer the following link: https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22diabetes%20doctor%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22common_name%22%7D%5D&city=Pune")
