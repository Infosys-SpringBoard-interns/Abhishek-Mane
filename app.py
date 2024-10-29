import streamlit as st
import pandas as pd
import pickle
import numpy as np

numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI']

with open("tabnet_obesity_model.pkl", "rb") as model_file:
    tabnet_model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

obesity_labels = {0: 'Underweight', 1: 'Normal Weight', 2: 'Overweight', 3: 'Obese'}

st.title("Obesity Level Prediction App")
st.write("This app predicts your obesity level based on health factors.")

age = st.number_input("Enter your age", min_value=1, max_value=100, step=1)
gender = st.selectbox("Select your gender", ["Male", "Female"])
height = st.number_input("Enter your height (in cm)", min_value=100.0, max_value=250.0, step=0.1)
weight = st.number_input("Enter your weight (in kg)", min_value=30.0, max_value=300.0, step=0.1)
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
diet_type = st.selectbox("Diet Type", ["Balanced", "High Protein", "Low Carb", "Vegetarian", "Vegan"])
smoking_habits = st.selectbox("Smoking Habits", ["Non-Smoker", "Former Smoker", "Current Smoker"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
family_history = st.selectbox("Family History of Obesity", ["No", "Yes"])
blood_pressure = st.selectbox("Blood Pressure Status", ["Normal", "Prehypertension", "Hypertension Stage 1", "Hypertension Stage 2"])
cholesterol_levels = st.selectbox("Cholesterol Levels", ["Normal", "Borderline High", "High"])
education_level = st.selectbox("Education Level", ["No formal education", "High School", "College", "Postgraduate"])
income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
geographical_region = st.selectbox("Geographical Region", ["Urban", "Suburban", "Rural"])

bmi = weight / ((height / 100) ** 2)

gender_numeric = 1 if gender == "Male" else 0
physical_activity_numeric = {"Low": 0, "Moderate": 1, "High": 2}[physical_activity]
diet_type_numeric = {"Balanced": 0, "High Protein": 1, "Low Carb": 2, "Vegetarian": 3, "Vegan": 4}[diet_type]
smoking_habits_numeric = {"Non-Smoker": 0, "Former Smoker": 1, "Current Smoker": 2}[smoking_habits]
alcohol_consumption_numeric = {"None": 0, "Moderate": 1, "Heavy": 2}[alcohol_consumption]
family_history_numeric = 1 if family_history == "Yes" else 0
blood_pressure_numeric = {"Normal": 0, "Prehypertension": 1, "Hypertension Stage 1": 2, "Hypertension Stage 2": 3}[blood_pressure]
cholesterol_levels_numeric = {"Normal": 0, "Borderline High": 1, "High": 2}[cholesterol_levels]
education_level_numeric = {"No formal education": 0, "High School": 1, "College": 2, "Postgraduate": 3}[education_level]
income_level_numeric = {"Low": 0, "Middle": 1, "High": 2}[income_level]
geographical_region_numeric = {"Urban": 0, "Suburban": 1, "Rural": 2}[geographical_region]

input_data = pd.DataFrame([[age, gender_numeric, height, weight, bmi, 
                            physical_activity_numeric, diet_type_numeric, smoking_habits_numeric,
                            alcohol_consumption_numeric, family_history_numeric, 
                            blood_pressure_numeric, cholesterol_levels_numeric,
                            education_level_numeric, income_level_numeric, geographical_region_numeric]], 
                          columns=['Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI',
                                   'Physical_Activity_Level', 'Diet_Type', 'Smoking_Habits',
                                   'Alcohol_Consumption', 'Family_History_Obesity',
                                   'Blood_Pressure', 'Cholesterol_Levels',
                                   'Education_Level', 'Income_Level', 'Geographical_Region'])

input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

if st.button("Predict Obesity Level"):
    prediction = tabnet_model.predict(input_data.to_numpy())[0]
    prediction_label = obesity_labels.get(prediction, "Unknown Class")
    st.subheader("Predicted Obesity Level:")
    st.write(f"**{prediction_label}**")
