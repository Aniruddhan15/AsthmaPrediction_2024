# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:55:58 2024

@author: aniru
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open("C:/AsthmaPrediction_2024/AsthmaPrediction_2024/finalized_model (3).sav", 'rb'))

# creating a function for Prediction
def asthma_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not likely to have asthma'
    else:
        return 'The person is likely to have asthma'

def main():
    # giving a title
    st.title('Asthma Prediction Web App')

    # getting the input data from the user
    columns = ['PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
               'PollutionExposure', 'PollenExposure', 'DustExposure', 'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 'HayFever', 
               'GastroesophagealReflux', 'LungFunctionFEV1', 'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath', 'ChestTightness', 'Coughing', 
               'NighttimeSymptoms', 'ExerciseInduced', 'DoctorInCharge']

    # Create a dictionary to store the user inputs
    user_input = {}
    
    for column in columns:
        user_input[column] = st.text_input(f'{column}')

    # Convert user input dictionary to list
    input_data = [user_input[column] for column in columns]

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Asthma Test Result'):
        diagnosis = asthma_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
