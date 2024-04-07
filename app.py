import numpy as np
import pandas as pd
import streamlit as st
import pickle
from src.utils import load_object
from src.exception import CustomException
import sys
import os


def predict_crop(N,P,K,temperature,humidity,ph,rainfall):
    try:
        model_path = os.path.join('artifacts','model.pkl')
        preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        print("Loading Model....")
        data_scaled = preprocessor.transform([[N,P,K,temperature,humidity,ph,rainfall]])
        prediction = model.predict(data_scaled)
        return prediction


    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        html_temp = """
        <div style="background-color:brown;padding:10px">
        <h2 style="color:white;text-align:center;">Crop Recommendation 🌱</h2>
        </div>
        <br>
        """

        st.markdown(html_temp,unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("Nitrogen (range 0-150)", min_value=0.00,max_value=150.00,value="min")
            P = st.number_input("Phosphorus (range 5-150)", min_value=5.00,max_value=150.00,value="min")
            K = st.number_input("Potassium (range 5-210)", min_value=5.00,max_value=210.00,value="min")
            rainfall = st.number_input("Rainfall (range 10-300)", min_value=0.00,max_value=300.00,value="min")


        with col2:
            temperature = st.number_input("Temperature (range 8-50)", min_value=8.00,max_value=50.00,value="min")
            humidity = st.number_input("Humidity (range 14-100)", min_value=14.00,max_value=100.00,value="min")
            ph = st.number_input("pH (range 3-10)", min_value=3.00,max_value=10.00,value="min")

        result=""
        if st.button("Predict"):
            result=predict_crop(N,P,K,temperature,humidity,ph,rainfall)
            if result:
                st.success('The output is {}'.format(result[0]))
            else:
                st.success('Cannot Determine')

    except Exception as e:
        raise CustomException(e, sys)

if __name__=='__main__':
    main()