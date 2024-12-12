import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from functions import get_location_info, get_weather_forecast, convert_to_binary, process_forecast, calculate_probability

st.set_page_config(page_title="Traffic Accident Prediction", page_icon="🚗", layout="centered")

#adding API key for forecast weather 
api_key = "1604376839c243cfa8f223439240712"  

#importing the rules
with open('frequent_itemsets.pkl', 'rb') as f: 
    frequent_itemsets = pickle.load(f) 

with open('apriori_rules.pkl', 'rb') as f: 
    apriori_rules = pickle.load(f) 


def get_accident_probabilities(city_name):
    location_info = get_location_info(city_name)
    
    if isinstance(location_info, dict):
        lat = location_info['latitude']
        lon = location_info['longitude']
        location_cluster = location_info['Location_Cluster']
        
        weather_forecast = get_weather_forecast(lat, lon, api_key)

        processed_forecast = process_forecast(weather_forecast)
      
        processed_forecast['Probability'] = processed_forecast.apply(
            lambda row: calculate_probability(row, apriori_rules), axis=1
        )
        
        return processed_forecast[['Date', 'Probability']]
    else:
        return location_info  
#creating ui
st.title("Accident Probability Predictor")

city_name = st.text_input("Enter the city name:")

if city_name:
    accident_probabilities = get_accident_probabilities(city_name)
    
    if isinstance(accident_probabilities, pd.DataFrame):
        st.write(f"Accident probabilities for the city {city_name}:")
        st.dataframe(accident_probabilities.style.format({"Probability": "{:.8f}"}))
    else:
        st.write(accident_probabilities)


