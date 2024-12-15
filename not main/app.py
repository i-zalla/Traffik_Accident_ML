import streamlit as st
import pandas as pd
import requests
import pickle
from datetime import datetime, timedelta

from functions import calculate_probability, process_forecast, get_location_info, get_weather_forecast

st.set_page_config(page_title="Traffic Accident Prediction", page_icon="🚗", layout="centered")

with open('frequent_itemsets.pkl', 'rb') as f: 
    frequent_itemsets = pickle.load(f) 

with open('apriori_rules.pkl', 'rb') as f: 
    apriori_rules = pickle.load(f) 

def get_location_info(state_name, file_path="result_roud4_scaler2.csv"):
    try:
        df = pd.read_csv(file_path, sep=';')
        required_columns = {'state_name', 'Location_Cluster', 'Longitude', 'Latitude'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            return f"Les colonnes manquantes dans le fichier : {', '.join(missing_columns)}."
        result = df[df['state_name'] == state_name]
        if not result.empty:
            location_info = {
                'Location_Cluster': result.iloc[0]['Location_Cluster'],
                'longitude': result.iloc[0]['Longitude'],
                'latitude': result.iloc[0]['Latitude']
            }
            return location_info
        else:
            return f"Le state_name '{state_name}' n'a pas été trouvé dans le fichier."
    except FileNotFoundError:
        return f"Le fichier '{file_path}' est introuvable."
    except Exception as e:
        return f"Une erreur s'est produite : {e}"

def get_accident_probabilities(city_name):
    try:
        st.write(f"Received city name: {city_name}")
        location_info = get_location_info(city_name)
        st.write(f"Location Info: {location_info}")

        if isinstance(location_info, dict):
            lat = location_info['latitude']
            lon = location_info['longitude']
            location_cluster = location_info['Location_Cluster']
            
            st.write(f"Latitude: {lat}, Longitude: {lon}, Location Cluster: {location_cluster}")
            
            weather_forecast = get_weather_forecast(lat, lon)
            st.write(f"Weather Forecast: {weather_forecast}")

            if not weather_forecast or isinstance(weather_forecast, str):
                return "Les prévisions météorologiques ne sont pas disponibles pour cette localisation."

            processed_forecast = process_forecast(weather_forecast, location_cluster)
            st.write(f"Processed Forecast: {processed_forecast}")

            processed_forecast['Probability'] = processed_forecast.apply(
                lambda row: calculate_probability(row, apriori_rules), axis=1
            )
            st.write(f"Probabilities and applied rules: {processed_forecast[['Date', 'Probability']]}")

            return processed_forecast[['Date', 'Probability']]
        else:
            return location_info  
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit UI
st.title("Accident Probability Predictor")

city_name = st.text_input("Enter the city name:")

if city_name:
    st.write(f"Input city name: {city_name}")
    accident_probabilities = get_accident_probabilities(city_name)
    
    if isinstance(accident_probabilities, pd.DataFrame):
        st.write(f"Accident probabilities for the city {city_name}:")
        st.dataframe(accident_probabilities.style.format({"Probability": "{:.8f}"}))
    else:
        st.write(accident_probabilities)
else:
    st.write("Please enter a city name.")
