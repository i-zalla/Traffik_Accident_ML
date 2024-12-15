import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

def get_weather_forecast(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_min,temperature_2m_max,precipitation_sum,snowfall_sum&timezone=auto"
    
    # Debugging info
    print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"Request URL: {url}")
    
    response = requests.get(url)
    
    print(f"Response Status Code: {response.status_code}")
    try:
        response_json = response.json()
        print(f"Response Content: {response_json}")
    except Exception as e:
        print(f"Failed to parse response JSON: {e}")
        return f"Erreur HTTP: {response.status_code} - {response.text}"
    
    if response.status_code == 200:
        if 'daily' in response_json:
            forecast = []
            daily_data = response_json['daily']
            for i in range(len(daily_data['time'])):
                date = daily_data['time'][i]
                if daily_data['snowfall_sum'][i] > 0:
                    description = "Snowy"
                elif daily_data['precipitation_sum'][i] > 0:
                    description = "Rainy"
                else:
                    description = "Clear"
                forecast.append({"Date": date, "Météo": description})
            return forecast
        else:
            return "Les données météorologiques ne sont pas disponibles."
    else:
        return f"Erreur HTTP: {response.status_code} - {response.text}"




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


def convert_to_binary(weather_description):
    rain_keywords = ['rain', 'drizzle', 'showers', 'thunderstorm', 'sleet', 'light rain', 'moderate rain', 'heavy rain', 'freezing rain', 'rain shower', 'patchy rain']
    snow_keywords = ['snow', 'blizzard', 'flurries', 'sleet', 'light snow', 'moderate snow', 'heavy snow', 'snow shower', 'patchy snow', 'freezing snow']
    Weather_Conditions_Rain = 1 if any(keyword in weather_description.lower() for keyword in rain_keywords) else 0
    Weather_Conditions_Snow = 1 if any(keyword in weather_description.lower() for keyword in snow_keywords) else 0
    return Weather_Conditions_Rain, Weather_Conditions_Snow


def process_forecast(forecast, location_cluster):
    df = pd.DataFrame(forecast)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df[['Weather_Conditions_Rain', 'Weather_Conditions_Snow']] = df['Météo'].apply(lambda x: pd.Series(convert_to_binary(x)))
    
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    weekday_dummies = pd.get_dummies(df['Weekday'], prefix='Weekday')

    # Ensure all expected columns are present
    for col in [f'Month_{i}' for i in range(2, 13)]:
        if col not in month_dummies.columns:
            month_dummies[col] = 0

    for col in [f'Weekday_{i}' for i in range(1, 7)]:
        if col not in weekday_dummies.columns:
            weekday_dummies[col] = 0
    
    # Add Location Cluster
    df['Location_Cluster'] = location_cluster

    df = pd.concat([df, month_dummies, weekday_dummies], axis=1)
    df.drop(columns=['Month', 'Day', 'Weekday'], inplace=True)

    ordered_columns = (
        [f'Month_{i}' for i in range(2, 13)] +
        [f'Weekday_{i}' for i in range(1, 7)] +
        ['Is_Weekend', 'Weather_Conditions_Rain', 'Weather_Conditions_Snow', 'Location_Cluster']
    )
    df = df[['Date', 'Météo'] + ordered_columns]
    return df





def calculate_probability(row, rules):
    confidences = []
    used_rules = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        if all(antecedent in row.index and row[antecedent] == 1 for antecedent in antecedents):
            confidences.append(rule['confidence'])
            used_rules.append(rule['antecedents'])
    probability = sum(confidences) / len(confidences) if confidences else 0
    return probability

