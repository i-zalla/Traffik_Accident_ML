import pandas as pd
import requests
import pickle
import streamlit as st
import datetime
from datetime import datetime, timedelta


df_ = pd.read_csv('association_ds.csv')

with open('./models/scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_location_info(state_name, file_path="result_roud4_scaler2.csv"):
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"Dataframe loaded from {file_path}")

        required_columns = {'state_name', 'Location_Cluster', 'Longitude', 'Latitude'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            print(f"Missing columns: {missing_columns}")
            return f"Les colonnes manquantes dans le fichier : {', '.join(missing_columns)}."

        result = df[df['state_name'] == state_name]
        print(f"Result for state_name '{state_name}': {result}")

        if not result.empty:
            location_info = {
                'Location_Cluster': result.iloc[0]['Location_Cluster'],
                'longitude': result.iloc[0]['Longitude'],
                'latitude': result.iloc[0]['Latitude']
            }
            return location_info
        else:
            print(f"State name '{state_name}' not found in file.")
            return f"Le state_name '{state_name}' n'a pas été trouvé dans le fichier."
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return f"Le fichier '{file_path}' est introuvable."
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Une erreur s'est produite : {e}"

def get_weather_forecast(lat, lon, api_key):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={lat},{lon}&days=7"
    print(f"API URL: {url}")

    response = requests.get(url)
    print(f"HTTP Response Status: {response.status_code}")

    if response.status_code == 200:
        weather_data = response.json()
        print(f"Weather data: {weather_data}")

        if 'forecast' in weather_data:
            forecast = []
            for day in weather_data['forecast']['forecastday']:
                date = day['date']
                description = day['day']['condition']['text']
                forecast.append({"Date": date, "Météo": description})
            return forecast
        else:
            print("Weather forecast not available in response.")
            return "Les données météorologiques ne sont pas disponibles."
    else:
        print(f"HTTP Error: {response.status_code} - {response.text}")
        return f"Erreur HTTP: {response.status_code} - {response.text}"

def convert_to_binary(weather_description):
    rain_keywords = [
        'rain', 'drizzle', 'showers', 'thunderstorm', 'sleet',
        'light rain', 'moderate rain', 'heavy rain',
        'freezing rain', 'rain shower', 'patchy rain'
    ]
    
    snow_keywords = [
        'snow', 'blizzard', 'flurries', 'sleet',
        'light snow', 'moderate snow', 'heavy snow',
        'snow shower', 'patchy snow', 'freezing snow'
    ]
    
    Weather_Conditions_Rain = 1 if any(keyword in weather_description.lower() for keyword in rain_keywords) else 0
    Weather_Conditions_Snow = 1 if any(keyword in weather_description.lower() for keyword in snow_keywords) else 0

    return Weather_Conditions_Rain, Weather_Conditions_Snow

def process_forecast(forecast, location_info):
    df = pd.DataFrame(forecast)
    print(f"Forecast dataframe: {df}")

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)

    df[['Weather_Conditions_Rain', 'Weather_Conditions_Snow']] = df['Météo'].apply(
        lambda x: pd.Series(convert_to_binary(x))
    )

    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    day_dummies = pd.get_dummies(df['Day'], prefix='Day')
    weekday_dummies = pd.get_dummies(df['Weekday'], prefix='Weekday')

    for col in [f'Month_{i}' for i in range(2, 13)]:
        if col not in month_dummies:
            month_dummies[col] = 0

    for col in [f'Day_{i}' for i in range(2, 32)]:
        if col not in day_dummies:
            day_dummies[col] = 0

    for col in [f'Weekday_{i}' for i in range(1, 7)]:
        if col not in weekday_dummies:
            weekday_dummies[col] = 0

    df = pd.concat([df, month_dummies, weekday_dummies], axis=1)
    df.drop(columns=['Month', 'Day', 'Weekday'], inplace=True)

    ordered_columns = (
        [f'Month_{i}' for i in range(2, 13)] +
        [f'Weekday_{i}' for i in range(1, 7)] +
        ['Is_Weekend', 'Weather_Conditions_Rain', 'Weather_Conditions_Snow']
    )
    df = df[['Date', 'Météo'] + ordered_columns]

    binary_columns = ordered_columns  
    df[binary_columns] = df[binary_columns].astype(int)

    df['Location_Cluster'] = location_info['Location_Cluster']
    df = pd.get_dummies(df, columns=['Location_Cluster'], prefix='Cluster')
    for col in df.columns:
        if col.startswith('Cluster_'):
            df[col] = df[col].astype(int)

    result_ = df.drop(columns=['Date', 'Météo'])
    print(f"Processed forecast dataframe: {result_}")
    return result_

def calculate_probability(row, rules):
    confidences = []
    used_rules = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        if all(antecedent in row and row[antecedent] == 1 for antecedent in antecedents):
            confidences.append(rule['confidence'])
            used_rules.append(rule['antecedents'])
    probability = sum(confidences) / len(confidences) if confidences else 0
    return probability


# ui part
st.title("Weather Probability Predictor")

state_name = st.text_input("Enter the state name")

if state_name:
    location_info = get_location_info(state_name)
    if isinstance(location_info, dict):
        print(f"Location info: {location_info}")

        scaled_input = [[location_info['longitude'], location_info['latitude'], 0, 0]]
        original_coords = scaler.inverse_transform(scaled_input)
        original_lon = original_coords[0][0]
        original_lat = original_coords[0][1]
        api_key = '92ab5ca77a9d445ca84124855241512'
        forecast = get_weather_forecast(original_lat, original_lon, api_key)
        if isinstance(forecast, list):
            print(f"Weather forecast: {forecast}")

            result = process_forecast(forecast, location_info)

            with open('apriori_rules.pkl', 'rb') as f:
                apriori_rules = pickle.load(f)

            result_df = pd.DataFrame(result)
            result_df['Probability'] = result_df.apply(lambda row: calculate_probability(row, apriori_rules), axis=1)
            
            # Generate dates for the next 7 days
            today = datetime.now()
            result_df['Date'] = [today + timedelta(days=i) for i in range(len(result_df))]
            
            st.subheader(f"Probabilities for {state_name}:")
            for _, row in result_df.iterrows():
                st.write(f"Date: {row['Date'].date()} - Probability: {row['Probability']:.2%}")
                print(f"Date: {row['Date'].date()} - Probability: {row['Probability']:.2%}")

        else:
            st.write(forecast)
            print(f"Forecast error: {forecast}")
    else:
        st.write(location_info)
        print(f"Location info error: {location_info}")
