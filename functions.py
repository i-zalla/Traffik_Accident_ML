import pandas as pd
import requests
from mlxtend.frequent_patterns import apriori, association_rules

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

def get_weather_forecast(lat, lon, api_key):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={lat},{lon}&days=7"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        weather_data = response.json()

        if 'forecast' in weather_data:
            forecast = []
            for day in weather_data['forecast']['forecastday']:
                date = day['date']
                description = day['day']['condition']['text']
                forecast.append({"Date": date, "Météo": description})
            return forecast
        else:
            return "Les données météorologiques ne sont pas disponibles."
    else:
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

def process_forecast(forecast):

    df = pd.DataFrame(forecast)

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

    return df

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