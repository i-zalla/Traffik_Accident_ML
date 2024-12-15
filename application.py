import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import os 

st.set_page_config(
    page_title="Traffic Accident Prediction",
    page_icon="🌆",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load necessary data and models
df_ = pd.read_csv('./dataset/association_ds.csv')

with open('./models/scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_location_info(state_name, file_path="./dataset/result_roud4_scaler2.csv"):
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

# Sidebar navigation
pages = ["Home", "Predict"]
selected_page = st.sidebar.radio("Go to", pages)

def home_page():
    st.title("Traffic Accident Prediction in the UK")
    
    # Displaying three medium-sized emojis side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center; font-size: 100px;'>🗺️</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align: center; font-size: 100px;'>☁️</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align: center; font-size: 100px;'>🚗</div>", unsafe_allow_html=True)

    st.write(
        """
        ### What can you do with our system?
        Our Traffic Accident Prediction system allows you to:

        - **Predict Accidents**: Get accurate predictions for traffic accidents for the next 7 days in various towns across the UK.
        - **Visualize Data**: Explore detailed maps and visualizations to understand accident probabilities and potential hotspots.
        - **Enhance Road Safety**: Gain insights to improve road safety and make informed decisions to prevent accidents.

        ### Why is this useful?
        This system provides valuable information to:

        - **Drivers**: Stay informed about accident probabilities and plan routes accordingly.
        - **Authorities**: Implement safety measures in high-risk areas to prevent accidents.
        - **Researchers**: Analyze patterns and trends in traffic accidents for further study.
        """
    )
    
    # Team section with purple background and names
    st.markdown(
        """
        <style>
        .team-section {
            background-color: #E0BBE4;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .team-member {
            display: inline-block;
            margin: 10px;
        }
        .team-member-name {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        .supervisor-section {
            background-color: #ADD8E6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="team-section">
        <p><b>Our team</b></p>
            <div class="team-member">
                <p class="team-member-name">Zalla Ikram</p>
            </div>
            <div class="team-member">
                <p class="team-member-name">Zouggari Imene</p>
            </div>
            <div class="team-member">
                <p class="team-member-name">Aouaichia Hafsa</p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="supervisor-section">
            <p class="team-member-name">Our Supervisor: Dif Nassima</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Prediction page
def predict_page():
    st.title("Traffic Accident Prediction")

    # Select box for town
    st.subheader("Input Parameters")
    town_options = ["Trowbridge", "Church Village", "Highbridge and Burnham Marine", "Llanishen", "Pontprennau/Old St. Mellons",
                    "Ely South", "Leggatts", "The Hemingfords", "Overton, Laverstoke and Steventon", "Ryemead",
                    "Greenhill", "Yarborough", "Otley and Yeadon", "Holmebrook", "Hucknall North",
                    "Clackmannanshire East", "Johnstone South, Elderslie & Howwood", "Shetland Central", "Cupar", "Sighthill/Gorgie",
                    "Chapelford & Old Hall", "Halewood North", "Charlestown", "Poynton West and Adlington", "Kersal"]

    town = st.selectbox("Select the town:", options=town_options + ["Other"])

    if town == "Other":
        town = st.text_input("Please enter the town:")

    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []

    if st.button("Get Result"):
        today = datetime.date.today()
        st.success(f"Showing results for the town: {town}")

        def get_color(prob):
            if prob < 30:
                return "green"
            elif prob < 67.5:
                return "orange"
            else:
                return "red"

        location_info = get_location_info(town)
        if isinstance(location_info, dict):
            print(f"Location info: {location_info}")

            scaled_input = [[location_info['longitude'], location_info['latitude'], 0, 0]]
            original_coords = scaler.inverse_transform(scaled_input)
            original_lon = original_coords[0][0]
            original_lat = original_coords[0][1]
            api_key = '8b3581aad7ba4840989135658241512'
            forecast = get_weather_forecast(original_lat, original_lon, api_key)
            if isinstance(forecast, list):
                print(f"Weather forecast: {forecast}")

                result = process_forecast(forecast, location_info)

                with open('./models/apriori_rules.pkl', 'rb') as f:
                    apriori_rules = pickle.load(f)

                result_df = pd.DataFrame(result)
                result_df['Probability'] = result_df.apply(lambda row: calculate_probability(row, apriori_rules) * 100, axis=1)

                # Tabbed interface
                tabs = st.tabs(["Weekly Overview"] + [f"{forecast[idx]['Date']}" for idx in range(len(forecast))])
                
                with tabs[0]:
                    st.subheader("Accident Probability for the Next 7 Days")
                    dates = [forecast[idx]['Date'] for idx in range(len(result_df))]
                    probabilities = [row['Probability'] for idx, row in result_df.iterrows()]

                    # Bar Chart
                    st.subheader("Bar Chart")
                    plt.figure(figsize=(10, 5))
                    sns.barplot(x=dates, y=probabilities)
                    plt.ylim(0, 100)  # Setting y-axis from 0 to 100
                    plt.xticks(rotation=45)
                    plt.title('Traffic Accident Probability - Bar Chart')
                    plt.ylabel('Probability (%)')
                    st.pyplot(plt)

                    # Line Chart
                    st.subheader("Line Chart")
                    plt.figure(figsize=(10, 5))
                    plt.plot(dates, probabilities, marker='o', linestyle='-', color='b')
                    plt.ylim(0, 100)  # Setting y-axis from 0 to 100
                    plt.xlabel('Date')
                    plt.ylabel('Probability (%)')
                    plt.title('Traffic Accident Probability Over the Next 7 Days')
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    st.pyplot(plt)
                
                for idx, tab in enumerate(tabs[1:], start=0):
                    with tab:
                        date = forecast[idx]['Date']
                        prob = result_df.at[idx, 'Probability']
                        st.subheader(f"Probability for {date}")
                        st.markdown(f"**Probability:** {prob:.4f}%")
                        
                        # Map
                        st.subheader("Map")
                        color = get_color(prob)
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=pd.DataFrame([{"lon": original_lon, "lat": original_lat, "probability": prob}]),
                            get_position=["lon", "lat"],
                            get_fill_color=[200, 30, 0, 160] if color == "red" else [255, 165, 0, 160] if color == "orange" else [0, 128, 0, 160],
                            get_radius=50000,
                            pickable=True,
                        )
                        view_state = pdk.ViewState(
                            latitude=original_lat,
                            longitude=original_lon,
                            zoom=6,
                            pitch=50,
                        )
                        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

            else:
                st.write(forecast)
                print(f"Forecast error: {forecast}")
        else:
            st.write(location_info)
            print(f"Location info error: {location_info}")

if selected_page == "Home":
    home_page()
elif selected_page == "Predict":
    predict_page()
