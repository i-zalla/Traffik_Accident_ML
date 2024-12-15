import streamlit as st
import datetime
from datetime import timedelta
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(
    page_title="Traffic Accident Prediction",
    page_icon="🌆",
    layout="centered",
    initial_sidebar_state="expanded",
)

#Sidebar
pages = ["Home", "Predict"]
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", pages)

#Home
def home_page():
    st.title("Predicting Traffic Accidents in the UK")
    st.image("undraw_bike_ride_7xit.svg", caption="Traffic Accident Prediction", use_container_width=True)
    st.write(
        """
        Welcome to our Traffic Accident Prediction system! This tool uses advanced machine learning algorithms to help predict the likelihood of traffic accidents occurring in the UK based on historical data.

        ### Features:
        - Accurate predictions for the next 7 days.
        - Detailed maps and visualizations.
        - Insights to enhance road safety.

        Navigate to the **Predict** page to explore the predictions and visualizations.
        """
    )

#Prediction
def predict_page():
    st.title("Traffic Accident Prediction")

    #Select box 
    st.subheader("Input Parameters")
    town = st.selectbox("Select the town:", ["Trowbridge", "Church Village", "Highbridge and Burnham Marine", "Llanishen", "Pontprennau/Old St. Mellons",
                                             "Ely South", "Leggatts", "The Hemingfords", "Overton, Laverstoke and Steventon", "Ryemead" ,
                                             "Greenhill", "Yarborough", "Otley and Yeadon", "Holmebrook", "Hucknall North" ,
                                             "Clackmannanshire East", "Johnstone South, Elderslie & Howwood", "Shetland Central", "Cupar", "Sighthill/Gorgie" ,
                                             "Chapelford & Old Hall", "Halewood North", "Charlestown", "Poynton West and Adlington", "Kersal" ,
                                             ])

    #predict generation 
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []

    if st.button("Get Result"):
        today = datetime.date.today()
        st.success(f"Showing results for the town: {town}")

        #example for now should be replaced
        predictions = [round(0.1 + 0.1 * i, 2) for i in range(7)]
        days = [today + timedelta(days=i) for i in range(7)]
        
        #Save predictions in session_state to persist them
        st.session_state["predictions"] = (days, predictions)

    if st.session_state["predictions"]:
        days, predictions = st.session_state["predictions"]

        #color coding display
        st.subheader("Accident Probability for the Next 7 Days")
        def get_color(prob):
            if prob < 0.3:
                return "green"
            elif prob < 0.6:
                return "orange"
            else:
                return "red"

        for day, prob in zip(days, predictions):
            color = get_color(prob)
            st.markdown(f"### Date: {day} - Probability: {prob}")
            st.markdown(f"<div style='background-color:{color}; color:white; padding:10px;'>Probability: {prob}</div>", unsafe_allow_html=True)


if selected_page == "Home":
    home_page()
elif selected_page == "Predict":
    predict_page()