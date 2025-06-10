import streamlit as st
import pandas as pd
import joblib
from src.data import load_raw_data, clean_data

df = load_raw_data('data/raw/houses.csv')
X,y = clean_data(df)

pipe = joblib.load('model/house_price_pipeline.joblib')

st.set_page_config(
    page_title=r"Houses's Selling Price Prediction",
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title(r"Houses's Selling Price Prediction")

st.write("Enter all important informations about your property to get an estimation of the selling price")
with st.form(key='house_data'):
    rooms = st.number_input("Number of Rooms", min_value=1)
    floor = st.selectbox("Number of Floors", [1, 2, 3])

    bedrooms = st.number_input("Number of bedrooms", min_value=1)
    living = st.number_input("Living Area (m²)", min_value=10, max_value=500)

    Type = st.selectbox("Type of the property", X['Type'].unique())
    Condition = st.selectbox("Select condition of the property", X['Condition'].unique())

    Standing = st.selectbox("Select Standing", ['High luxury', 'Luxury', 'Regular'])
    construction_year = st.number_input('Construction year', min_value=1800, max_value=2025, value=2000)

    renovation_year = st.number_input('Renovation year', min_value=1800, max_value=2025, value=2015)
    heating = st.selectbox('Heating', X['Heating'].unique())

    hot_water = st.selectbox('Hot water', ['Gas boiler', 'Electric heater', 'Solar', 'District heating', 'None'])
    energy_rating = st.selectbox('Energy efficiency rating', X['Energy efficiency rating'].unique())

    co2_rating = st.selectbox('Environmental (CO₂) impact rating', X['Environmental (CO₂) impact rating'].unique())
    total_surface = st.number_input('Total surface (m²)', min_value=10.0, max_value=1000.0, value=100.0)

    bathrooms = st.number_input('Number of bathrooms', min_value=0, max_value=10, value=1)

    toilet_rooms = st.number_input('Number of toilet rooms', min_value=0, max_value=10, value=1)
    shower_rooms = st.number_input('Number of shower rooms', min_value=0, max_value=10, value=1)

    land_surface = st.number_input('Land surface (m²)', min_value=0.0, max_value=5000.0, value=300.0)
    balconies = st.number_input('Number of balconies', min_value=0, max_value=10, value=1)

    balcony_surface = st.number_input('Balcony surface (m²)', min_value=0.0, max_value=200.0, value=10.0)
    outside_parking = st.number_input('Outside parking lots', min_value=0, max_value=10, value=1)

    inside_parking = st.number_input('Inside parking lots', min_value=0, max_value=10, value=0)
    internal_surface = st.number_input('Internal surface (m²)', min_value=0.0, max_value=500.0, value=90.0)

    terraces = st.number_input('Number of terraces', min_value=0, max_value=10, value=1)
    outside_garages = st.number_input('Outside garages', min_value=0, max_value=10, value=1)

    environment = st.selectbox('Environment', ['Calm', 'Residential'])
    st.form_submit_button("Done")

user_data = {
    'Rooms':rooms,
    'Floors':floor,
    'Bedrooms':bedrooms,
    'Living':living,
    'Type':Type,
    'Condition':Condition,
    'Standing':Standing,
    'Construction year':construction_year,
    'Renovation year':renovation_year,
    'Heating':heating,
    'Hot water':hot_water,
    'Energy efficiency rating':energy_rating,
    'Environmental (CO₂) impact rating':co2_rating,
    'Total':total_surface,
    'Hot water waste':'Main drainage',
    'Bathrooms':bathrooms,
    'Toilet rooms':toilet_rooms,
    'Shower rooms':shower_rooms,
    'Land':land_surface,
    'Balconies':balconies,
    'Balcony':balcony_surface,
    'Parking lots (outside)':outside_parking,
    'Parking lots (inside)':inside_parking,
    'Internal':internal_surface,
    'Terraces':terraces,
    'Garages (outside)': outside_garages,
    'Environment':environment
}

input_df = pd.DataFrame([user_data])
predictions = pipe.predict(input_df)[0]

if predictions != None:
    st.markdown(
    f"<h1 style='text-align: center; color: green;'>Predicted Price: €{predictions:,}</h1>",
    unsafe_allow_html=True)
    st.write(f"Thank you for testing me :) \n made by Juba")

