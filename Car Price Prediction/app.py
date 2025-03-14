import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the model
model = pk.load(open('model.pkl', 'rb'))

# Set page config with title and background image
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-image: url('https://source.unsplash.com/1600x900/?car,luxury');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .main {
            background: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            color: #f8c102;
        }
        .stButton>button {
            background-color: #f8c102 !important;
            color: black !important;
            font-size: 18px !important;
            padding: 10px 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚗 Car Price Prediction Model</h1>", unsafe_allow_html=True)

# Load dataset
cars_data = pd.read_csv('Cardetails.csv')

# Extract brand name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# UI Layout
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        name = st.selectbox('Select Car Brand', cars_data['name'].unique())
        year = st.slider('Car Manufactured Year', 1994, 2024)
        km_driven = st.slider('No of kms Driven', 11, 200000)
        fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
        seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())

    with col2:
        transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
        owner = st.selectbox('Ownership Type', cars_data['owner'].unique())
        mileage = st.slider('Car Mileage (kmpl)', 10, 40)
        engine = st.slider('Engine Capacity (CC)', 700, 5000)
        max_power = st.slider('Max Power (HP)', 0, 200)
        seats = st.slider('No of Seats', 2, 10)

# Prediction
if st.button("🚀 Predict Car Price"):
    input_data_model = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    # Encode categorical variables
    input_data_model.replace({
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'transmission': {'Manual': 1, 'Automatic': 2},
        'name': {brand: idx+1 for idx, brand in enumerate(cars_data['name'].unique())}
    }, inplace=True)

    # Predict price
    car_price = model.predict(input_data_model)
    
    # Display result with styling
    st.markdown(f"""
        <div style="background:#f8c102;padding:15px;border-radius:10px;text-align:center;">
            <h2 style="color:black;">🚘 Estimated Car Price: ₹ {round(car_price[0], 2)}</h2>
        </div>
    """, unsafe_allow_html=True)
