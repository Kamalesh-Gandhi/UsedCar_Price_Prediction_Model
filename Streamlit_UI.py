import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import json
from sklearn.preprocessing import LabelEncoder


# Set page configuration
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

# Load the trained model
model = joblib.load('xgboost_best_model.pkl')  

# Load categorical encoders
encoders = joblib.load('categorical_encoders.pkl')  

# Load dataset from CSV to extract unique values for dropdown options
df_cars = pd.read_csv(r'DataSets\New_Structured_Data\Temp_preprocessed_data.csv')

# Load CSS file
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the CSS
load_css("Customization_styles.css")

# Create brand-model mapping
categorical_columns = ['oem', 'model', 'Fuel Type', 'Transmission', 'city', 'bt', 'Insurance Validity']
unique_values = {col: df_cars[col].unique().tolist() for col in categorical_columns}
brand_model_mapping = df_cars.groupby('oem')['model'].apply(list).to_dict()

# Header
st.markdown(
    """
    <div class="header">
        <span>Car</span><span class="orange">Dheko</span>
        <div class="subtitle">Used Car Price Prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs for Inputs
tab1, tab2, tab3 = st.tabs(["**Car Details**", "**Performance & Usage**", "**Condition & Ratings**"])

# Tab 1: Car Details
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.selected_oem = st.selectbox(" Company 🚗", unique_values['oem'], index=0, key="oem")
    with col2:
        st.session_state.model_options = list(set(brand_model_mapping.get(st.session_state.selected_oem, [])))
        model_name = st.selectbox("Model", st.session_state.model_options, index=0, key="model")

    fuel_type = st.selectbox("Fuel Type ⛽", unique_values['Fuel Type'], index=0)
    transmission = st.selectbox("🔧 Transmission", unique_values['Transmission'], index=0)

# Tab 2: Performance & Usage
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        model_year = st.number_input("Model Year 📅", min_value=2000, max_value=2024, step=1)
        registration_year = st.number_input("Registration Year 📅", min_value=2000, max_value=2024, step=1)
    with col2:
        mileage = st.number_input("Mileage (km/l) ⛽", min_value=0.0, step=0.1, format="%.1f")
        km = st.number_input("Kilometers Driven ", min_value=0, step=1000)

    gear_box = st.number_input("Gear Box (Speed) ⚙️", min_value=1, step=1)

# Tab 3: Condition & Ratings
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        insurance_validity = st.selectbox(" Insurance Validity 📜", unique_values['Insurance Validity'], index=0)
        safety = st.slider(" Safety Rating (0-10) 🛡️", min_value=0, max_value=10, step=1)
    with col2:
        interior = st.slider(" Interior Rating (0-10)", min_value=0, max_value=10, step=1)
        exterior = st.slider(" Exterior Rating (0-10)", min_value=0, max_value=10, step=1)

# Sidebar for Additional Information
st.sidebar.image("https://cdn.infomance.com/wp-content/uploads/car-dekho-1-1920x1005.webp", caption="Car Price Predictor", use_container_width=True)
st.sidebar.header("Additional Information")
city = st.sidebar.selectbox(" City 🏙️", unique_values['city'], index=0)
bt = st.sidebar.selectbox(" Boot Type 🧳", unique_values['bt'], index=0)
owner_no = st.sidebar.number_input("Number of Owners", min_value=1, max_value=10, step=1)

# Predict Price Button

if st.button("**Predict Price**", key="predict"):
    # Create a placeholder for the loading indicator
    placeholder = st.empty()  # Placeholder for loading message
        
    # Display loading message
    placeholder.info("⏳ Predicting the price... Please wait.")
        
    # Simulate loading time
    time.sleep(2)

    # Encode categorical inputs
    input_data = {
            'oem': encoders['oem'].transform([st.session_state.selected_oem])[0],
            'model': encoders['model'].transform([model_name])[0],
            'Fuel Type': encoders['Fuel Type'].transform([fuel_type])[0],
            'Transmission': encoders['Transmission'].transform([transmission])[0],
            'Insurance Validity': encoders['Insurance Validity'].transform([insurance_validity])[0],
            'city': encoders['city'].transform([city])[0],
            'bt': encoders['bt'].transform([bt])[0],
            'modelYear': model_year,
            'Registration Year': registration_year,
            'Mileage': mileage,
            'ownerNo': owner_no,
            'Gear Box': gear_box,
            'km': km,
            'Safety': safety,
            'Interior': interior,
            'Exterior': exterior,
        }

    # Prepare input for prediction
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    predicted_price = model.predict(input_array)[0]

    # Replace loading message with the result
    placeholder.success(f"🎉 Predicted Price: ₹ {predicted_price:,.2f}")
    



