import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Load trained model
# -------------------------
model = joblib.load("models/delivery_eta_model.joblib")


# -------------------------
# Prediction page
# -------------------------
def show_prediction():

    # -------------------------
    # Title
    # -------------------------
    st.title("🚚 Delivery ETA Predictor")

    st.write(
        "Enter the delivery information below to predict "
        "the estimated delivery time."
    )

    # -------------------------
    # Input form
    # -------------------------
    with st.form("delivery_form"):

        st.subheader("Delivery Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Delivery Person Age",
                min_value=18,
                max_value=60,
                value=30
            )

            rating = st.number_input(
                "Delivery Person Rating",
                min_value=1.0,
                max_value=5.0,
                value=4.5,
                step=0.1
            )

            vehicle_condition = st.number_input(
                "Vehicle Condition",
                min_value=0,
                max_value=5,
                value=1,
                step=1
            )

            multiple_deliveries = st.number_input(
                "Multiple Deliveries",
                min_value=0,
                max_value=5,
                value=1,
                step=1
            )

            preparation_time = st.number_input(
                "Preparation Time (minutes)",
                min_value=0.0,
                max_value=120.0,
                value=15.0,
                step=1.0
            )

            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1
            )

        with col2:
            weather = st.selectbox(
                "Weather Conditions",
                [
                    " Sunny",
                    " Stormy",
                    " Sandstorms",
                    " Cloudy",
                    " Fog",
                    " Windy"
                ]
            )

            traffic = st.selectbox(
                "Road Traffic Density",
                [
                    "High",
                    "Jam",
                    "Low",
                    "Medium"
                ]
            )

            order_type = st.selectbox(
                "Type of Order",
                [
                    "Snack",
                    "Drinks",
                    "Buffet",
                    "Meal"
                ]
            )

            vehicle_type = st.selectbox(
                "Type of Vehicle",
                [
                    "motorcycle",
                    "scooter",
                    "electric_scooter",
                    "bicycle"
                ]
            )

            festival = st.selectbox(
                "Festival",
                [
                    "No",
                    "Yes"
                ]
            )

            city = st.selectbox(
                "City",
                [
                    "Urban",
                    "Metropolitian",
                    "Semi-Urban"
                ]
            )

        # -------------------------
        # Location
        # -------------------------

        st.subheader("Location")

        col3, col4 = st.columns(2)

        with col3:
            restaurant_latitude = st.number_input(
                "Restaurant Latitude",
                value=19.0,
                format="%.6f"
            )

            restaurant_longitude = st.number_input(
                "Restaurant Longitude",
                value=75.0,
                format="%.6f"
            )

        with col4:
            delivery_latitude = st.number_input(
                "Delivery Location Latitude",
                value=19.05,
                format="%.6f"
            )

            delivery_longitude = st.number_input(
                "Delivery Location Longitude",
                value=75.05,
                format="%.6f"
            )

        submitted = st.form_submit_button(
            "Predict Delivery Time"
        )

    # -------------------------
    # Prediction
    # -------------------------

    if submitted:

        input_data = pd.DataFrame({
            "Delivery_person_Age": [age],
            "Delivery_person_Ratings": [rating],
            "Restaurant_latitude": [restaurant_latitude],
            "Restaurant_longitude": [restaurant_longitude],
            "Delivery_location_latitude": [delivery_latitude],
            "Delivery_location_longitude": [delivery_longitude],
            "Vehicle_condition": [vehicle_condition],
            "multiple_deliveries": [multiple_deliveries],
            "Preparation_Time_min": [preparation_time],
            "Distance_km": [distance],
            "Weatherconditions": [weather],
            "Road_traffic_density": [traffic],
            "Type_of_order": [order_type],
            "Type_of_vehicle": [vehicle_type],
            "Festival": [festival],
            "City": [city]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Estimated delivery time: **{prediction:.1f} minutes**"
        )
