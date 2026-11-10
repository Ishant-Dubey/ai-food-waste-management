"""
AI-Based Food Waste Reduction and Donation Advisory System
Streamlit Frontend
"""

import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load the trained model and preprocessing files
model = joblib.load("models/SVR_Food_Waste.pkl")
expected_columns = joblib.load("models/columns.pkl")
scaler = joblib.load("models/scaler.pkl")


def setup_page():
    st.set_page_config(
        page_title="AI Food Waste Reduction & Donation Advisory",
        page_icon="🌱",
        layout="wide",
    )


def render_header():
    st.title("AI-Based Food Waste Reduction and Donation Advisory System")
    st.write(
        "Helping food businesses estimate food waste, manage their inventory, "
        "and find ways to donate surplus food."
    )
    st.divider()


# Dashboard
def render_dashboard():
    st.header("Dashboard")
    st.write("Keep track of food waste and donation activities from one place.")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.metric("Total Recorded Waste", "-- kg")
            st.caption("Database connection will be added later.")

    with col2:
        with st.container(border=True):
            st.metric("Predicted Waste (Today)", "-- kg")
            st.caption("Waiting for live prediction data.")

    with col3:
        with st.container(border=True):
            st.metric("Donations Facilitated", "-- units")
            st.caption("Donation tracking will be added later.")

    st.subheader("Waste Analytics & Trends")
    with st.container(border=True):
        st.info("Waste trends and visualizations will appear here once data is available.")

    st.subheader("Recent Activity")
    with st.container(border=True):
        st.info("No recent activity yet. Waste predictions and donation records will appear here.")


# Waste Prediction
def render_waste_prediction():
    st.header("Waste Prediction")
    st.write("Enter the details below to estimate how much food waste your business may generate.")

    with st.form("waste_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Business & Environmental Details")

            business_type = st.selectbox(
                "Business Type",
                ["Cafeteria", "Cloud Kitchen", "Restaurant", "Supermarket"],
                help="Select the type of business.",
            )

            food_category = st.selectbox(
                "Food Category",
                ["Bakery", "Dairy", "Fruits", "Meat", "Prepared Food", "Vegetables"],
                help="Choose the category of food.",
            )

            weather = st.selectbox(
                "Weather Condition",
                ["Sunny", "Cloudy", "Rainy", "Hot"],
                help="Select the current weather condition.",
            )

            promotion = st.selectbox(
                "Promotion Active",
                ["No", "Yes"],
                help="Is there an active discount or promotion?",
            )

            prediction_date = st.date_input(
                "Date",
                value=datetime.date.today(),
                help="The selected date is used to calculate weekday and month features.",
            )

            st.caption(f"Day: {prediction_date.strftime('%A')}")

        with col2:
            st.subheader("Inventory Details")

            purchase_quantity = st.number_input(
                "Purchase Quantity (kg / units)",
                min_value=0.0,
                value=50.0,
                step=1.0,
                help="Total quantity of food purchased.",
            )

            daily_demand = st.number_input(
                "Daily Demand (kg / units)",
                min_value=0.0,
                value=30.0,
                step=1.0,
                help="Expected quantity of food sold or used each day.",
            )

            shelf_life_days = st.number_input(
                "Shelf Life (Days)",
                min_value=1,
                value=5,
                step=1,
                help="How many days the food stays fresh under proper storage.",
            )

            storage_temperature = st.number_input(
                "Storage Temperature (°C)",
                value=4.0,
                step=0.5,
                help="Temperature at which the food is stored.",
            )

        submit_btn = st.form_submit_button("Predict Waste", use_container_width=True)

    st.subheader("Prediction Result")

    with st.container(border=True):
        if submit_btn:

            # Convert the selected date into cyclical features
            weekday = prediction_date.weekday()
            weekday_sin = np.sin(2 * np.pi * weekday / 7)
            weekday_cos = np.cos(2 * np.pi * weekday / 7)

            month = prediction_date.month
            month_sin = np.sin(2 * np.pi * (month - 1) / 12)
            month_cos = np.cos(2 * np.pi * (month - 1) / 12)

            # Prepare the input features for the model
            input_features = {
                "Purchase_Quantity": purchase_quantity,
                "Shelf_Life_Days": shelf_life_days,
                "Storage_Temperature": storage_temperature,
                "Daily_Demand": daily_demand,
                "Promotion": 1 if promotion == "Yes" else 0,
                "weekday_sin": round(float(weekday_sin), 4),
                "weekday_cos": round(float(weekday_cos), 4),
                "month_sin": round(float(month_sin), 4),
                "month_cos": round(float(month_cos), 4),
                "Business_Type_" + business_type: 1,
                "Food_Category_" + food_category: 1,
                "Weather_" + weather: 1,
            }

            input_df = pd.DataFrame([input_features])

            # Add missing columns and arrange them in the expected order
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[expected_columns]

            # Scale the numeric features using the saved scaler
            numeric_cols = list(scaler.feature_names_in_)
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            # Get the predicted waste quantity
            prediction = model.predict(input_df)[0]

            with st.expander("View Input Feature Summary"):
                st.json(input_features)

            st.metric(
                "Predicted Food Waste",
                f"{prediction:.2f} kg",
                help="Estimated food waste based on the details provided.",
            )

        else:
            st.write("Fill in the details above and click **Predict Waste** to get an estimate.")


# Donation Advisory
def render_donation_advisory():
    st.header("Donation Advisory")
    st.write(
        "Surplus food that is still safe to consume can be donated to local "
        "charities and food-relief organizations instead of going to waste."
    )

    st.subheader("Recommended Donations")

    with st.container(border=True):
        st.info(
            "This section will be developed later. It will help identify surplus food, "
            "check shelf-life limits, and suggest suitable donation options."
        )


# About the Project
def render_about():
    st.header("About the Project")

    st.write(
        "The **AI-Based Food Waste Reduction and Donation Advisory System** aims to help "
        "restaurants, supermarkets, and other food businesses manage their inventory "
        "and reduce unnecessary food waste."
    )

    st.subheader("Objectives")

    st.markdown(
        """
        - **Predict Waste:** Estimate food waste using a supervised machine learning model.
        - **Reduce Food Loss:** Help businesses make better purchasing and inventory decisions.
        - **Encourage Donations:** Identify surplus food that can be donated to local organizations.
        - **Support Sustainability:** Contribute to UN SDG 12.3, which aims to halve food waste by 2030.
        """
    )


# Main function
def main():
    setup_page()
    render_header()

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Waste Prediction", "Donation Advisory", "About"],
    )

    if page == "Dashboard":
        render_dashboard()

    elif page == "Waste Prediction":
        render_waste_prediction()

    elif page == "Donation Advisory":
        render_donation_advisory()

    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()