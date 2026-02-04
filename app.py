import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction")

st.markdown("Enter customer details to predict churn.")

with st.form("churn_form"):
    account_length = st.number_input("Account Length", min_value=0)
    total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)
    total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0)
    total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0)
    total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
    number_vmail_messages = st.number_input("Voicemail Messages", min_value=0)

    international_plan = st.selectbox(
        "International Plan", ["yes", "no"]
    )
    voice_mail_plan = st.selectbox(
        "Voice Mail Plan", ["yes", "no"]
    )

    area_code = st.selectbox(
        "Area Code", [408, 415, 510]
    )

    submit = st.form_submit_button("Predict Churn")

if submit:
    payload = {
        "account_length": account_length,
        "total_day_minutes": total_day_minutes,
        "total_eve_minutes": total_eve_minutes,
        "total_night_minutes": total_night_minutes,
        "total_intl_minutes": total_intl_minutes,
        "customer_service_calls": customer_service_calls,
        "number_vmail_messages": number_vmail_messages,
        "international_plan": international_plan,
        "voice_mail_plan": voice_mail_plan,
        "area_code": area_code,
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        if response.status_code == 200:
            if result["churn_prediction"] == 1:
                st.error("🚨 Customer is likely to churn")
            else:
                st.success("✅ Customer is unlikely to churn")

            st.json(result)
        else:
            st.error("Prediction failed")

    except Exception as e:
        st.error(f"API error: {e}")
