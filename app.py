import streamlit as st
import requests
from src.logger import logging

API_URL = "http://localhost:8000/predict"

logging.info("Streamlit app started")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title(" Customer Churn Prediction")

st.markdown("Enter customer details to predict churn.")

with st.form("churn_form"):
    account_length = st.number_input("Account Length", min_value=1)
    total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)
    total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0)
    total_night_minutes = st.number_input("Total Night Minutes", min_value=23.2)
    total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
    number_vmail_messages = st.number_input("Voicemail Messages", min_value=0)
    total_day_calls = st.number_input("Total Day Calls", min_value=0)
    total_eve_calls = st.number_input("Total Evening Calls", min_value=0)
    total_night_calls = st.number_input("Total Night Calls", min_value=33)
    total_intl_calls = st.number_input("Total International Calls", min_value=0)

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
    
    logging.info("Prediction button clicked")
    
    payload = {
        "account_length": account_length,
        "total_day_minutes": total_day_minutes,
        "total_eve_minutes": total_eve_minutes,
        "total_night_minutes": total_night_minutes,
        "total_intl_minutes": total_intl_minutes,
        "customer_service_calls": customer_service_calls,
        "number_vmail_messages": number_vmail_messages,
        "total_day_calls": total_day_calls,
        "total_eve_calls": total_eve_calls,
        "total_night_calls": total_night_calls,
        "total_intl_calls": total_intl_calls,
        "international_plan": international_plan,
        "voice_mail_plan": voice_mail_plan,
        "area_code": area_code,
    }

    logging.info(f"Payload created: {payload}")
    
    try:
        
        logging.info("Sending request to FastAPI backend")
        
        response = requests.post(API_URL, json=payload)
        logging.info(f"Response status code: {response.status_code}")
        
        result = response.json()
        
        logging.info(f"Response received: {result}")

        if response.status_code == 200:
            if result["churn_prediction"] == 1:
                logging.info("Prediction result: Customer likely to churn")
                st.error("🚨 Customer is likely to churn")
            else:
                st.success("✅ Customer is unlikely to churn")
                logging.info("Prediction result: Customer unlikely to churn")

            st.json(result)
        else:
            logging.error("Prediction failed due to non-200 response")
            st.error("Prediction failed")

    except Exception as e:
        logging.exception("Exception occurred during API call")
        st.error(f"API error: {e}")
