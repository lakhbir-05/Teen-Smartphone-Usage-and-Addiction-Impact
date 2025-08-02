
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Page config
st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")

# Load trained CatBoost model
model = joblib.load("catboost_model.pkl")

# Sidebar for Settings
st.sidebar.title("⚙️ Settings")


show_summary = st.sidebar.checkbox("📄 Show Input Summary", value=True)




st.title("📱 Teen Smartphone Addiction Predictor")
st.subheader("🔍 Predict Addiction Risk")

mode = st.radio("Choose Prediction Mode:", ["🔘 Single Entry", "📁 Bulk Prediction via CSV"])

required_cols = [
    'ID', 'Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance',
    'Social_Interactions', 'Exercise_Hours', 'Anxiety_Level',
    'Depression_Level', 'Self_Esteem', 'Parental_Control',
    'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day', 'Apps_Used_Daily',
    'Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education',
    'Family_Communication', 'Weekend_Usage_Hours'
]

def classify(risk):
    if risk < 3:
        return "Low"
    elif risk < 6:
        return "Medium"
    else:
        return "High"

if mode == "🔘 Single Entry":
    # Input fields
    age = st.slider("Age", 10, 25, 16)
    daily_usage = st.slider("Daily Phone Usage (hours)", 0.0, 16.0, 6.0)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    academic_perf = st.slider("Academic Performance (1–10)", 1, 10, 6)
    social_inter = st.slider("Social Interactions (1–10)", 1, 10, 5)
    exercise_hours = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
    anxiety_level = st.slider("Anxiety Level", 1, 10, 4)
    depression_level = st.slider("Depression Level", 1, 10, 4)
    self_esteem = st.slider("Self Esteem", 1, 10, 6)
    parental_control = st.slider("Parental Control", 1, 10, 5)
    screen_before_bed = st.slider("Screen Time Before Bed", 0.0, 5.0, 1.0)
    phone_checks = st.slider("Phone Checks Per Day", 0, 100, 25)
    apps_used = st.slider("Apps Used Daily", 1, 20, 6)
    social_media_time = st.slider("Time on Social Media", 0.0, 10.0, 3.0)
    gaming_time = st.slider("Time on Gaming", 0.0, 10.0, 1.0)
    education_time = st.slider("Time on Education", 0.0, 10.0, 2.0)
    family_comm = st.slider("Family Communication", 1, 10, 6)
    weekend_usage = st.slider("Weekend Usage", 0.0, 20.0, 8.0)

    input_data = pd.DataFrame([[0, age, daily_usage, sleep_hours, academic_perf, social_inter,
        exercise_hours, anxiety_level, depression_level, self_esteem,
        parental_control, screen_before_bed, phone_checks, apps_used,
        social_media_time, gaming_time, education_time,
        family_comm, weekend_usage]], columns=required_cols)

    if st.button("📊 Predict Addiction Level"):
        prediction = model.predict(input_data)[0]
        category = classify(prediction)

        if category == "Low":
            st.markdown("<div class='risk low'>🟢 Low Risk: Keep up the healthy balance ✅</div>", unsafe_allow_html=True)
        elif category == "Medium":
            st.markdown("<div class='risk medium'>🟡 Medium Risk: Monitor your screen habits ⚠️</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk high'>🔴 High Risk: Consider reducing phone use ⛔</div>", unsafe_allow_html=True)

        st.subheader("📉 Addiction Meter")
        st.success(f"Predicted Addiction Score: {prediction:.2f}")
        st.progress(min(int(prediction * 10), 100))

        if show_summary:
            st.subheader("📄 Input Summary")
            st.write(input_data.T)

elif mode == "📁 Bulk Prediction via CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if all(col in df.columns for col in required_cols):
                predictions = model.predict(df)
                df["Predicted_Addiction_Level"] = predictions
                df["Risk_Category"] = df["Predicted_Addiction_Level"].apply(classify)

                st.subheader("📋 Predictions")
                st.dataframe(df.head())

                st.subheader("📊 Risk Segmentation")
                risk_counts = df["Risk_Category"].value_counts()
                fig, ax = plt.subplots()
                ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", colors=["green", "orange", "red"])
                ax.axis("equal")
                st.pyplot(fig)

                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download CSV with Predictions", data=csv, file_name="predicted_addiction_levels.csv", mime="text/csv")
            else:
                st.error("❌ Uploaded CSV is missing required columns.")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
