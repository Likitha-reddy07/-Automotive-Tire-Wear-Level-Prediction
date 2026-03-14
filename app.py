import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# Load model and scaler
model = joblib.load("model/tire_wear_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Page Config
st.set_page_config(page_title="Tire Wear Prediction Dashboard", page_icon="🚗", layout="wide")

# ---------------- Sidebar Menu ----------------
with st.sidebar:
    st.markdown("## 🚗 Tire Wear Intelligence")
    st.write("Navigate")

    selected = option_menu(
        menu_title="",
        options=["Overview", "Prediction", "Analytics", "Feature Importance"],
        icons=["house", "graph-up", "bar-chart", "list-check", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

# ---------------- Overview Page ----------------
if selected == "Overview":
    st.markdown("# 🚗 Automotive Tire Wear Level Prediction System")
    st.markdown("### 📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", "1000")
    col2.metric("Model Used", "Random Forest")
    col3.metric("Target Output", "Tire Wear %")
    col4.metric("Accuracy (R2 Score)", "0.59")

    st.write("---")



# ---------------- Prediction Page ----------------
elif selected == "Prediction":
    st.markdown("# 🔍 Tire Wear Prediction Page")

    left_col, right_col = st.columns([1.3, 2.7])

    with left_col:
        st.markdown("## 🛠 Input Details")

        speed_kmph = st.slider("Speed (kmph)", 30, 120, 60)
        braking_frequency = st.slider("Braking Frequency", 1, 20, 5)
        road_condition = st.slider("Road Condition (1=Good, 5=Bad)", 1, 5, 3)
        tyre_pressure = st.slider("Tyre Pressure (psi)", 28, 40, 32)
        load_weight = st.slider("Load Weight (kg)", 200, 1000, 500)
        temperature = st.slider("Temperature (°C)", 15, 45, 30)
        steering_angle = st.slider("Steering Angle (degrees)", 0, 60, 20)

    input_data = pd.DataFrame({
        "speed_kmph": [speed_kmph],
        "braking_frequency": [braking_frequency],
        "road_condition": [road_condition],
        "tyre_pressure": [tyre_pressure],
        "load_weight": [load_weight],
        "temperature": [temperature],
        "steering_angle": [steering_angle]
    })

    with right_col:
        st.markdown("## 📌 Input Data Summary")
        st.dataframe(input_data, use_container_width=True)

        st.write("---")
        st.markdown("## 📊 Prediction Output")

        # Automatic prediction
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        st.success(f"🚗 Predicted Tire Wear Level: **{prediction:.2f}%**")

        st.markdown("### 🔥 Wear Indicator")
        st.progress(int(prediction))

        if prediction < 30:
            st.info("🟢 Tire Condition: Good (No maintenance needed)")
        elif prediction < 70:
            st.warning("🟡 Tire Condition: Medium (Service recommended soon)")
        else:
            st.error("🔴 Tire Condition: Critical (Replace tire immediately)")

# ---------------- Analytics Page ----------------
elif selected == "Analytics":
    st.markdown("# 📈 Analytics Dashboard")

    df = pd.read_csv("dataset/tire_wear_dataset.csv")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Min Wear", f"{df['tire_wear'].min():.2f}%")
    col2.metric("Max Wear", f"{df['tire_wear'].max():.2f}%")
    col3.metric("Average Wear", f"{df['tire_wear'].mean():.2f}%")
    col4.metric("Total Samples", df.shape[0])

    st.write("---")
    st.markdown("## 📌 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------- Feature Importance Page ----------------
elif selected == "Feature Importance":
    st.markdown("# ⭐ Feature Importance (Factors Affecting Wear)")

    feature_names = ["speed_kmph", "braking_frequency", "road_condition", "tyre_pressure",
                     "load_weight", "temperature", "steering_angle"]

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feature_df, use_container_width=True)

    st.write("---")
    st.markdown("## 📊 Importance Chart")
    st.bar_chart(feature_df.set_index("Feature"))