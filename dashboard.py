import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import os

# Load model and scaler
model = joblib.load('model/tire_wear_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# History file path
HISTORY_FILE = "history.csv"

# Configure page theme and background color
st.set_page_config(
    page_title="Tire Wear Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background color and styling
st.markdown("""
<style>
    body, .main {
        background: linear-gradient(135deg, #e6f2ff 0%, #cce0ff 50%, #b3d9ff 100%);
        background-image: 
            url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><defs><pattern id="tech" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="2" fill="rgba(31,119,180,0.1)"/><circle cx="50" cy="50" r="2" fill="rgba(31,119,180,0.1)"/><circle cx="90" cy="90" r="2" fill="rgba(31,119,180,0.1)"/><line x1="10" y1="10" x2="50" y2="50" stroke="rgba(31,119,180,0.05)" stroke-width="0.5"/><line x1="50" y1="50" x2="90" y2="90" stroke="rgba(31,119,180,0.05)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23tech)"/></svg>');
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #e8eef2;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Subheader styling */
    h2 {
        color: #264e86;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    /* Container styling */
    .stContainer {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* prevent vertical scrolling and force page to fit viewport */
    html, body, .block-container {
        overflow-y: hidden;
        height: 100vh;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Tire Wear Prediction Dashboard")

st.sidebar.header("Enter Vehicle Data")

# Optional dataset preview
with st.sidebar.expander("🔍 View raw dataset"):
    try:
        raw = pd.read_csv("dataset/tire_wear_dataset.csv")
        st.dataframe(raw.head(100))
    except Exception as e:
        st.write("Could not load dataset:", e)

# User inputs
distance = st.sidebar.slider("Speed (kmph)", 30, 120, 60)
pressure = st.sidebar.slider("Braking Frequency", 1, 20, 5)
road = st.sidebar.slider("Road Condition (1=Good, 5=Bad)", 1, 5, 3)
style = st.sidebar.slider("Tyre Pressure (psi)", 28, 40, 32)
temp = st.sidebar.slider("Temperature (°C)", 15, 45, 30)
weight = st.sidebar.slider("Load Weight (kg)", 200, 1000, 500)
steering = st.sidebar.slider("Steering Angle (degrees)", 0, 60, 20)

# Prediction

# save history file name
history_file = "history.csv"

data = [[distance, pressure, road, style, temp, weight, steering]]
data_scaled = scaler.transform(data)
prediction = model.predict(data_scaled)[0]

# append prediction to history
row = [distance, pressure, road, style, temp, weight, steering, prediction]
if os.path.exists(history_file):
    with open(history_file, "a") as f:
        f.write(",".join(map(str, row)) + "\n")
else:
    with open(history_file, "w") as f:
        f.write("Speed_kmph,Braking_Frequency,Road_Condition,Tyre_Pressure,Temperature,Load_Weight,Steering_Angle,Wear\n")
        f.write(",".join(map(str, row)) + "\n")


# arrange results and visualization side by side to reduce height
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔍 Prediction Result")
    st.write(f"Predicted Tire Wear: **{prediction:.2f}%**")

    # Alerts
    if prediction > 75:
        st.error("⚠️ Critical Condition! Replace Tire Immediately")
    elif prediction > 40:
        st.warning("⚠️ Moderate Wear. Maintenance Needed")
    else:
        st.success("✅ Tire Condition is Good")

with col2:
    st.subheader("📊 Wear Visualization")
    df = pd.DataFrame({
        'Speed': [distance],
        'Wear': [prediction]
    })
    st.line_chart(df.set_index('Speed'))

# Save to history (update file after prediction)
new_row = {
    "Speed_kmph": distance,
    "Braking_Frequency": pressure,
    "Road_Condition": road,
    "Tyre_Pressure": style,
    "Temperature": temp,
    "Load_Weight": weight,
    "Steering_Angle": steering,
    "Wear": prediction
}
if os.path.exists(HISTORY_FILE):
    hist_df = pd.read_csv(HISTORY_FILE)
    hist_df = hist_df.append(new_row, ignore_index=True)
else:
    hist_df = pd.DataFrame([new_row])
hist_df.to_csv(HISTORY_FILE, index=False)

# use tabs for the extra sections to keep page short
tab1, tab2 = st.tabs(["Additional Charts", "History"])

with tab1:
    st.subheader("📈 Additional Charts")
    try:
        raw = pd.read_csv("dataset/tire_wear_dataset.csv")
        fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].scatter(raw['speed_kmph'], raw['tire_wear'], alpha=0.5)
        axs[0].set_title("Wear vs Speed")
        axs[0].set_xlabel("Speed")

        axs[1].scatter(raw['tyre_pressure'], raw['tire_wear'], color='orange', alpha=0.5)
        axs[1].set_title("Wear vs Pressure")
        axs[1].set_xlabel("Pressure")

        axs[2].scatter(raw['temperature'], raw['tire_wear'], color='green', alpha=0.5)
        axs[2].set_title("Wear vs Temp")
        axs[2].set_xlabel("Temperature")

        st.pyplot(fig2)
    except Exception:
        st.write("Could not generate additional charts.")

with tab2:
    st.subheader("📁 Prediction History")
    if st.button("Clear history"):
        open(HISTORY_FILE, "w").close()
        st.success("History cleared")
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        st.dataframe(history)
        csv = history.to_csv(index=False)
        st.download_button("🗂️ Download history as CSV", csv, "history.csv", "text/csv")
    else:
        st.write("No history available yet.")
with tab3:
    st.subheader("📊 Dataset Statistics")
    try:
        dataset = pd.read_csv("dataset/tire_wear_dataset.csv")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Records", len(dataset))
        with col_stat2:
            st.metric("Avg Wear", f"{dataset['tire_wear'].mean():.2f}%")
        with col_stat3:
            st.metric("Max Wear", f"{dataset['tire_wear'].max():.2f}%")
        
        st.subheader("📉 Wear Distribution")
        fig_dist, ax_dist = plt.subplots()
        ax_dist.hist(dataset['tire_wear'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax_dist.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Your Prediction: {prediction:.2f}%')
        ax_dist.set_xlabel("Wear Level (%)")
        ax_dist.set_ylabel("Frequency")
        ax_dist.legend()
        st.pyplot(fig_dist)
        
        st.subheader("🔗 Feature Correlations")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr_matrix = dataset.corr()
        im = ax_corr.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax_corr.set_xticks(range(len(corr_matrix.columns)))
        ax_corr.set_yticks(range(len(corr_matrix.columns)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax_corr.set_yticklabels(corr_matrix.columns, fontsize=8)
        plt.colorbar(im, ax=ax_corr)
        st.pyplot(fig_corr)
    except Exception as e:
        st.write(f"Could not load dataset stats: {e}")

with tab4:
    st.subheader("💡 Model Insights & Recommendations")
    
    insights = f"""
    **Your Current Status:**
    - Wear Level: {prediction:.2f}%
    - Operating Distance: {distance} km
    - Tire Pressure: {pressure} psi
    - Road Type: Type {road}
    - Driving Style: Style {style}
    - Temperature: {temp}°C
    
    **Key Insights:**
    """
    
    if distance > 15000:
        insights += "\n- ⚠️ **High mileage detected** – Consider tire rotation or replacement soon"
    if pressure < 25 or pressure > 35:
        insights += "\n- ⚠️ **Pressure out of normal range** – Adjust to optimal (28-32 psi)"
    if temp > 40:
        insights += "\n- ⚠️ **High temperature** – This accelerates tire wear; reduce speed"
    if style == 3:
        insights += "\n- ℹ️ **Aggressive driving style detected** – Smooth acceleration reduces wear"
    if road == 3:
        insights += "\n- ℹ️ **Rough road type** – Off-road driving causes faster wear"
    
    if prediction < 30:
        insights += "\n- ✅ **Excellent condition** – Maintain current driving habits"
    elif prediction < 50:
        insights += "\n- ℹ️ **Good condition** – Continue monitoring; inspect in 1-2 months"
    
    insights += """
    
    **Tips to Reduce Tire Wear:**
    1. Maintain proper tire pressure (28-32 psi)
    2. Avoid sudden acceleration/braking
    3. Drive on smooth, well-maintained roads when possible
    4. Reduce speed on rough terrain
    5. Regular tire rotation every 5,000-7,000 km
    6. Align wheels annually
    7. Avoid extreme temperatures
    """
    
    st.markdown(insights)
# Additional visualizations
st.subheader("📈 Additional Charts")
try:
    raw = pd.read_csv("dataset/tire_wear_dataset.csv")
    fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].scatter(raw['speed_kmph'], raw['tire_wear'], alpha=0.5)
    axs[0].set_title("Wear vs Speed")
    axs[0].set_xlabel("Speed")

    axs[1].scatter(raw['tyre_pressure'], raw['tire_wear'], color='orange', alpha=0.5)
    axs[1].set_title("Wear vs Pressure")
    axs[1].set_xlabel("Pressure")

    axs[2].scatter(raw['temperature'], raw['tire_wear'], color='green', alpha=0.5)
    axs[2].set_title("Wear vs Temp")
    axs[2].set_xlabel("Temperature")

    st.pyplot(fig2)
except Exception:
    st.write("Could not generate additional charts.")

# History
st.subheader("📁 Prediction History")

if st.button("Clear history"):
    open(HISTORY_FILE, "w").close()
    st.success("History cleared")
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
    st.dataframe(history)
    csv = history.to_csv(index=False)
    st.download_button("⬇️ Download history as CSV", csv, "history.csv", "text/csv")
else:
    st.write("No history available yet.")