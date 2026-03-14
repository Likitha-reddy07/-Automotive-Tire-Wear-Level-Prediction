# Automotive Tire Wear Level Prediction

🚗 Project Overview

This project demonstrates a complete workflow for predicting tire wear levels using a machine learning regression model (Random Forest). It includes dataset generation, model training, and a Streamlit web app for interactive prediction.

🔍 Key Features

✅ Synthetic tire wear dataset generation (CSV output)
✅ Random Forest regression model training and serialization
✅ Streamlit-based web app for live wear prediction
✅ Reproducible pipeline with minimal setup

🧩 Project Structure

app.py – Streamlit dashboard for making predictions
train_model.py – Trains the ML model and saves it to model
dataset_generator.py – Generates tire_wear_dataset.csv dataset
dataset – Contains the generated dataset
model – Saved trained model artifacts and metada
## How to Run

### Step 1: Install requirements
pip install -r requirements.txt

### Step 2: Generate dataset
python utils/dataset_generator.py

### Step 3: Train the model
python train_model.py

### Step 4: Run Streamlit app
streamlit run app.py
