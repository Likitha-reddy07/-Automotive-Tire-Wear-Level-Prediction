import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/tire_wear_dataset.csv")

# Features and Target
X = df.drop("tire_wear", axis=1)
y = df["tire_wear"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Training Completed Successfully!")
print("📌 Evaluation Results:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model and scaler
joblib.dump(model, "model/tire_wear_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Files Created Successfully!")
print("📁 model/tire_wear_model.pkl")
print("📁 model/scaler.pkl")
