import numpy as np
import pandas as pd
import os

# Set random seed for same dataset each time
np.random.seed(42)

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Number of rows
size = 2000

# Generate synthetic feature values
speed_kmph = np.random.randint(30, 121, size)                 # 30 to 120
braking_frequency = np.random.randint(1, 21, size)            # 1 to 20
road_condition = np.random.randint(1, 6, size)                # 1 to 5
tyre_pressure = np.random.uniform(28, 40, size)               # 28 to 40 psi
load_weight = np.random.randint(200, 1001, size)              # 200 to 1000 kg
temperature = np.random.uniform(15, 45, size)                 # 15 to 45 Celsius
steering_angle = np.random.uniform(0, 60, size)               # 0 to 60 degrees

# Improved formula for tire wear calculation
tire_wear = (
    (speed_kmph * 0.25) +
    (braking_frequency * 3.5) +
    (road_condition * 6.0) +
    (load_weight * 0.04) +
    (temperature * 0.5) +
    (steering_angle * 1.8) -
    (tyre_pressure * 2.0) +
    np.random.normal(0, 10, size)   # noise
)

# Normalize tire wear values to range 0 to 100
tire_wear = (tire_wear - tire_wear.min()) / (tire_wear.max() - tire_wear.min()) * 100
tire_wear = np.clip(tire_wear, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    "speed_kmph": speed_kmph,
    "braking_frequency": braking_frequency,
    "road_condition": road_condition,
    "tyre_pressure": tyre_pressure,
    "load_weight": load_weight,
    "temperature": temperature,
    "steering_angle": steering_angle,
    "tire_wear": tire_wear
})

# Save dataset to CSV
df.to_csv("dataset/tire_wear_dataset.csv", index=False)

print("✅ Dataset created successfully!")
print("📁 Saved as: dataset/tire_wear_dataset.csv")
print("📌 Total Records:", len(df))
