# File: fraud_detection_autoencoder.py

"""
Fraud Detection using AutoEncoder (PyOD)

This script trains an AutoEncoder model to detect fraudulent transactions
based on reconstruction error. Fraud cases are expected to have higher
reconstruction error since they differ from normal transaction patterns.
"""

import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Download dataset from Kaggle and place in same directory
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)

# -------------------------------
# Step 2: Preprocessing
# -------------------------------
# Drop Time column (not useful for anomaly detection)
data = data.drop(["Time"], axis=1)

# Separate features and label
X = data.drop("Class", axis=1)
y = data["Class"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train AutoEncoder Model
# -------------------------------
model = AutoEncoder(
    epochs=20,
    batch_size=32,
    contamination=0.01,
    verbose=1
)

model.fit(X_train)

# -------------------------------
# Step 5: Prediction
# -------------------------------
y_pred = model.predict(X_test)  # 0 = normal, 1 = anomaly

# -------------------------------
# Step 6: Evaluation
# -------------------------------
fraud_detected = np.sum(y_pred)
actual_fraud = np.sum(y_test)

print("\nResults:")
print("Actual Fraud Cases:", actual_fraud)
print("Detected Fraud Cases:", fraud_detected)

# Reconstruction error
reconstruction_error = model.decision_function(X_test)

print("\nSample Reconstruction Errors:")
print(reconstruction_error[:10])
