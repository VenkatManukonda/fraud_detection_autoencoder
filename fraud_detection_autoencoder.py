import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pyod.models.auto_encoder import AutoEncoder


# -------------------------------
# Step 1: Load Dataset
# -------------------------------
data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)


# -------------------------------
# Step 2: Split Features & Labels
# -------------------------------
X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]


# -------------------------------
# Step 3: Normalize Data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Step 5: Build AutoEncoder Model
# -------------------------------
model = AutoEncoder(
    contamination=0.01,
    verbose=1
)


# -------------------------------
# Step 6: Train Model
# -------------------------------
model.fit(X_train)


# -------------------------------
# Step 7: Predict Anomalies
# -------------------------------
y_pred = model.predict(X_test)  # 0 = normal, 1 = fraud/anomaly


# -------------------------------
# Step 8: Results
# -------------------------------
print("\nResults:")
print("Actual Fraud Cases:", np.sum(y_test))
print("Detected Fraud Cases:", np.sum(y_pred))


# -------------------------------
# Step 9: Reconstruction Error
# -------------------------------
scores = model.decision_function(X_test)

print("\nSample Reconstruction Errors:")
print(scores[:10])
