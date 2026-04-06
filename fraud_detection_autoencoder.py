import pandas as pd
import numpy as np

from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# -------------------------------
# STEP 1: LOAD DATA (MISSING IN YOUR FILE)
# -------------------------------
data = pd.read_csv("creditcard.csv")
print("Dataset shape:", data.shape)


# -------------------------------
# STEP 2: SPLIT FEATURES & LABEL
# -------------------------------
X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]


# -------------------------------
# STEP 3: SCALE DATA
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------
# STEP 4: TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# STEP 5: AUTOENCODER MODEL
# -------------------------------
model = AutoEncoder(contamination=0.01, verbose=1)


# -------------------------------
# STEP 6: TRAIN MODEL
# -------------------------------
model.fit(X_train)


# -------------------------------
# STEP 7: PREDICTION
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# STEP 8: RESULTS
# -------------------------------
print("\nResults:")
print("Actual Fraud Cases:", np.sum(y_test))
print("Detected Fraud Cases:", np.sum(y_pred))


# -------------------------------
# STEP 9: RECONSTRUCTION ERROR
# -------------------------------
scores = model.decision_function(X_test)
print("\nSample Reconstruction Errors:", scores[:10])
