import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pyod.models.auto_encoder import AutoEncoder


# -------------------------------
# LOAD DATA (MUST BE FIRST)
# -------------------------------
data = pd.read_csv("creditcard.csv")
print("Dataset shape:", data.shape)


# -------------------------------
# FEATURES + LABEL
# -------------------------------
X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]


# -------------------------------
# NORMALIZE
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# AUTOENCODER MODEL
# -------------------------------
model = AutoEncoder(contamination=0.01, verbose=1)


# -------------------------------
# TRAIN
# -------------------------------
model.fit(X_train)


# -------------------------------
# PREDICT
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# RESULTS
# -------------------------------
print("\nResults:")
print("Actual Fraud Cases:", np.sum(y_test))
print("Detected Fraud Cases:", np.sum(y_pred))


# -------------------------------
# RECONSTRUCTION ERROR
# -------------------------------
scores = model.decision_function(X_test)
print("\nSample Reconstruction Errors:", scores[:10])
