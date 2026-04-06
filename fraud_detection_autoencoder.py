from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing
X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize AutoEncoder using current PyOD API
model = AutoEncoder(contamination=0.01, verbose=1)

# Fit the model
model.fit(X_train)

# Predictions: 0 = normal, 1 = anomaly
y_pred = model.predict(X_test)

# Evaluate results
fraud_detected = y_pred.sum()
actual_fraud = y_test.sum()

print("\nResults:")
print("Actual Fraud Cases:", actual_fraud)
print("Detected Fraud Cases:", fraud_detected)

# Reconstruction error (optional)
reconstruction_error = model.decision_function(X_test)
print("\nSample Reconstruction Errors:", reconstruction_error[:10])
