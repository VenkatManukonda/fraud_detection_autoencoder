# Fraud Detection using AutoEncoder

This project implements an unsupervised deep learning approach to detect fraudulent transactions.

## Approach
Instead of learning fraud patterns explicitly, the model learns normal transaction behavior. Transactions that deviate from this pattern produce higher reconstruction error and are flagged as anomalies.

## Tools Used
- PyOD
- Scikit-learn
- Pandas

## How to Run
pip install -r requirements.txt
python fraud_detection_autoencoder.py
