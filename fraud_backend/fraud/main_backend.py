import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mse
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package='Custom', name='mse')
def custom_mse(y_true, y_pred):
    return mse(y_true, y_pred)
transaction_data=pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/combined_data.csv')
autoencoder = tf.keras.models.load_model('C:/Yantra Central Backend/fraud_backend/fraud/model2.h5',custom_objects={'custom_mse': custom_mse})

X = transaction_data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)
y = transaction_data['FraudIndicator']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
decoded_data = autoencoder.predict(X_test)

# Calculate RMSE for each transaction
rmse = np.sqrt(np.mean(np.square(X_test - decoded_data), axis=1))

# Add RMSE to the dataframe
transaction_data['RMSE'] = np.nan
transaction_data.loc[y_test.index, 'RMSE'] = rmse
suspicious_threshold = 0.8  # Adjust the threshold based on your preference

# Flag transactions as suspicious based on the threshold error
transaction_data['SuspiciousFlag'] = 0
transaction_data.loc[transaction_data['RMSE'] > suspicious_threshold, 'SuspiciousFlag'] = 1

# Update trust scores based on the suspicious flag and fraud flag
def update_trust_score(trust_score, suspicious_flag, fraud_flag):
    if fraud_flag == 1:
        trust_score = max(0, trust_score - 0.2)  # Decrease trust score more for flagged fraudulent transactions
    else:
        if suspicious_flag == 1:
            trust_score = max(0, trust_score - 0.1)  # Decrease trust score for suspicious transactions
        else:
            trust_score = min(1, trust_score + 0.1)  # Increase trust score for non-suspicious transactions
    return trust_score

transaction_data['MerchantTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['MerchantTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)
transaction_data['CustomerTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['CustomerTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)
transaction_data['TransactionTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['TransactionTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)


