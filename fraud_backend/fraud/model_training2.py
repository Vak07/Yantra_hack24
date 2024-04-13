import pandas as pd
import numpy as np
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mse
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package='Custom', name='mse')
def custom_mse(y_true, y_pred):
    return mse(y_true, y_pred)
model1 = tf.keras.models.load_model('C:/Yantra Central Backend/fraud_backend/fraud/model1.h5',custom_objects={'custom_mse': custom_mse})
model1.compile(optimizer='adam', loss='mse')
transaction_data=pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/combined_data.csv')
transaction_data = transaction_data.drop(['Timestamp', 'Name', 'Address'], axis=1)
transaction_data['SuspiciousFlag'].fillna(0, inplace=True)
transaction_data['FraudIndicator'].fillna(0, inplace=True)
transaction_data['Category'] = pd.Categorical(transaction_data['Category'])
transaction_data['Category'] = transaction_data['Category'].cat.codes
if 'MerchantName' in transaction_data.columns:
    # Encode 'MerchantName' using one-hot encoding
    transaction_data = pd.get_dummies(transaction_data, columns=['MerchantName'], prefix='Merchant')

# Check if 'Location' is present before one-hot encoding
if 'Location' in transaction_data.columns:
    transaction_data = pd.get_dummies(transaction_data, columns=['Location'])#

X = transaction_data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)
y = transaction_data['FraudIndicator']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]

decoded_data = model1.predict(X_test)
rmse = np.sqrt(np.mean(np.square(X_test - decoded_data), axis=1))


transaction_data['RMSE'] = np.nan
transaction_data.loc[y_test.index, 'RMSE'] = rmse
threshold = transaction_data['RMSE'].quantile(0.965)


transaction_data['PredictedFraud'] = 0
transaction_data.loc[transaction_data['RMSE'] > threshold, 'PredictedFraud'] = 1
merchant_trust_scores = transaction_data.groupby('MerchantID')['RMSE'].mean().reset_index()
merchant_trust_scores.columns = ['MerchantID', 'MerchantTrustScore']

customer_trust_scores = transaction_data.groupby('CustomerID')['RMSE'].mean().reset_index()
customer_trust_scores.columns = ['CustomerID', 'CustomerTrustScore']

transaction_trust_scores = transaction_data.groupby('TransactionID')['RMSE'].mean().reset_index()
transaction_trust_scores.columns = ['TransactionID', 'TransactionTrustScore']
X = transaction_data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)

# Standardize data including the new features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train/test split with the updated feature set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu')(input_layer)  # Adjusted the number of neurons for better representation
encoder = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss=custom_mse)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[early_stopping])
autoencoder.save('C:/Yantra Central Backend/fraud_backend/fraud/model2.h5')

