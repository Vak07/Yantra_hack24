import pandas as pd
import numpy as np
from collections import Counter


account_activity = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/account_activity.csv')
customer_data = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/customer_data.csv')
fraud_indicators = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/fraud_indicators.csv')
suspicious_activity = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/suspicious_activity.csv')
merchant_data = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/merchant_data.csv')
transaction_category_labels = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/transaction_category_labels.csv')
amount_data = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/amount_data.csv')
anomaly_scores = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/anomaly_scores.csv')
transaction_metadata = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/transaction_metadata.csv')
transaction_records = pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/transaction_records.csv')

transaction_data = pd.merge(transaction_records, transaction_metadata, on='TransactionID')
transaction_data = pd.merge(transaction_data, amount_data, on='TransactionID')
transaction_data = pd.merge(transaction_data, merchant_data, on='MerchantID')
transaction_data = pd.merge(transaction_data, transaction_category_labels, on='TransactionID')
transaction_data = pd.merge(transaction_data, fraud_indicators, on='TransactionID', how='left')
transaction_data = pd.merge(transaction_data, suspicious_activity, on='CustomerID', how='left')
transaction_data = pd.merge(transaction_data, customer_data, on='CustomerID')

transaction_data.to_csv('C:/Yantra Central Backend/fraud_backend/datasets/combined_data.csv')
# transaction_data = transaction_data.drop(['Timestamp', 'Name', 'Address'], axis=1)
# transaction_data['SuspiciousFlag'].fillna(0, inplace=True)
# transaction_data['FraudIndicator'].fillna(0, inplace=True)
# transaction_data['Category'] = pd.Categorical(transaction_data['Category'])
# transaction_data['Category'] = transaction_data['Category'].cat.codes
# if 'MerchantName' in transaction_data.columns:
#     # Encode 'MerchantName' using one-hot encoding
#     transaction_data = pd.get_dummies(transaction_data, columns=['MerchantName'], prefix='Merchant')

# # Check if 'Location' is present before one-hot encoding
# if 'Location' in transaction_data.columns:
#     transaction_data = pd.get_dummies(transaction_data, columns=['Location'])#

# X = transaction_data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)
# y = transaction_data['FraudIndicator']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# input_dim = X_train.shape[1]

# input_layer = Input(shape=(input_dim,))
# encoder = Dense(32, activation='relu')(input_layer)
# encoder = Dense(16, activation='relu')(encoder)
# decoder = Dense(32, activation='relu')(encoder)
# decoder = Dense(input_dim, activation='sigmoid')(decoder)

# autoencoder = Model(inputs=input_layer, outputs=decoder)

# # Compile Autoencoder
# autoencoder.compile(optimizer='adam', loss='mse')
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train the model
# autoencoder.fit(X_train, X_train,
#                 epochs=20,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(X_test, X_test),
#                 callbacks=[early_stopping])

# decoded_data = autoencoder.predict(X_test)
# rmse = np.sqrt(np.mean(np.square(X_test - decoded_data), axis=1))


# transaction_data['RMSE'] = np.nan
# transaction_data.loc[y_test.index, 'RMSE'] = rmse
# threshold = transaction_data['RMSE'].quantile(0.965)


# transaction_data['PredictedFraud'] = 0
# transaction_data.loc[transaction_data['RMSE'] > threshold, 'PredictedFraud'] = 1
# merchant_trust_scores = transaction_data.groupby('MerchantID')['RMSE'].mean().reset_index()
# merchant_trust_scores.columns = ['MerchantID', 'MerchantTrustScore']

# customer_trust_scores = transaction_data.groupby('CustomerID')['RMSE'].mean().reset_index()
# customer_trust_scores.columns = ['CustomerID', 'CustomerTrustScore']

# transaction_trust_scores = transaction_data.groupby('TransactionID')['RMSE'].mean().reset_index()
# transaction_trust_scores.columns = ['TransactionID', 'TransactionTrustScore']
# X = transaction_data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)

# # Standardize data including the new features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train/test split with the updated feature set
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# input_dim = X_train.shape[1]

# input_layer = Input(shape=(input_dim,))
# encoder = Dense(32, activation='relu')(input_layer)  # Adjusted the number of neurons for better representation
# encoder = Dense(16, activation='relu')(encoder)
# decoder = Dense(32, activation='relu')(encoder)
# decoder = Dense(input_dim, activation='sigmoid')(decoder)

# autoencoder = Model(inputs=input_layer, outputs=decoder)
# autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.fit(X_train, X_train,
#                 epochs=20,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(X_test, X_test),
#                 callbacks=[early_stopping])

# # Step 14: Make Predictions and Calculate RMSE
# # Make predictions on the test set
# decoded_data = autoencoder.predict(X_test)

# # Calculate RMSE for each transaction
# rmse = np.sqrt(np.mean(np.square(X_test - decoded_data), axis=1))

# # Add RMSE to the dataframe
# transaction_data['RMSE'] = np.nan
# transaction_data.loc[y_test.index, 'RMSE'] = rmse
# suspicious_threshold = 0.8  # Adjust the threshold based on your preference

# # Flag transactions as suspicious based on the threshold error
# transaction_data['SuspiciousFlag'] = 0
# transaction_data.loc[transaction_data['RMSE'] > suspicious_threshold, 'SuspiciousFlag'] = 1

# # Update trust scores based on the suspicious flag and fraud flag
# def update_trust_score(trust_score, suspicious_flag, fraud_flag):
#     if fraud_flag == 1:
#         trust_score = max(0, trust_score - 0.2)  # Decrease trust score more for flagged fraudulent transactions
#     else:
#         if suspicious_flag == 1:
#             trust_score = max(0, trust_score - 0.1)  # Decrease trust score for suspicious transactions
#         else:
#             trust_score = min(1, trust_score + 0.1)  # Increase trust score for non-suspicious transactions
#     return trust_score

# transaction_data['MerchantTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['MerchantTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)
# transaction_data['CustomerTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['CustomerTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)
# transaction_data['TransactionTrustScore'] = transaction_data.apply(lambda row: update_trust_score(row['TransactionTrustScore'], row['SuspiciousFlag'], row['PredictedFraud']), axis=1)
