class TrustScoreMatrix:
    def __init__(self, data):
        self.data = data

    def calculate_trust_scores(self):
       merchant_trust_scores = transaction_data.groupby('MerchantID')['RMSE'].mean().reset_index()
       merchant_trust_scores.columns = ['MerchantID', 'MerchantTrustScore']

       customer_trust_scores = transaction_data.groupby('CustomerID')['RMSE'].mean().reset_index()
       customer_trust_scores.columns = ['CustomerID', 'CustomerTrustScore']

       transaction_trust_scores = transaction_data.groupby('TransactionID')['RMSE'].mean().reset_index()
       transaction_trust_scores.columns = ['TransactionID', 'TransactionTrustScore']
     return merchant_trust_scores, customer_trust_scores, transaction_trust_scores
    while(true):
        transaction_data = pd.merge(transaction_data, merchant_trust_scores, on='MerchantID', how='left')
        transaction_data = pd.merge(transaction_data, customer_trust_scores, on='CustomerID', how='left')
        transaction_data = pd.merge(transaction_data, transaction_trust_scores, on='TransactionID', how='left')
    
    trust_threshold = 0.5  # Adjust the threshold based on your preference


    transaction_data['PredictedFraud'] = 0
     transaction_data.loc[(transaction_data['RMSE'] > threshold) & (transaction_data['MerchantTrustScore'] < trust_threshold) & 
                      (transaction_data['CustomerTrustScore'] < trust_threshold) & 
                      (transaction_data['TransactionTrustScore'] < trust_threshold), 'PredictedFraud'] = 1
    average_transaction_amount = transaction_data.groupby('CustomerID')['TransactionAmount'].mean().reset_index()
average_transaction_amount.columns = ['CustomerID', 'AverageTransactionAmount']
# Calculate the frequency of transactions for each customer
transaction_frequency = transaction_data.groupby('CustomerID').size().reset_index(name='TransactionFrequency')

# Merge additional features back into the main dataframe
transaction_data = pd.merge(transaction_data, average_transaction_amount, on='CustomerID', how='left')
transaction_data = pd.merge(transaction_data, transaction_frequency, on='CustomerID', how='left')


transaction_data['AverageTransactionAmount'].fillna(0, inplace=True)
transaction_data['TransactionFrequency'].fillna(0, inplace=True)
