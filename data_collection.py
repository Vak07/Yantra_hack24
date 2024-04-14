import pandas as pd

class DataCollection:
    def __init__(self):
        # Initialize dataframes or connections
        pass

    def load_data(self):
        # Load datasets
        account_activity = pd.read_csv('account_activity.csv')
        customer_data = pd.read_csv('customer_data.csv')
        fraud_indicators = pd.read_csv('fraud_indicators.csv')
        suspicious_activity = pd.read_csv('suspicious_activity.csv')
        merchant_data = pd.read_csv('merchant_data.csv')
        transaction_category_labels = pd.read_csv('transaction_category_labels.csv')
        amount_data = pd.read_csv('amount_data.csv')
        anomaly_scores = pd.read_csv('anomaly_scores.csv')
        transaction_metadata = pd.read_csv('transaction_metadata.csv')
        transaction_records = pd.read_csv('transaction_records.csv')

        return account_activity, customer_data, fraud_indicators, suspicious_activity, merchant_data, \
               transaction_category_labels, amount_data, anomaly_scores, transaction_metadata, transaction_records

    def merge_data(self, *dataframes):
        # Merge relevant datasets
        merged_data = pd.merge(dataframes[9], dataframes[8], on='TransactionID')
        for df in dataframes[7::-1]:
            merged_data = pd.merge(merged_data, df, on=('TransactionID', 'MerchantID'), how='left')
        return merged_data
