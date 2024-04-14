import json
import pandas as pd
from random import randint

def intermediary():
        intermediaries = {
                'MasterCard', 
                'Visa', 
                'RuPay', 
                'American Express',
                'PayTM', 
                'PhonePay', 
                'MoneyUnion', 
                'Google Pay', 
                'PayPal', 
                'HDFC Bank', 
                'ICICI Bank', 
                'RazorPay', 
                'Unicard', 
                'SBI',
                'Zerodha',
                'HDFC Securities',
                'Groww'
                }
        intermediaries = list(intermediaries)
        return intermediaries[randint(0, len(intermediaries))]

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

val = list()

for x in data:
        row = dict()
        row['Transaction_id'] = 
        row['Payer_id'] = 
        row['Payee_id'] = 
        row['Transaction_amount'] = 
        row['Payment_Intermediary'] = intermediary()
        

with open("mockData.json", 'w') as obj:
        json.dump(val, obj)