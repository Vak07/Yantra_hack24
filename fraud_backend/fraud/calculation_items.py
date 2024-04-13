import pandas as pd 
transaction_data=pd.read_csv('C:/Yantra Central Backend/fraud_backend/datasets/combined_data.csv')

def total_transactions():
    total=0
    sum=transaction_data['TransactionAmount'].sum()
    merchants=transaction_data.shape[0]
    print(sum,merchants)
    return sum,merchants

def suspicion_data():
    suspicious_data=transaction_data[transaction_data['SuspiciousFlag']==1]
    # print(suspicious_data)
    extracted_data=transaction_data[['MerchantID','TransactionAmount','Timestamp']]
    data_dict={}
    for index, row in extracted_data.iterrows():
     print("Entry")
     print(f"MerchantID: {row['MerchantID']}")
     print(f"TransactionAmount: {row['TransactionAmount']}")
     print(f"Timestamp: {row['Timestamp']}")
     data_dict[index] = {
        'MerchantID': row['MerchantID'],
        'TransactionAmount': row['TransactionAmount'],
        'Timestamp': row['Timestamp']
     }
     return data_dict
    
# total_transactions()