import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ResultAnalysis:
    def __init__(self, transaction_data, y_test):
        self.transaction_data = transaction_data
        self.y_test = y_test

    def flag_transactions(self, threshold):
        # Flag transactions as fraudulent based on the threshold
        self.transaction_data['PredictedFraud'] = 0
        self.transaction_data.loc[self.transaction_data['RMSE'] > threshold, 'PredictedFraud'] = 1

    def analyze_results(self):
        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, self.transaction_data.loc[self.y_test.index, 'PredictedFraud'])
        print(f'Accuracy: {accuracy:.4f}')

        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.transaction_data.loc[self.y_test.index, 'PredictedFraud'])
        print('Confusion Matrix:')
        print(conf_matrix)

        # Classification report
        class_report = classification_report(self.y_test, self.transaction_data.loc[self.y_test.index, 'PredictedFraud'])
        print('Classification Report:')
        print(class_report)
