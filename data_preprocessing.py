import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def drop_columns(self, columns):
        # Drop unnecessary columns
        self.data = self.data.drop(columns=columns, axis=1)

    def handle_missing_values(self, fill_value=0):
        # Handle missing values
        self.data['SuspiciousFlag'].fillna(fill_value, inplace=True)
        self.data['FraudIndicator'].fillna(fill_value, inplace=True)

    def encode_categorical(self, columns):
        # Convert categorical variables to numerical
        for col in columns:
            self.data[col] = pd.Categorical(self.data[col])
            self.data[col] = self.data[col].cat.codes

    def preprocess_data(self, test_size=0.2, random_state=42):
        # Split into features and target
        X = self.data.drop(['TransactionID', 'FraudIndicator', 'SuspiciousFlag'], axis=1)
        y = self.data['FraudIndicator']

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
