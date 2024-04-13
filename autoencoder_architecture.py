import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

class AutoencoderModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.threshold = None
        self.transaction_data = None

    def build_model(self):
        # Define Autoencoder architecture
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(16, activation='LeakyReLU')(input_layer)
        encoder = Dense(8, activation='LeakyReLU')(encoder)
        decoder = Dense(16, activation='LeakyReLU')(encoder)
        decoder = Dense(self.input_dim, activation='LeakyReLU')(decoder)

        self.autoencoder = Model(inputs=input_layer, outputs=decoder)

    def compile_model(self, optimizer='SGD', loss='mse'):
        # Compile Autoencoder
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def train_model(self, X_train, X_test, epochs=20, batch_size=32):
        # Set up early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test),
                             callbacks=[early_stopping])

    def predict_and_calculate_rmse(self, X_test):
        # Make predictions on the test set
        decoded_data = self.autoencoder.predict(X_test)
        rmse = np.sqrt(np.mean(np.square(X_test - decoded_data), axis=1))

        # Add RMSE to the dataframe
        self.transaction_data['RMSE'] = np.nan
        self.transaction_data.loc[self.y_test.index, 'RMSE'] = rmse

        # Determine a threshold for anomaly detection
        self.threshold = self.transaction_data['RMSE'].quantile(
