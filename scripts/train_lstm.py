import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv("data/dataset/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df_original = df.copy()
    print(f"Dataset shape: {df.shape}")

    # Check for missing values and duplicates
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")

    # Handle infinite values
    print("Checking for infinite values...")
    infinite_cols = df.columns[df.isin([np.inf, -np.inf]).any()]
    print("Columns with infinite values:", infinite_cols)

    # Replace infinite values with NaN and drop rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Prepare features and target
    print("Preparing features and target...")
    # Map labels to binary values
    df[' Label'] = df[' Label'].map({'BENIGN': 0, 'DDoS': 1})

    # Select relevant features (excluding Label)
    selected_features = [col for col in df.columns if col != ' Label']
    X = df[selected_features].values
    y = df[' Label'].values

    # Reshape data for LSTM (samples, timesteps, features)
    timesteps = 1
    X = X.reshape((X.shape[0], timesteps, X.shape[1]))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Create LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\nModel Summary:")
    model.summary()

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # Final evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    # Save the model
    import os
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'ids_lstm_model.keras')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved successfully at: {model_path}")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    main()
