import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def generate_training_data(n_samples=10000):
    """Generate synthetic training data"""
    # Normal traffic
    normal_traffic = np.random.normal(loc=0.3, scale=0.2, size=(n_samples//2, 78))
    normal_labels = np.zeros(n_samples//2)
    
    # Attack traffic (higher values and more variance)
    attack_traffic = np.random.normal(loc=0.7, scale=0.3, size=(n_samples//2, 78))
    attack_labels = np.ones(n_samples//2)
    
    # Combine data
    X = np.vstack([normal_traffic, attack_traffic])
    y = np.concatenate([normal_labels, attack_labels])
    
    return X, y

def main():
    print("Generating synthetic training data...")
    X, y = generate_training_data()
    
    # Reshape data for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nCreating LSTM model...")
    model = Sequential([
        LSTM(64, input_shape=(1, 78), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Save the model
    model_path = os.path.join('models', 'ids_lstm_model.keras')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
