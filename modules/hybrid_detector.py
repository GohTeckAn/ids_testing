import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class HybridDetector:
    def __init__(self):
        self.models_dir = 'models'
        self.general_model = None
        self.specialized_models = {}
        self.scalers = {}
        self._ensure_directories()
        self._load_or_create_models()

    def _ensure_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)

    def _create_general_model(self, input_shape):
        """Creates a general anomaly detection model"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _create_specialized_model(self, input_shape, num_classes):
        """Creates a specialized attack classification model"""
        model = Sequential([
            LSTM(32, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def train_general_model(self, datasets):
        """Train the general anomaly detection model on all datasets"""
        combined_data = []
        combined_labels = []
        
        # Combine all datasets
        for dataset in datasets:
            X = dataset['features']
            y = dataset['labels']  # Binary: 0 for normal, 1 for any attack
            combined_data.append(X)
            combined_labels.append(y)
        
        X = np.concatenate(combined_data)
        y = np.concatenate(combined_labels)
        
        # Create and fit scaler
        self.scalers['general'] = StandardScaler()
        X_scaled = self.scalers['general'].fit_transform(X)
        
        # Reshape for LSTM
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Create and train model
        self.general_model = self._create_general_model((1, X_scaled.shape[2]))
        self.general_model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)
        
        # Save model
        self.general_model.save(os.path.join(self.models_dir, 'general_model.keras'))

    def train_specialized_model(self, attack_type, dataset):
        """Train a specialized model for a specific attack type"""
        X = dataset['features']
        y = dataset['labels']  # Multi-class for specific attack types
        
        # Create and fit scaler
        self.scalers[attack_type] = StandardScaler()
        X_scaled = self.scalers[attack_type].fit_transform(X)
        
        # Reshape for LSTM
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Create and train model
        num_classes = len(np.unique(y))
        self.specialized_models[attack_type] = self._create_specialized_model(
            (1, X_scaled.shape[2]), num_classes
        )
        
        # Convert labels to one-hot encoding
        y_onehot = tf.keras.utils.to_categorical(y)
        
        # Train model
        self.specialized_models[attack_type].fit(
            X_scaled, y_onehot, epochs=10, batch_size=32, validation_split=0.2
        )
        
        # Save model
        self.specialized_models[attack_type].save(
            os.path.join(self.models_dir, f'{attack_type}_model.keras')
        )

    def detect(self, traffic_features):
        """
        Perform two-stage detection:
        1. General anomaly detection
        2. Specific attack classification if anomaly detected
        """
        # Scale features using general scaler
        X_scaled = self.scalers['general'].transform(traffic_features)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # First stage: general anomaly detection
        anomaly_scores = self.general_model.predict(X_scaled)
        
        results = []
        for i, score in enumerate(anomaly_scores):
            if score > 0.5:  # Anomaly detected
                # Second stage: specific attack classification
                attack_predictions = {}
                for attack_type, model in self.specialized_models.items():
                    # Scale features using attack-specific scaler
                    X_attack = self.scalers[attack_type].transform(traffic_features[i:i+1])
                    X_attack = X_attack.reshape((1, 1, X_attack.shape[1]))
                    
                    # Get prediction probabilities
                    probs = model.predict(X_attack)[0]
                    attack_predictions[attack_type] = probs
                
                results.append({
                    'is_anomaly': True,
                    'anomaly_score': float(score),
                    'attack_predictions': attack_predictions
                })
            else:
                results.append({
                    'is_anomaly': False,
                    'anomaly_score': float(score),
                    'attack_predictions': None
                })
        
        return results

    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        general_model_path = os.path.join(self.models_dir, 'general_model.keras')
        if os.path.exists(general_model_path):
            self.general_model = load_model(general_model_path)
            logger.info("Loaded general model")
        
        # Load specialized models
        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('_model.keras') and not model_file.startswith('general'):
                attack_type = model_file.split('_')[0]
                model_path = os.path.join(self.models_dir, model_file)
                self.specialized_models[attack_type] = load_model(model_path)
                logger.info(f"Loaded specialized model for {attack_type}")
