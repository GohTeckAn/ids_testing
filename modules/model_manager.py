import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from .hybrid_detector import HybridDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model_dir = 'models'
        self.current_model = None
        self.scaler = None
        self.test_data = None
        self.test_labels = None
        logger.info("Initializing ModelManager...")
        self._ensure_directories()
        self._load_or_create_model()

    def _ensure_directories(self):
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Ensured model directory exists: {self.model_dir}")

    def _create_model(self):
        logger.info("Creating new LSTM model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(1, 78), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        return model

    def _load_or_create_model(self):
        model_path = os.path.join(self.model_dir, 'ids_lstm_model.keras')
        try:
            if os.path.exists(model_path):
                logger.info(f"Found existing model at {model_path}")
                self.current_model = load_model(model_path)
                logger.info("Model loaded successfully")
                logger.info("Model architecture:")
                self.current_model.summary(print_fn=logger.info)
            else:
                logger.info("No existing model found. Creating new model...")
                self.current_model = self._create_model()
                logger.info("New model created")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating new model instead...")
            self.current_model = self._create_model()

    def _train_initial_model(self):
        # Load and prepare initial dataset
        dataset_path = 'data/dataset/initial_dataset.csv'
        if not os.path.exists(dataset_path):
            logger.info("Initial dataset not found. Please run scripts/prepare_dataset.py first.")
            return

        X, y, self.scaler = prepare_data(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.current_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        self.save_model()
        logger.info("Initial model trained and saved.")

    def train_new_model(self, csv_files):
        # Load and preprocess data from CSV files
        data = self._load_csv_data(csv_files)
        X, y = self._preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train new model
        new_model = self._create_model()
        new_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        accuracy = self._evaluate_model(new_model, X_test, y_test)
        
        if accuracy > 0.85:  # Threshold for model acceptance
            self.current_model = new_model
            self.save_model()
            return True
        return False

    def _load_csv_data(self, csv_files):
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _preprocess_data(self, data):
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Implement actual preprocessing logic
        # This should match the preprocessing in prepare_dataset.py
        return np.array([]), np.array([])

    def _evaluate_model(self, model, X_test, y_test):
        scores = model.evaluate(X_test, y_test, verbose=0)
        return scores[1]  # Return accuracy

    def predict(self, features):
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        logger.info(f"Making prediction with input shape: {features.shape}")
        
        # Ensure features are in the correct shape (batch_size, timesteps, features)
        if len(features.shape) == 2:
            features = features.reshape((features.shape[0], 1, -1))
            logger.info(f"Reshaped features to: {features.shape}")
        
        predictions = self.current_model.predict(features)
        logger.info(f"Generated predictions shape: {predictions.shape}")
        return predictions

    def train(self, X, y):
        # Split data into train and test sets
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.train(X_train, y_train)
    
    def predict_hybrid(self, X):
        return self.model.predict(X)
    
    def get_test_data(self):
        """Return the test data for evaluation"""
        return self.X_test, self.y_test

    def save_model(self):
        if self.current_model:
            model_path = os.path.join(self.model_dir, 'ids_lstm_model.keras')
            self.current_model.save(model_path)
            logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        self.current_model = load_model(model_path)

    def load_test_data(self):
        """Load and prepare test data"""
        try:
            # Load test dataset from the test directory
            test_data_path = os.path.join('data', 'test', 'test_data.csv')
            test_data = pd.read_csv(test_data_path)
            
            # Separate features and labels
            features = [
                ' Destination Port',
                'Total Length of Fwd Packets',
                ' Total Length of Bwd Packets',
                ' Fwd Packet Length Max',
                ' Fwd Packet Length Mean',
                'Bwd Packet Length Max',
                ' Bwd Packet Length Mean',
                ' Fwd Header Length',
                ' Bwd Header Length',
                ' Average Packet Size',
                ' Avg Fwd Segment Size',
                ' Avg Bwd Segment Size',
                ' Subflow Fwd Bytes',
                ' Subflow Bwd Bytes',
                'Init_Win_bytes_forward'
            ]
            
            X = test_data[features]
            y = test_data[' Label']
            
            # Scale the features
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            
            # Reshape data for LSTM (samples, timesteps, features)
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            
            self.test_data = X
            self.test_labels = y
            return True
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return False
    
    def evaluate_model(self, model_name):
        """Evaluate a trained model using test data"""
        try:
            if not self.load_test_data():
                raise Exception("Failed to load test data")

            model_path = os.path.join('models', model_name)
            logger.info(f"Loading model from {model_path}")
            
            # Load the model
            model = load_model(model_path)
            
            # Print model summary
            model.summary(print_fn=logger.info)
            
            # Make predictions
            y_pred = model.predict(self.test_data)
            
            # Reshape predictions if needed (in case of multi-dimensional output)
            if len(y_pred.shape) > 2:
                y_pred = y_pred.reshape(y_pred.shape[0], -1)
            
            # Convert predictions to binary (0 or 1)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Ensure predictions are 1-dimensional
            if y_pred_binary.shape[1] == 1:
                y_pred_binary = y_pred_binary.ravel()
            
            # Convert string labels to binary
            y_true_binary = np.array([1 if label == 'DDoS' else 0 for label in self.test_labels])
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Confusion Matrix:\n{cm}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def get_available_models(self):
        """Get list of available model files"""
        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.keras') or file.endswith('.h5'):
                models.append(file)
        return models
