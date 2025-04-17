import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.hybrid_detector import HybridDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the features we want to use
SELECTED_FEATURES = [
    ' Destination Port',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    ' Bwd Packet Length Mean',
    ' Fwd Header Length',
    ' Average Packet Size',
    ' Avg Fwd Segment Size',
    ' Avg Bwd Segment Size',
    ' Fwd Header Length.1',
    ' Subflow Fwd Bytes',
    ' Subflow Bwd Bytes',
    'Init_Win_bytes_forward'
]

def load_and_preprocess_dataset(file_path, attack_type=None, sample_fraction=0.3):
    """
    Load and preprocess a dataset with specific features
    sample_fraction: fraction of data to use (0 to 1)
    """
    logger.info(f"Loading dataset from {file_path}")
    
    # Read dataset
    df = pd.read_csv(file_path)
    
    # Sample a fraction of the data
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
        logger.info(f"Using {sample_fraction*100:.1f}% of the dataset ({len(df)} samples)")
    
    # Select only the features we want
    X = df[SELECTED_FEATURES].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Create binary labels for general model (0: benign, 1: attack)
    y_binary = (df[' Label'] != 'BENIGN').astype(int)
    
    # Create specific attack labels if needed
    if attack_type:
        # Create multi-class labels for specific attack type
        y_specific = df[' Label'].apply(
            lambda x: x if x == attack_type or x == 'BENIGN' else 'OTHER'
        )
        le = LabelEncoder()
        y_specific = le.fit_transform(y_specific)
        
        return {
            'features': X.values,
            'binary_labels': y_binary.values,
            'specific_labels': y_specific,
            'label_encoder': le
        }
    
    return {
        'features': X.values,
        'binary_labels': y_binary.values
    }

def train_hybrid_detector():
    """
    Train the hybrid detector with multiple datasets
    """
    detector = HybridDetector()
    
    # Load datasets
    datasets_dir = 'data/dataset'
    
    # Load DDoS dataset
    ddos_data = load_and_preprocess_dataset(
        os.path.join(datasets_dir, 'ddos_dataset.csv'),
        attack_type='DDoS'
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        ddos_data['features'], 
        ddos_data['binary_labels'],
        test_size=0.2,
        random_state=42
    )
    _, _, y_train_specific, y_test_specific = train_test_split(
        ddos_data['features'],
        ddos_data['specific_labels'],
        test_size=0.2,
        random_state=42
    )
    
    # Create training and test datasets
    train_data = {
        'features': X_train,
        'binary_labels': y_train_binary,
        'specific_labels': y_train_specific
    }
    
    test_data = {
        'features': X_test,
        'binary_labels': y_test_binary,
        'specific_labels': y_test_specific
    }
    
    # Train general model
    logger.info("Training general anomaly detection model...")
    detector.train_general_model([{'features': X_train, 'labels': y_train_binary}])
    
    # Train specialized models
    logger.info("Training DDoS specialized model...")
    detector.train_specialized_model('ddos', {
        'features': X_train,
        'labels': y_train_specific
    })
    
    logger.info("Training completed successfully!")
    return detector, test_data

def test_detector(detector, test_data):
    """
    Test the trained detector
    """
    logger.info("Testing detector...")
    
    # Make predictions
    results = detector.detect(test_data['features'])
    
    # Calculate accuracy
    correct = 0
    total = len(results)
    
    for i, result in enumerate(results):
        is_attack = test_data['binary_labels'][i] == 1
        if result['is_anomaly'] == is_attack:
            correct += 1
    
    accuracy = correct / total
    logger.info(f"Test accuracy: {accuracy:.2%}")
    
    return accuracy

if __name__ == "__main__":
    try:
        # Train the detector and get test data
        detector, test_data = train_hybrid_detector()
        
        # Test the detector
        accuracy = test_detector(detector, test_data)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save test results
        with open('results/test_results.txt', 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.2%}\n")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
