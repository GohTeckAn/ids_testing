import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
from datetime import datetime
import time

# Create directories
os.makedirs('data/logs', exist_ok=True)
os.makedirs('data/results', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DDoSDetector:
    def __init__(self, model_path='models/ddos_model.keras'):
        try:
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Define feature order expected by model
            self.features =  [
            ' Destination Port', ' Total Length of Fwd Packets',
            ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
            ' Fwd Packet Length Mean', 'Bwd Packet Length Max',
            ' Bwd Packet Length Mean', ' Fwd Header Length',
            ' Average Packet Size', ' Avg Fwd Segment Size',
            ' Avg Bwd Segment Size', ' Fwd Header Length.1',
            ' Subflow Fwd Bytes', ' Subflow Bwd Bytes',
            'Init_Win_bytes_forward'
            ]
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def preprocess_data(self, df):
        """Preprocess data for model input"""
        try:
            # Ensure all features are present
            missing_features = set(self.features) - set(df.columns)
            if missing_features:
                logger.error(f"Missing features in data: {missing_features}")
                return None
            
            # Select and order features
            X = df[self.features].copy()
            
            # Handle missing values
            if X.isna().any().any():
                logger.warning("Found missing values in data, filling with 0")
                X = X.fillna(0)
            
            # Scale features (add your scaling logic here if needed)
            
            return X
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}", exc_info=True)
            return None

    def predict(self, traffic_file):
        """Predict DDoS attacks in traffic data"""
        try:
            # Read traffic data
            logger.info(f"Reading traffic data from {traffic_file}")
            df = pd.read_csv(traffic_file)
            
            # Preprocess data
            X = self.preprocess_data(df)
            if X is None:
                return None
            
            # Make predictions
            logger.info("Making predictions")
            predictions = self.model.predict(X)
            
            # Add predictions to dataframe
            df[' Label'] = ['DDoS' if p > 0.5 else 'BENIGN' for p in predictions]
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join('data/results', f'detection_{timestamp}.csv')
            df.to_csv(output_file, index=False)
            
            # Log summary
            ddos_count = (df[' Label'] == 'DDoS').sum()
            benign_count = (df[' Label'] == 'BENIGN').sum()
            logger.info(f"Detection results: {ddos_count} DDoS, {benign_count} BENIGN")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return None

def analyze_traffic(traffic_file):
    """Analyze traffic file for DDoS attacks"""
    try:
        # Create detector
        detector = DDoSDetector()
        
        # Analyze traffic
        result_file = detector.predict(traffic_file)
        
        if result_file:
            logger.info(f"Analysis complete. Results saved to: {result_file}")
            return result_file
        else:
            logger.error("Analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Error analyzing traffic: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Check if traffic file is provided
    import sys
    if len(sys.argv) > 1:
        traffic_file = sys.argv[1]
    else:
        # Use most recent traffic file
        traffic_dir = 'data/traffic'
        files = [f for f in os.listdir(traffic_dir) if f.startswith('traffic_')]
        if not files:
            logger.error("No traffic files found")
            sys.exit(1)
        traffic_file = os.path.join(traffic_dir, sorted(files)[-1])
    
    # Analyze traffic
    analyze_traffic(traffic_file)
