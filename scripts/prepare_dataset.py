import os
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import numpy as np

def download_dataset():
    """
    Download CICIDS2017 dataset (using a smaller portion for demonstration)
    In practice, you should download the full dataset from:
    https://www.unb.ca/cic/datasets/ids-2017.html
    """
    data_dir = 'data/dataset'
    os.makedirs(data_dir, exist_ok=True)
    
    # For demonstration, we'll use a smaller sample
    # In practice, download the full dataset
    sample_data = {
        'Timestamp': pd.date_range(start='2017-07-01', periods=1000, freq='S'),
        'Source IP': ['192.168.1.' + str(i % 255) for i in range(1000)],
        'Destination IP': ['10.0.0.' + str(i % 255) for i in range(1000)],
        'Protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 1000),
        'Length': np.random.randint(64, 1500, 1000),
        'Source Port': np.random.randint(1024, 65535, 1000),
        'Destination Port': np.random.randint(1, 1024, 1000),
        'Flags': np.random.choice(['S', 'SA', 'A', 'R', 'F'], 1000),
        'Label': np.random.choice(['NORMAL', 'DOS', 'PortScan', 'APT'], 1000, p=[0.7, 0.1, 0.1, 0.1])
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(os.path.join(data_dir, 'initial_dataset.csv'), index=False)
    return os.path.join(data_dir, 'initial_dataset.csv')

def prepare_data(csv_path):
    """Prepare data for LSTM model"""
    df = pd.read_csv(csv_path)
    
    # Convert categorical variables
    df['Protocol'] = pd.Categorical(df['Protocol']).codes
    df['Flags'] = pd.Categorical(df['Flags']).codes
    
    # Convert IP addresses to numerical values
    df['Source IP'] = df['Source IP'].apply(lambda x: int(''.join([f"{int(i):03d}" for i in x.split('.')])))
    df['Destination IP'] = df['Destination IP'].apply(lambda x: int(''.join([f"{int(i):03d}" for i in x.split('.')])))
    
    # Create features and labels
    features = ['Source IP', 'Destination IP', 'Protocol', 'Length', 
               'Source Port', 'Destination Port', 'Flags']
    
    X = df[features].values
    y = pd.get_dummies(df['Label']).values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for LSTM (samples, time steps, features)
    X = X.reshape(-1, 1, len(features))
    
    return X, y, scaler

if __name__ == '__main__':
    dataset_path = download_dataset()
    X, y, scaler = prepare_data(dataset_path)
    print(f"Dataset prepared: {X.shape}, {y.shape}")
