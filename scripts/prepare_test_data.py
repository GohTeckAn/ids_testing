import pandas as pd
import os
from sklearn.model_selection import train_test_split

# List of features we want to keep (matching exact column names from dataset)
SELECTED_FEATURES = [
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
    'Init_Win_bytes_forward',
    ' Label'
]

def prepare_test_data():
    # Read the original dataset
    data_path = os.path.join('data', 'dataset', 'ddos_dataset.csv')
    df = pd.read_csv(data_path)
    
    # Print original dataset info
    print("\nOriginal Dataset Info:")
    print(df.info())
    
    # Print all column names for debugging
    print("\nActual column names in dataset:")
    for col in df.columns:
        print(f"'{col}'")
    
    # Verify all required features exist
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_features:
        print("\nWarning: Missing features:", missing_features)
        return
    
    # Select only the features we need
    df_selected = df[SELECTED_FEATURES]
    
    # Split into train and test sets (80-20 split)
    train_data, test_data = train_test_split(df_selected, test_size=0.2, random_state=42)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join('data', 'test'), exist_ok=True)
    
    # Save test data
    test_data_path = os.path.join('data', 'test', 'test_data.csv')
    test_data.to_csv(test_data_path, index=False)
    
    print(f"\nTest data saved to: {test_data_path}")
    print("\nTest Dataset Info:")
    print(test_data.info())
    
    # Print sample of test data
    print("\nSample of test data:")
    print(test_data.head())
    
    return test_data_path

if __name__ == "__main__":
    prepare_test_data()
