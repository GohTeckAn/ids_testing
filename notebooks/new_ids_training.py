# %% [markdown]
# # Lightweight LSTM IDS with 15 Features
# Optimized for real-time detection with selected features

# %% [markdown]
# ## 1. Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import joblib

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('ggplot')

# %%
# Read the dataset
print("Loading dataset...")
df = pd.read_csv("../data/dataset/Friday-WorkingHours-Afternoon-DDos_remove_space.pcap_ISCX.csv")
df_original = df.copy()
print(f"Dataset shape: {df.shape}")

# %% [markdown]
# ## 2. Data Preprocessing with Selected Features

# %%
# Select only the 15 important features
selected_features = [
    'Destination Port', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Mean', 'Bwd Packet Length Max',
    'Bwd Packet Length Mean', 'Fwd Header Length',
    'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward'
]

# Check if all selected features exist in the dataframe
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

# Prepare features and target
df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
X = df[selected_features].copy()
y = df['Label'].values

# %%
# Visualize class distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=' Label', data=df)
plt.title('Traffic Type Distribution')
plt.xticks(rotation=45)

# Add percentage annotations
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width()/2
    y = p.get_height() + 0.02*total
    ax.annotate(percentage, (x, y), ha='center')

plt.show()

print("\nClass distribution (normalized):")
print(df[' Label'].value_counts(normalize=True))

# %%
# Clean data - handle infinite and missing values
print("Cleaning data...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)  # Fill missing with feature means

# Remove duplicates if any
X = X.drop_duplicates()
y = y[X.index]
# %%
# Visualize class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x=y)
plt.title('Traffic Type Distribution')
plt.xticks([0, 1], ['BENIGN', 'DDoS'])

# Add percentage annotations
total = len(y)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width()/2
    y_pos = p.get_height() + 0.02*total
    ax.annotate(percentage, (x, y_pos), ha='center')

plt.show()

# %%
# ## 3. Feature Engineering and Data Preparation

# %%
# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Handle class imbalance using oversampling
print("\nClass distribution before oversampling:")
print(pd.Series(y).value_counts())

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_normalized, y)

print("\nClass distribution after oversampling:")
print(pd.Series(y_resampled).value_counts())

# %%
# Reshape for LSTM (samples, timesteps=1, features)
X_reshaped = X_resampled.reshape((X_resampled.shape[0], 1, X_resampled.shape[1]))

# Split data - 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_reshaped, y_resampled,
    test_size=0.4,
    random_state=42,
    stratify=y_resampled
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nData shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# %%
# ## 4. Optimized LSTM Model for Real-Time Detection

# %%
# Create lightweight LSTM model
def create_lightweight_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    return model

model = create_lightweight_lstm((X_train.shape[1], X_train.shape[2]))
model.summary()

# %%
# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_resampled)

# Reshape data for LSTM (samples, timesteps, features)
timesteps = 1
X_reshaped = X_normalized.reshape((X_normalized.shape[0], timesteps, X_normalized.shape[1]))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_resampled, 
    test_size=0.2, 
    random_state=42,
    stratify=y_resampled
)

# Create validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nData shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [markdown]
# ## 4. Enhanced Model Architecture and Training

# %%
# Calculate class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\nClass weights:", class_weights)

# %%
# Create enhanced LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# %%
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\nClass weights:", class_weights)

# %%
# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=8,
    mode='max',
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_lightweight_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weights,
    verbose=1
)

# %%
# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot AUC
    plt.subplot(1, 3, 2)
    plt.plot(history.history['auc'], label='Training')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    # Plot precision-recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['recall'], label='Recall')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# %%
# Load best model
model.load_weights('best_lightweight_model.keras')

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")
print(f"Test AUC: {test_results[4]:.4f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['BENIGN', 'DDoS']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BENIGN', 'DDoS'],
            yticklabels=['BENIGN', 'DDoS'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 6. Save Model for Real-Time Detection

# %%
# Save the final model and scaler
model.save('models/lightweight_ddos_model.keras')
joblib.dump(scaler, 'models/lightweight_scaler.pkl')

print("\nModel and scaler saved successfully for real-time deployment!")

# %%
# 7. Test with Self Produce CSV Data

# %%
# Define the paths at the top level of your script (not inside any function)
csv_path = 'data/test/traffic_20250328_003416(testsample1).csv'
MODEL_PATH = 'models/lightweight_ddos_model.keras'
SCALER_PATH = 'models/lightweight_scaler.pkl'