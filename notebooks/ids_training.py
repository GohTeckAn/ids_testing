# %% [markdown]
# # IDS LSTM Model Training
# This notebook demonstrates the training process for our Intrusion Detection System using LSTM.

# %% [markdown]
# ## 1. Import Libraries and Load Dataset
from tensorflow.keras.models import Sequential
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
#matplotlib inline

# %%
# Read the dataset
print("Loading dataset...")
df = pd.read_csv("../data/dataset/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df_original = df.copy()
print(f"Dataset shape: {df.shape}")

# %% [markdown]
# ## 2. Data Preprocessing and Analysis

# %%
# Display data info and initial statistics
print("\nDataset Info:")
print(df.info())

print("\nFeature Statistics:")
print(df.describe().T)

# %%
# Check for missing values and duplicates
print("Missing values:")
print(df.isnull().sum())
print(f"\nNumber of duplicates: {df.duplicated().sum()}")

# %%
# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=' Label', data=df)
plt.title('Traffic Type Distribution')
plt.xticks(rotation=45)
plt.show()

print("\nClass distribution (normalized):")
print(df[' Label'].value_counts(normalize=True))

# %% [markdown]
# ## 3. Feature Selection and Data Cleaning

# %%
# Handle infinite values
print("Checking for infinite values...")
infinite_cols = df.columns[df.isin([np.inf, -np.inf]).any()]
print("Columns with infinite values:", infinite_cols)

# Replace infinite values with NaN and drop rows with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# %%
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

# %% [markdown]
# ## 4. Model Creation and Training

# %%
# Create LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# %% [markdown]
# ## 5. Model Evaluation

# %%
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Final evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Final Test Loss: {loss:.4f}")

# %% [markdown]
# ## 6. Save the Model

# %%
# Save the trained model
model.save('models/test_ddos.keras')
print("Model saved successfully!")
