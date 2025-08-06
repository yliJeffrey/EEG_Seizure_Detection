import mne
import numpy as np
import os
import pickle
import re

# Define the path to your pickle file
seizure_times_path = 'seizure_times.pkl'

# load the pickle file to get the seizure time (dictionary: key-filename; value-list of seizure times)
def load_seizure_time(file_path):
    # 'rb' means 'read binary' mode
    with open(file_path, 'rb') as file:
        # Load the data from the file
        seizure_times = pickle.load(file)
    return seizure_times

def load_edf(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']  # e.g. 256Hz
    # print(f'{filepath}.shape = {data.shape}\tsfreq: {sfreq}')
    return data, sfreq

def slice_data(data, sfreq, delta, interval, offset=5, seq_len=60):
    n_channels, total_len = data.shape
    total_seconds = total_len // sfreq
    n_datas = total_seconds - seq_len
    slice_data = np.zeros((n_datas, seq_len, n_channels, sfreq))
    label = np.zeros(n_datas)
    count = 0
    for i in range(n_datas):
        for j in range(seq_len):
            slice_data[i][j] = data[:, (i + j) * sfreq : (i + j + 1) * sfreq]
            # if  ((i * seq_len) + j > delta + offset) and ((i * seq_len) + j < delta + interval - offset):
        if (i + seq_len >= delta + offset) and (i + seq_len <= delta + interval - offset):            
            label[i] = 1
            count += 1
    return slice_data, label

# seizure_info is a dictionary: key-filename, value-list of seizure times
def generate_dataset(seizure_info, data_dir, delta=100, offset=10, seq_len=60):
    datas, labels = [], []
    for file_name, seizure in seizure_info.items():
        data, sfreq = load_edf(data_dir + file_name)
        if data.shape[0] != 23:
            continue
        interval = seizure[0][1] - seizure[0][0]
        if interval <= 14:
            continue
        # delta = interval // 2
        valid_seconds = interval + delta * 2
        valid_data = data[:, (seizure[0][0] - delta) * int(sfreq) : (seizure[0][1] + delta) * int(sfreq)]
        slice_datas, label = slice_data(valid_data, int(sfreq), delta, interval, offset, seq_len)        
       
        labels.append(label)
        datas.append(slice_datas)
        print(f'\n{file_name}\tvalid_data.shape = {valid_data.shape}\tslice_data.shape = {slice_datas.shape}\tlabel.shape = {label.shape}\n')

    merged_datas = np.concatenate(datas, axis=0)   
    merged_labels = np.concatenate(labels, axis=0)
    return merged_datas, merged_labels


import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

# Convolutional Neural Network
def create_eegnet(in_channels=23, in_length=256):
    """
    Create EEGNet model using TensorFlow/Keras
    Input shape: (batch_size, channels, length)
    """
    inputs = keras.Input(shape=(in_channels, in_length))
    
    # Add channel dimension: (batch_size, channels, length) -> (batch_size, 1, channels, length)
    x = layers.Reshape((1, in_channels, in_length))(inputs)
    
    # Conv2D layers
    x = layers.Conv2D(16, (1, 7), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.Conv2D(32, (1, 5), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Simple Neural Network
def create_simple_nn(in_channels=23, in_length=256):
    """
    Create a simple feedforward neural network
    Input shape: (batch_size, channels, length)
    """
    inputs = keras.Input(shape=(in_channels, in_length))
    
    # Flatten the input
    x = layers.Flatten()(inputs)
    
    # Simple feedforward layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Basic LSTM model
def create_lstm_model(seq_len=60, n_channels=23, sfreq=256):
    """
    Create LSTM model for seizure detection
    Input shape: (batch_size, seq_len, n_channels, sfreq)
    """
    inputs = keras.Input(shape=(seq_len, n_channels, sfreq))
    
    # Reshape to flatten the channel and frequency dimensions
    # From (batch_size, seq_len, n_channels, sfreq) to (batch_size, seq_len, n_channels*sfreq)
    x = layers.Reshape((seq_len, n_channels * sfreq))(inputs)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Improved LSTM model
def create_improved_lstm_model(seq_len=60, n_channels=23, sfreq=256):
    """
    Improved LSTM model with better architecture for seizure detection
    """
    inputs = keras.Input(shape=(seq_len, n_channels, sfreq))
    
    # Option 1: CNN feature extraction + LSTM
    # Extract features from each time step using CNN
    x = layers.TimeDistributed(layers.Conv1D(64, 7, activation='relu', padding='same'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.Conv1D(32, 5, activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)
    
    # Now x has shape (batch_size, seq_len, 32)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))(x)
    
    # Attention mechanism (simple)
    # x = layers.Dense(64, activation='tanh')(x)
    # attention_weights = layers.Dense(1, activation='softmax')(x)
    # x = layers.Multiply()([x, attention_weights])
    
    # Dense layers with regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Convolutional neural network and LSTM hybrid
def create_cnn_lstm_hybrid(seq_len=60, n_channels=23, sfreq=256):
    """
    Alternative hybrid CNN-LSTM approach
    """
    inputs = keras.Input(shape=(seq_len, n_channels, sfreq))
    
    # Reshape for spatial convolution across channels
    x = layers.Reshape((seq_len, n_channels, sfreq, 1))(inputs)
    
    # 3D CNN to capture spatial-temporal patterns
    x = layers.Conv3D(32, (3, 3, 7), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((1, 1, 2))(x)
    x = layers.Conv3D(64, (3, 3, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    
    # Reshape for LSTM
    x = layers.RepeatVector(seq_len)(x)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(64, return_sequences=False, dropout=0.3)(x)
    
    # Output
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model


def main():
    seizure_info = load_seizure_time(seizure_times_path)
    sliced_seizure_info = {}
    # for file, seizure in seizure_info.items():
    #     if re.search("^chb04/", file):
    #         break
    #     # if re.search("^chb24/", file):
    #     #     break
    #     sliced_seizure_info[file] = seizure

    count = 0
    processed_folders = set()  # Keep track of which folders we've processed
    data_dir = 'chb-mit-scalp-eeg-database-1.0.0/'

    for file_name, seizure in seizure_info.items():
        # Extract the folder name (e.g., "chb01", "chb02", etc.)
        folder_match = re.match(r"^(chb\d+)/", file_name)
        if folder_match:
            folder_name = folder_match.group(1)
            
            # Skip if patients are too young (<3): chb06, 10, 12, 13
            if folder_name in ["chb06", "chb10", "chb12", "chb13"]:
                continue
        
            # Skip if we've already processed this folder
            if folder_name in processed_folders:
                continue
            
            sliced_seizure_info[file_name] = seizure

            # Mark this folder as processed
            processed_folders.add(folder_name)



    # sliced_seizure_info = dict(list(seizure_info.items())[:30])   # get the first 10 patients

    # generate dataset
    data_dir = 'chb-mit-scalp-eeg-database-1.0.0/'
    merged_datas, merged_labels = generate_dataset(sliced_seizure_info, data_dir, delta=100, offset=2, seq_len=2)
    print(f'merged_datas: {merged_datas.shape}\tmerged_labels: {merged_labels.shape}')


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(merged_datas, merged_labels, stratify=None, test_size=0.2, shuffle=False)

    # Create and compile the model
    # model = create_simple_nn(in_channels=X_train.shape[1], in_length=X_train.shape[2])
    model = create_improved_lstm_model(seq_len=X_train.shape[1], n_channels=X_train.shape[2], sfreq=X_train.shape[3])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stopping],   # Early Stopping
        verbose=1
    )

    # Save the trained model
    model.save('seizure_detection_lstm_2_seconds_model.keras')
    print("Model saved as seizure_detection_lstm_2_seconds_model.keras")

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))


if __name__ == "__main__":
    main()
