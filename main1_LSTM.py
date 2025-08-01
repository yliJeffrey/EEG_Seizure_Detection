import mne
import numpy as np
import os

def load_edf(filepath):
    raw = mne.io.read_raw_edf(filepath)
    data = raw.get_data()
    sfreq = raw.info['sfreq']  # e.g. 256Hz
    # print(f'{filepath}.shape = {data.shape}\tsfreq: {sfreq}')
    return data, sfreq

def parse_seizure_summary(file_path):
    seizures = []
    current_file_name = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("File Name:"):
            current_file_name = line.split(": ")[1]
            i += 1
            continue
        if line.startswith("Number of Seizures in File:"):
            num_seizures = int(line.split(": ")[1])
            if num_seizures > 0:
                for _ in range(num_seizures):
                    i += 1 
                    start_time_line = lines[i].strip()
                    start_time = int(start_time_line.split(": ")[1].split(" ")[0])
                    i += 1 
                    end_time_line = lines[i].strip()
                    end_time = int(end_time_line.split(": ")[1].split(" ")[0])
                    seizures.append({
                        'file_name': current_file_name,
                        'seizure_start_time': start_time,
                        'seizure_end_time': end_time
                    })        
        i += 1
    return seizures

def slice_data(data, sfreq, delta, interval, offset=5, seq_len=60):
    n_channels, total_len = data.shape
    total_seconds = total_len // sfreq
    n_datas = total_seconds - seq_len
    slice_data = np.zeros((n_datas, seq_len, n_channels, sfreq))
    label = np.zeros(n_datas)
    for i in range(n_datas):
        for j in range(seq_len):
            slice_data[i][j] = data[:, (i + j) * sfreq : (i + j + 1) * sfreq]
            if  ((i * seq_len) + j > delta + offset) and ((i * seq_len) + j < delta + interval - offset):
                label[i] = 1
    return slice_data, label

def generate_dataset(seizure_info, data_dir, delta=100, offset=5, seq_len=60):
    datas, labels = [], []
    for seizure in seizure_info:        
        data, sfreq = load_edf(data_dir + seizure['file_name'])
        interval = seizure['seizure_end_time'] - seizure['seizure_start_time']
        # delta = interval // 2
        valid_seconds = interval + delta * 2
        valid_data = data[:, (seizure['seizure_start_time'] - delta) * int(sfreq) : (seizure['seizure_end_time'] + delta) * int(sfreq)]
        slice_datas, label = slice_data(valid_data, int(sfreq), delta, interval, offset, seq_len)        
       
        labels.append(label)
        datas.append(slice_datas)
        print(f'\n{seizure["file_name"]}\tvalid_data.shape = {valid_data.shape}\tslice_data.shape = {slice_datas.shape}\tlabel.shape = {label.shape}\n')

    merged_datas = np.concatenate(datas, axis=0)   
    merged_labels = np.concatenate(labels, axis=0)
    return merged_datas, merged_labels





import tensorflow as tf
from tensorflow import keras
from keras import layers

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



def main():
    # load_edf('data_org/chb01_03.edf')

    ############ START FROM HERE ############
    # get seizure intervals
    seizure_info = parse_seizure_summary('data_org/chb01-summary.txt')
    for seizure in seizure_info:        
        print(f"{seizure['file_name']}: [({seizure['seizure_start_time']} ~ {seizure['seizure_end_time']})]")

    # generate dataset
    data_dir = 'data_org/'
    merged_datas, merged_labels = generate_dataset(seizure_info, data_dir, delta=100, offset=1, seq_len=60)
    print(f'merged_datas: {merged_datas.shape}\tmerged_labels: {merged_labels.shape}')


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(merged_datas, merged_labels, stratify=None, test_size=0.2, shuffle=False)

    # Create and compile the model
    # model = create_simple_nn(in_channels=X_train.shape[1], in_length=X_train.shape[2])
    model = create_lstm_model(seq_len=X_train.shape[1], n_channels=X_train.shape[2], sfreq=X_train.shape[3])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))



if __name__ == "__main__":
    main()
