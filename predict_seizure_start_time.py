# Predict the start of seizure in a edf file
# Using the 2-second interval LSTM classifier to classify each interval in the edf file 
# and the first interval that is classified as seizure is the start time

import numpy as np
from tensorflow import keras
import mne
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

# Load edf
def load_edf(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']  # e.g. 256Hz
    print(f'{filepath}.shape = {data.shape}\tsfreq: {sfreq}')
    return data, sfreq

# Convert the file into intervals
# sfreq: frequency of the data (256Hz)
# delta (seconds): length of non-seizure data (before and after seizure)
# interval (seconds): length of seizure data
# offset (seconds): classified as seizure if at least "offset" seconds of the window overlaps a seizure
# seq_len (seconds): length of the sliding window/interval
def slice_data(data, sfreq, seq_len=60):
    n_channels, total_len = data.shape
    total_seconds = total_len // sfreq
    n_datas = total_seconds - seq_len
    slice_data = np.zeros((n_datas, seq_len, n_channels, sfreq))

    for i in range(n_datas):
        for j in range(seq_len):
            slice_data[i][j] = data[:, (i + j) * sfreq : (i + j + 1) * sfreq]

    return slice_data


# Evaluation
def calculate_overall_metric(actual_start, predicted_start, tolerance_seconds=10):
    if len(predicted_start) == 0:
        return 0.0, 0.0
    
    # Find valid prediction
    valid_indices = np.array([i for i, p in enumerate(predicted_start) if p != -1])
    detection_rate = len(valid_indices) / len(predicted_start)

    if len(valid_indices) == 0:
        return detection_rate, 0.0

    # Calculate accuracy for valid predictions only
    within_tolerence_count = 0
    for i in valid_indices:
        absolute_error = abs(predicted_start[i] - actual_start[i])
        if absolute_error <= tolerance_seconds:
            within_tolerence_count += 1

    accuracy_within_tolerence = within_tolerence_count / len(valid_indices)

    return detection_rate, accuracy_within_tolerence


def main():
    seizure_info = load_seizure_time(seizure_times_path)
    data_dir = "chb-mit-scalp-eeg-database-1.0.0/"

    actual_start = []
    predicted_start = []

    # load every edf file
    for file_name, seizure in seizure_info.items():
        # Extract the folder name (e.g., "chb01", "chb02", etc.)
        folder_match = re.match(r"^(chb\d+)/", file_name)
        if folder_match:
            folder_name = folder_match.group(1)
            # Skip if patients are too young (<3): chb06, 10, 12, 13
            if folder_name in ["chb06", "chb10", "chb12", "chb13"]:
                continue

        data, sfreq = load_edf(data_dir + file_name)

        # Exclude files that do not have 23 channels
        if data.shape[0] != 23:
            continue
        
        # Exclude files with less than 14 seconds of seizure
        interval = seizure[0][1] - seizure[0][0]
        if interval <= 14:
            continue

        sliced_data = slice_data(data, int(sfreq), seq_len=2)
        loaded_model = keras.models.load_model('seizure_detection_lstm_2_seconds_model.keras')
        pred_prob = loaded_model.predict(sliced_data)
        pred_binary = (pred_prob > 0.5).astype(int).flatten()
        print(f"File: {file_name}")
        print(f"Total Number of Results: {len(pred_binary)}")
        actual_start.append(seizure[0][0])
        print(f"Actual Seizure Start Time (seconds): {seizure[0][0]}")

        start_time = 0
        seizure_counter = 0
        isFound = False
        for result in pred_binary:
            if result == 0:
                seizure_counter = 0
            else:
                seizure_counter += 1

            if seizure_counter == 7:
                isFound = True
                break

            start_time += 1
        
        if isFound:
            predicted_start.append(start_time - 7)
            print(f"Predicted Seizure Start Time (seconds): {start_time - 7}\n")
        else:
            predicted_start.append(-1)

    detection_rate, accuracy_within_tolerence = calculate_overall_metric(actual_start, predicted_start, tolerance_seconds=10)
    print(f"Detection Rate: {detection_rate}")
    print(f"Accuracy within 10s: {accuracy_within_tolerence}")

if __name__ == "__main__":
    main()