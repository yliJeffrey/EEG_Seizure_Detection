import mne
import numpy as np
import os
import pickle

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


seizure_info = load_seizure_time(seizure_times_path)

data_dir = 'chb-mit-scalp-eeg-database-1.0.0/'
count = 0
for file_name, seizure in seizure_info.items():
    data, sfreq = load_edf(data_dir + file_name)
    print(f"{file_name}: {data.shape}")
    if data.shape[0] == 23:
        count += 1

print(f"Number of edf files with 23 channels: {count}")