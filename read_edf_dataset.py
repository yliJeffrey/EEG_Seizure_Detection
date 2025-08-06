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
    channel_names = raw.ch_names
    # print(f'{filepath}.shape = {data.shape}\tsfreq: {sfreq}')
    return data, sfreq, channel_names


seizure_info = load_seizure_time(seizure_times_path)

data_dir = 'chb-mit-scalp-eeg-database-1.0.0/'
count = 0
processed_folders = set()  # Keep track of which folders we've processed

for file_name, seizure in seizure_info.items():
    # Extract the folder name (e.g., "chb01", "chb02", etc.)
    folder_match = re.match(r"^(chb\d+)/", file_name)
    if folder_match:
        folder_name = folder_match.group(1)
        
        # Skip if we've already processed this folder
        if folder_name in processed_folders:
            continue
            
        # Mark this folder as processed
        processed_folders.add(folder_name)
        
        data, sfreq, channel_names = load_edf(data_dir + file_name)
        print(f"{file_name}: {data.shape}")
        print(f"Channel name: {channel_names}")
        if data.shape[0] == 23:
            count += 1

print(f"Number of edf files with 23 channels: {count}")