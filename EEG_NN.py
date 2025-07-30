import mne
import numpy.ma as ma
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

#Discrete energy calculation
def discrete_energy(data, s):
    d_e = 0
    for t1 in range(data.shape[1]):
        for t2 in range(t1 + 1, data.shape[1]):
            x = np.sqrt(np.sum(np.square(data[:, t1] - data[:, t2])))
            d_e += x ** (-1 * s) if x != 0 else 0
    return d_e

# === Data Preprocessing === # 
# Read in the whole dataset


# Define the path to the pickle file
file_path = 'seizure_times.pkl' 

# read in the seizure_time file
with open(file_path, 'rb') as file:
    # Load the data from the file
    seizure_times = pickle.load(file)   # dictionary of seizure time (key: filename, value: list of start and end seizure times))

# Split the seizure times into 1-second intervals
seizure_data = np.array({})
for time_list in seizure_times.values():
    for seizure_time in time_list:      # each seizure_time is a tuple (start_time, end_time) in seconds
        start_time = seizure_time[0]
        end_time = seizure_time[1]
        for i in range(start_time, end_time):
            start_index = i * 256             # each second has 256 samples
            end_index = (i + 1) * 256



        



# For each seizure data, we need to find an non-seizure data to balance the dataset






# #discrete energy graph 1 second with the seizure data
# #discrete energy graph 1 second with the non-seizure data
# mne.sys_info()
# seizdiff_max=[]
# nonseizdiff_max=[]

# de_seiz=[]
# #de_min_seiz=[]

# de_noseiz=[]
# #de_min_noseiz=[]

# seizures = {}


# # Load seizure data
# for i in seizures:
#     file = "chb-mit-scalp-eeg-database-1.0.0/chb"+str(i[3:5])+"/"+str(i[6:])
#     data = mne.io.read_raw_edf(file)
#     raw = data.get_data()
#     count=False
#     for j in seizures[i]:
#         raw_seizure = raw[:,j[0]*256:j[1]*256]
#         seizdiff = raw_seizure[1:len(raw_seizure)]-raw_seizure[0:len(raw_seizure)-1]
#         seizdiff_max.append(np.max(seizdiff))
#         for s in range(23, 26):
#             de_seiz.append(energy_graph(raw_seizure[:, 2000:2256], s))
#             print("siezure for patient "+str(i[6:]))
#         if (2000<j[0]*256 and 2256<j[1]*256) or (2000>j[0]*256 and 2256>j[1]*256):
#             de_noseiz.append(energy_graph(raw[:, 2000:2256], s))
#             print("non siezure for patient "+str(i[6:]))
#         else:
#             print("find a new way")
#     if count == False:
#         nonseizdiff = raw[1:len(raw)]-raw[0:len(raw)-1]
#         nonseizdiff_max.append(np.max(nonseizdiff))
#         count=True


# # neural network for detecting if a person has a seizure or not based on the discrete differences of the data

# # Example 1D data (replace with your actual data)
# # For simplicity, assume each sample is a 1D array of length 100
# X_seizure = np.array(x) # 100 samples of 1D data
# X_non_seizure = np.array(y) # 100 samples of 1D data

# # Combine data
# X = np.concatenate((X_seizure, X_non_seizure), axis=0)
# y = np.concatenate((np.ones(len(X_seizure)), np.zeros(len(X_non_seizure))), axis=0)

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Reshape the data to 2D before scaling - Each sample is a row, and there's one feature column
# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)

# # Optional: Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Define the model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input shape for 1D data
#     Dropout(0.5),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid') # Binary classification
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")