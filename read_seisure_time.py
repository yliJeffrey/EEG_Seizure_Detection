import pickle

# Define the path to your pickle file
file_path = 'seizure_times.pkl' 

# 'rb' means 'read binary' mode
with open(file_path, 'rb') as file:
    # Load the data from the file
    loaded_data = pickle.load(file)

n = 0
for test,time in loaded_data.items():
    print(f"{test}: {time}")
    n += len(time)
    difference = time[0][1] - time[0][0]
    print(difference)

print(n)