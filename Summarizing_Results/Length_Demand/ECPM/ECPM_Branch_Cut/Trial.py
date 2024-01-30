import os

# Directory containing the CSV files
directory = 'C:/Users/zizo_/Desktop/Python/Summarizing_Results/ECPM_Branch_Cut'

# List all files in the directory
files = os.listdir(directory)

# Loop through the files and rename the ones starting with 'EBD' to start with 'ECPM'
for filename in files:
    if filename.startswith('EBD') and filename.endswith('.csv'):
        new_filename = filename.replace('EBD', 'ECPM')
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))