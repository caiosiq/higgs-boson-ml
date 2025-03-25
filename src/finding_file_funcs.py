import os
import re
# Get all data files for a given directory and year (MC files)
def get_files(directory, year,full):
    file_paths = []
    labels = []
    
    # Find the correct files
    for file_name in os.listdir(directory):
        match = re.match(r'([a-zA-Z0-9]+)(\d{4})\.csv', file_name)
        if match:
            base_label, file_year = match.groups()
            if year == 'both' or year == file_year:
                file_paths.append(os.path.join(directory, file_name))
                if 'higgs' not in base_label and full==False:
                    labels.append('background')
                else:
                    labels.append(base_label)
    
    return file_paths, labels

# Get the real data files for a given year
def get_real_data_files(directory, year):
    real_data_files = []
    if year == '2011' or year == 'both':
        real_data_files.append(os.path.join(directory, 'clean_data_2011.csv'))
    if year == '2012' or year == 'both':
        real_data_files.append(os.path.join(directory, 'clean_data_2012.csv'))
    return real_data_files