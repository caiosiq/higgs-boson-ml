import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from src.plotting_funcs import *
from src.finding_file_funcs import *
from src.cutting_funcs import *
from src.get_scalling_funcs import *

# Updated load_data function to handle the combined dataset with leading leptons and boson pairs

def load_data(file_path):
    # Define all columns based on the combined structure, including original columns
    columns = [
        'Run', 'Event', 'PID1', 'Q1', 'E1', 'px1', 'py1', 'pz1', 'eta1', 'phi1',
        'PID2', 'Q2', 'E2', 'px2', 'py2', 'pz2', 'eta2', 'phi2',
        'PID3', 'Q3', 'E3', 'px3', 'py3', 'pz3', 'eta3', 'phi3',
        'PID4', 'Q4', 'E4', 'px4', 'py4', 'pz4', 'eta4', 'phi4',
        'Leading_lepton_pt', 'Leading_lepton_eta', 'Leading_lepton_phi', 'Leading_lepton_energy',
        'Leading_lepton_PID', 'Second_lepton_pt', 'Second_lepton_eta', 'Second_lepton_phi', 'Second_lepton_energy',
        'Second_lepton_PID', 'Third_lepton_pt', 'Third_lepton_eta', 'Third_lepton_phi', 'Third_lepton_energy',
        'Third_lepton_PID', 'Fourth_lepton_pt', 'Fourth_lepton_eta', 'Fourth_lepton_phi', 'Fourth_lepton_energy',
        'Fourth_lepton_PID', 'InvMass_Pair1', 'InvMass_Pair2', 'PT_Pair1', 'PT_Pair2'
    ]
    
    # Specify data types for each column
    dtype = {
        'Run': 'float64', 'Event': 'float64',
        'PID1': 'float64', 'Q1': 'float64', 'PID2': 'float64', 'Q2': 'float64',
        'PID3': 'float64', 'Q3': 'float64', 'PID4': 'float64', 'Q4': 'float64',
        'E1': 'float64', 'px1': 'float64', 'py1': 'float64', 'pz1': 'float64',
        'eta1': 'float64', 'phi1': 'float64', 'E2': 'float64', 'px2': 'float64', 'py2': 'float64', 'pz2': 'float64',
        'eta2': 'float64', 'phi2': 'float64', 'E3': 'float64', 'px3': 'float64', 'py3': 'float64', 'pz3': 'float64',
        'eta3': 'float64', 'phi3': 'float64', 'E4': 'float64', 'px4': 'float64', 'py4': 'float64', 'pz4': 'float64',
        'eta4': 'float64', 'phi4': 'float64',
        'Leading_lepton_pt': 'float64', 'Leading_lepton_eta': 'float64', 'Leading_lepton_phi': 'float64', 'Leading_lepton_energy': 'float64',
        'Leading_lepton_PID': 'float64', 'Second_lepton_pt': 'float64', 'Second_lepton_eta': 'float64', 'Second_lepton_phi': 'float64', 'Second_lepton_energy': 'float64',
        'Second_lepton_PID': 'float64', 'Third_lepton_pt': 'float64', 'Third_lepton_eta': 'float64', 'Third_lepton_phi': 'float64', 'Third_lepton_energy': 'float64',
        'Third_lepton_PID': 'float64', 'Fourth_lepton_pt': 'float64', 'Fourth_lepton_eta': 'float64', 'Fourth_lepton_phi': 'float64', 'Fourth_lepton_energy': 'float64',
        'Fourth_lepton_PID': 'float64', 'InvMass_Pair1': 'float64', 'InvMass_Pair2': 'float64', 'PT_Pair1': 'float64', 'PT_Pair2': 'float64'
    }

    # Load data with specified types
    df = pd.read_csv(file_path, names=columns, dtype=dtype, skiprows=1, low_memory=False)

    # Identify columns expected to be integer and convert them, handling non-integer issues
    int_columns = ['Run', 'Event', 'PID1', 'Q1', 'PID2', 'Q2', 'PID3', 'Q3', 'PID4', 'Q4']

    # Convert columns to integers where possible
    for col in int_columns:
        # Print rows with non-integer values in this column, if any
        non_int_values = df[pd.to_numeric(df[col], errors='coerce').isna()]
        if not non_int_values.empty:
            print(f"Non-integer values found in {col}:")
            print(non_int_values[[col]].head())  # Show a sample

        # Now coerce to integer safely (removing or setting NaN values to a default integer if necessary)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df
# Main function to load datasets based on directory and year, apply cuts, and plot together
def main(relative_path_mc, relative_path_data, year, cut_list, quantities, analyze_real_data_only=False):
    # Normalize paths for compatibility
    directory_data = os.path.normpath(relative_path_data)

    # Get and load real data files, applying cuts
    real_data_files = get_real_data_files(directory_data, year)
    real_dataframes = [apply_cuts(load_data(file), cut_list) for file in real_data_files]

    # If analyzing real data only, plot only the real data
    if analyze_real_data_only:
        plot_histograms([], [], real_dataframes)
        return

    # Proceed with loading MC data and plotting both MC and real data
    directory_mc = os.path.normpath(relative_path_mc)
    file_paths_mc, labels = get_files(directory_mc, year,full=True)
    # Load and apply cuts to each dataset (MC)
    dataframes_mc = []
    scale_factors = []  # Store scale factors for each dataset
    for file_path, label in zip(file_paths_mc, labels):
        print(f"Processing file: {file_path}\nLabel: {label}")
        df = load_data(file_path)
        df_cut = apply_cuts(df, cut_list)
        
        # Get the scale factor based on the filename and year
        scale_factor = get_scale_factor(file_path)
        
        # Store the dataframe and scale factor
        dataframes_mc.append(df_cut)
        scale_factors.append(scale_factor)
        print(f"Scale factor: {scale_factor}")
        print(f'Scaled Counts :{scale_factor*len(df)}')
    
    # Define colors for each dataset
    colors = plt.cm.get_cmap('tab10', len(file_paths_mc)).colors

    # Plot histograms with scaling and real data scatter
    #plot_histograms(dataframes_mc, labels, real_dataframes, scale_factors, quantities,reverse=True)
    #plot_inv4mass(dataframes_mc, labels, real_dataframes, scale_factors)
    p_value_plot(dataframes_mc, labels, real_dataframes, scale_factors)


if __name__ == "__main__":
  
    relative_path_mc = r"outputs/MC_processed_combined_NN_filtered"
    relative_path_data = r"outputs/data_processed_combined_NN_filtered"
    year = 'both'

    # List of manual cuts to apply
    #cut_list = ['InvMass_Pair2>12','InvMass_Pair2<50','InvMass_Pair1>70','Leading_lepton_pt<100', 'Second_lepton_pt<70','PT_Pair1<60','PT_Pair2<60']
    cut_list=[]
    # quantities = {
    #     'Largest Boson Mass': lambda df: df['InvMass_Pair1'],
    #     'Smallest Boson Mass': lambda df: df['InvMass_Pair2'],
    #     'Largest Boson Pt': lambda df: df['PT_Pair1'],
    #     'Smallest Boson Pt': lambda df: df['PT_Pair2'],
        
    #     'Leading Lepton Pt': lambda df: df['Leading_lepton_pt'],
    #     'Second Lepton Pt': lambda df: df['Second_lepton_pt'],
    #     'Invariant Mass 4l':inv_mass_4l,
    #     r'$p_t$':lambda df: pd.concat([pt(df['px1'], df['py1']), pt(df['px2'], df['py2']), pt(df['px3'], df['py3']), pt(df['px4'], df['py4'])]), 
    #     r'$\eta$':lambda df: pd.concat([df['eta1'], df['eta2'], df['eta3'], df['eta4']]),
    #     'Muon Pt': pt_muons,
    #     'Electron Pt': pt_electrons
    # }
    quantities = {

        'Invariant Mass 4l':inv_mass_4l,
    }
    # Run main function
    main(relative_path_mc, relative_path_data, year, cut_list, quantities, analyze_real_data_only=False)
