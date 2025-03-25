import os
import pandas as pd
import numpy as np

# Calculate transverse momentum (pt)
def pt(px, py):
    return np.sqrt(px**2 + py**2)

# Calculate phi
def phi(px, py):
    return np.arctan2(py, px)

# Calculate eta (pseudorapidity)
def eta(px, py, pz):
    p_t = pt(px, py)
    return np.arcsinh(pz / p_t)

# Calculate invariant mass with error handling
def inv_mass_pair(E1, px1, py1, pz1, E2, px2, py2, pz2):
    mass_squared = (E1 + E2)**2 - ((px1 + px2)**2 + (py1 + py2)**2 + (pz1 + pz2)**2)
    return np.sqrt(mass_squared) if mass_squared >= 0 else np.nan

# Calculate transverse momentum of a pair
def pt_pair(px1, py1, px2, py2):
    return np.sqrt((px1 + px2)**2 + (py1 + py2)**2)

# Filter events to ensure proper pairing of leptons based on PID
def filter_pid_events(df):
    positive_pid_count = (df[['PID1', 'PID2', 'PID3', 'PID4']] > 0).sum(axis=1)
    negative_pid_count = (df[['PID1', 'PID2', 'PID3', 'PID4']] < 0).sum(axis=1)
    return df[(positive_pid_count < 3) & (negative_pid_count < 3)]

# Process and combine all information into a single DataFrame
def process_file_combined(file_path, output_folder, n_max=None):
    columns = ['Run', 'Event', 'PID1', 'Q1', 'E1', 'px1', 'py1', 'pz1', 'eta1', 'phi1',
            'PID2', 'Q2', 'E2', 'px2', 'py2', 'pz2', 'eta2', 'phi2',
            'PID3', 'Q3', 'E3', 'px3', 'py3', 'pz3', 'eta3', 'phi3',
            'PID4', 'Q4', 'E4', 'px4', 'py4', 'pz4', 'eta4', 'phi4']

    dtype_spec = {
        'Run': 'int64', 'Event': 'int64',
        'PID1': 'int64', 'Q1': 'int64', 'E1': 'float64', 'px1': 'float64', 'py1': 'float64', 'pz1': 'float64', 'eta1': 'float64', 'phi1': 'float64',
        'PID2': 'int64', 'Q2': 'int64', 'E2': 'float64', 'px2': 'float64', 'py2': 'float64', 'pz2': 'float64', 'eta2': 'float64', 'phi2': 'float64',
        'PID3': 'int64', 'Q3': 'int64', 'E3': 'float64', 'px3': 'float64', 'py3': 'float64', 'pz3': 'float64', 'eta3': 'float64', 'phi3': 'float64',
        'PID4': 'int64', 'Q4': 'int64', 'E4': 'float64', 'px4': 'float64', 'py4': 'float64', 'pz4': 'float64', 'eta4': 'float64', 'phi4': 'float64',
    }
    
    # Load data
    df = pd.read_csv(file_path, names=columns, dtype=dtype_spec, skiprows=1, low_memory=False)
    
    # Apply row limit if necessary
    if n_max is not None and len(df) > n_max:
        df = df.iloc[:n_max]
    
    # Filter events based on PID pairing criteria
    df = filter_pid_events(df)
    
    # Reset index to ensure calculations align with filtered rows
    df = df.reset_index(drop=True)

    # Step 1: Process leading leptons, including all four ordered by pt
    lepton_info_df = pd.DataFrame({
        'pt1': pt(df['px1'], df['py1']), 'eta1': df['eta1'], 'phi1': phi(df['px1'], df['py1']), 'energy1': df['E1'], 'PID1': df['PID1'],
        'pt2': pt(df['px2'], df['py2']), 'eta2': df['eta2'], 'phi2': phi(df['px2'], df['py2']), 'energy2': df['E2'], 'PID2': df['PID2'],
        'pt3': pt(df['px3'], df['py3']), 'eta3': df['eta3'], 'phi3': phi(df['px3'], df['py3']), 'energy3': df['E3'], 'PID3': df['PID3'],
        'pt4': pt(df['px4'], df['py4']), 'eta4': df['eta4'], 'phi4': phi(df['px4'], df['py4']), 'energy4': df['E4'], 'PID4': df['PID4']
    })

    lepton_info_df['sorted_leptons'] = lepton_info_df.apply(lambda row: sorted(
        [(row['pt1'], row['eta1'], row['phi1'], row['energy1'], row['PID1']),
         (row['pt2'], row['eta2'], row['phi2'], row['energy2'], row['PID2']),
         (row['pt3'], row['eta3'], row['phi3'], row['energy3'], row['PID3']),
         (row['pt4'], row['eta4'], row['phi4'], row['energy4'], row['PID4'])],
        key=lambda x: x[0],  # Sort by pt
        reverse=True), axis=1)

    # Extract ordered lepton information
    df_leptons_ordered = pd.DataFrame({
        'Leading_lepton_pt': lepton_info_df['sorted_leptons'].apply(lambda x: x[0][0]),
        'Leading_lepton_eta': lepton_info_df['sorted_leptons'].apply(lambda x: x[0][1]),
        'Leading_lepton_phi': lepton_info_df['sorted_leptons'].apply(lambda x: x[0][2]),
        'Leading_lepton_energy': lepton_info_df['sorted_leptons'].apply(lambda x: x[0][3]),
        'Leading_lepton_PID': lepton_info_df['sorted_leptons'].apply(lambda x: x[0][4]),
        'Second_lepton_pt': lepton_info_df['sorted_leptons'].apply(lambda x: x[1][0]),
        'Second_lepton_eta': lepton_info_df['sorted_leptons'].apply(lambda x: x[1][1]),
        'Second_lepton_phi': lepton_info_df['sorted_leptons'].apply(lambda x: x[1][2]),
        'Second_lepton_energy': lepton_info_df['sorted_leptons'].apply(lambda x: x[1][3]),
        'Second_lepton_PID': lepton_info_df['sorted_leptons'].apply(lambda x: x[1][4]),
        'Third_lepton_pt': lepton_info_df['sorted_leptons'].apply(lambda x: x[2][0]),
        'Third_lepton_eta': lepton_info_df['sorted_leptons'].apply(lambda x: x[2][1]),
        'Third_lepton_phi': lepton_info_df['sorted_leptons'].apply(lambda x: x[2][2]),
        'Third_lepton_energy': lepton_info_df['sorted_leptons'].apply(lambda x: x[2][3]),
        'Third_lepton_PID': lepton_info_df['sorted_leptons'].apply(lambda x: x[2][4]),
        'Fourth_lepton_pt': lepton_info_df['sorted_leptons'].apply(lambda x: x[3][0]),
        'Fourth_lepton_eta': lepton_info_df['sorted_leptons'].apply(lambda x: x[3][1]),
        'Fourth_lepton_phi': lepton_info_df['sorted_leptons'].apply(lambda x: x[3][2]),
        'Fourth_lepton_energy': lepton_info_df['sorted_leptons'].apply(lambda x: x[3][3]),
        'Fourth_lepton_PID': lepton_info_df['sorted_leptons'].apply(lambda x: x[3][4]),
    })

    # Step 2: Process boson pairs
    Z_mass = 91.1876
    inv_masses_1, inv_masses_2, pt_pairs_1, pt_pairs_2 = [], [], [], []

    for index, row in df.iterrows():
        particles = [
            (row['PID1'], row['E1'], row['px1'], row['py1'], row['pz1']),
            (row['PID2'], row['E2'], row['px2'], row['py2'], row['pz2']),
            (row['PID3'], row['E3'], row['px3'], row['py3'], row['pz3']),
            (row['PID4'], row['E4'], row['px4'], row['py4'], row['pz4']),
        ]
        
        valid_pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        valid_pairs = [pairing for pairing in valid_pairings
                       if particles[pairing[0][0]][0] != particles[pairing[0][1]][0] and 
                       particles[pairing[1][0]][0] != particles[pairing[1][1]][0]]

        best_diff = float('inf')
        best_masses, best_pts = (None, None), (None, None)

        for (i1, j1), (i2, j2) in valid_pairs:
            mass1 = inv_mass_pair(*particles[i1][1:], *particles[j1][1:])
            mass2 = inv_mass_pair(*particles[i2][1:], *particles[j2][1:])
            pt1 = pt_pair(particles[i1][2], particles[i1][3], particles[j1][2], particles[j1][3])
            pt2 = pt_pair(particles[i2][2], particles[i2][3], particles[j2][2], particles[j2][3])

            if np.isnan(mass1) or np.isnan(mass2):
                continue
            
            diff = min(abs(Z_mass - mass1), abs(Z_mass - mass2))
            if diff < best_diff:
                best_diff = diff
                best_masses = (mass1, mass2)
                best_pts = (pt1, pt2)

        if best_masses[0] < best_masses[1]:
            best_masses = (best_masses[1], best_masses[0])
            best_pts = (best_pts[1], best_pts[0])

        inv_masses_1.append(best_masses[0])
        inv_masses_2.append(best_masses[1])
        pt_pairs_1.append(best_pts[0])
        pt_pairs_2.append(best_pts[1])

    # Add results to the dataframe and remove NaNs
    df_computed = pd.DataFrame({
        'InvMass_Pair1': inv_masses_1,
        'InvMass_Pair2': inv_masses_2,
        'PT_Pair1': pt_pairs_1,
        'PT_Pair2': pt_pairs_2
    })

    # Combine all information into a single DataFrame and remove NaNs
    df_combined = pd.concat([df, df_leptons_ordered, df_computed], axis=1).dropna(axis=1, how='all')

    # Check for columns consistency before saving
    print("Final columns before saving:", df_combined.columns)
    print("Sample row length:", df_combined.iloc[0].count())

    # Save the combined file
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    df_combined.to_csv(output_file_path, index=False)

# Main function to process all files and combine info
def process_all_files_combined(mc_folder, output_folder, n_max=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(mc_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(mc_folder, file_name)
            print(f"Processing {file_name}")
            process_file_combined(file_path, output_folder, n_max)

if __name__ == "__main__":
    mc_folder = r"C:\Users\CaioV\Dropbox (MIT)\JLab_Higgs_Caio_Marina\Higgs_Boson\MC"
    output_folder = r"C:\Users\CaioV\Dropbox (MIT)\JLab_Higgs_Caio_Marina\Higgs_Boson\MC_processed_combined"
    n_max = None
    process_all_files_combined(mc_folder, output_folder)
