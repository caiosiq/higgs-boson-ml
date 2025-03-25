import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def pt(px, py):
    return np.sqrt(px**2 + py**2)

def pt_muons(df):
    muon_mask = (df['PID1'] == 11) | (df['PID1'] == -11)
    pt1_muons = pt(df['px1'][muon_mask], df['py1'][muon_mask])

    muon_mask = (df['PID2'] == 11) | (df['PID2'] == -11)
    pt2_muons = pt(df['px2'][muon_mask], df['py2'][muon_mask])

    muon_mask = (df['PID3'] == 11) | (df['PID3'] == -11)
    pt3_muons = pt(df['px3'][muon_mask], df['py3'][muon_mask])

    muon_mask = (df['PID4'] == 11) | (df['PID4'] == -11)
    pt4_muons = pt(df['px4'][muon_mask], df['py4'][muon_mask])

    return pd.concat([pt1_muons, pt2_muons, pt3_muons, pt4_muons])

def pt_electrons(df):
    electron_mask = (df['PID1'] == 13) | (df['PID1'] == -13)
    pt1_electrons = pt(df['px1'][electron_mask], df['py1'][electron_mask])

    electron_mask = (df['PID2'] == 13) | (df['PID2'] == -13)
    pt2_electrons = pt(df['px2'][electron_mask], df['py2'][electron_mask])

    electron_mask = (df['PID3'] == 13) | (df['PID3'] == -13)
    pt3_electrons = pt(df['px3'][electron_mask], df['py3'][electron_mask])

    electron_mask = (df['PID4'] == 13) | (df['PID4'] == -13)
    pt4_electrons = pt(df['px4'][electron_mask], df['py4'][electron_mask])

    return pd.concat([pt1_electrons, pt2_electrons, pt3_electrons, pt4_electrons])

def inv_mass(E, px, py, pz):
    return np.sqrt(E**2 - (px**2 + py**2 + pz**2))

def inv_mass_4l(df):
    E_tot = df['E1'] + df['E2'] + df['E3'] + df['E4']
    px_tot = df['px1'] + df['px2'] + df['px3'] + df['px4']
    py_tot = df['py1'] + df['py2'] + df['py3'] + df['py4']
    pz_tot = df['pz1'] + df['pz2'] + df['pz3'] + df['pz4']
    
    return inv_mass(E_tot, px_tot, py_tot, pz_tot)

def pt_total(df):
    px_tot = df['px1'] + df['px2'] + df['px3'] + df['px4']
    py_tot = df['py1'] + df['py2'] + df['py3'] + df['py4']
    return pt(px_tot,py_tot)

def nth_lepton_info(df, n):
    # Calculate transverse momentum for each lepton
    pt1 = pt(df['px1'], df['py1'])
    pt2 = pt(df['px2'], df['py2'])
    pt3 = pt(df['px3'], df['py3'])
    pt4 = pt(df['px4'], df['py4'])
    
    # Create a DataFrame with all the info for the leptons
    lepton_info_df = pd.DataFrame({
        'pt1': pt1, 'eta1': df['eta1'], 'phi1': df['phi1'], 'energy1': df['E1'],
        'pt2': pt2, 'eta2': df['eta2'], 'phi2': df['phi2'], 'energy2': df['E2'],
        'pt3': pt3, 'eta3': df['eta3'], 'phi3': df['phi3'], 'energy3': df['E3'],
        'pt4': pt4, 'eta4': df['eta4'], 'phi4': df['phi4'], 'energy4': df['E4']
    })
    
    # Sort the leptons by pt (from highest to lowest) for each event
    lepton_info_df['sorted_leptons'] = lepton_info_df.apply(lambda row: sorted([
        (row['pt1'], row['eta1'], row['phi1'], row['energy1']),
        (row['pt2'], row['eta2'], row['phi2'], row['energy2']),
        (row['pt3'], row['eta3'], row['phi3'], row['energy3']),
        (row['pt4'], row['eta4'], row['phi4'], row['energy4'])
    ], key=lambda x: x[0], reverse=True), axis=1)
    
    # Extract the nth lepton's information (pt, eta, phi, energy)
    lepton_info_df['nth_lepton_pt'] = lepton_info_df['sorted_leptons'].apply(lambda x: x[n][0])
    lepton_info_df['nth_lepton_eta'] = lepton_info_df['sorted_leptons'].apply(lambda x: x[n][1])
    lepton_info_df['nth_lepton_phi'] = lepton_info_df['sorted_leptons'].apply(lambda x: x[n][2])
    lepton_info_df['nth_lepton_energy'] = lepton_info_df['sorted_leptons'].apply(lambda x: x[n][3])
    
    return lepton_info_df[['nth_lepton_pt', 'nth_lepton_eta', 'nth_lepton_phi', 'nth_lepton_energy']]

def z_pair_info(df):
    Z_mass = 91.1876  # Z boson mass in GeV

    # Function to calculate invariant mass given a pair of energies and momenta
    def inv_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
        return np.sqrt((E1 + E2)**2 - ((px1 + px2)**2 + (py1 + py2)**2 + (pz1 + pz2)**2))

    # Function to calculate transverse momentum of a pair
    def pt(px1, py1, px2, py2):
        return np.sqrt((px1 + px2)**2 + (py1 + py2)**2)

    # Initialize lists to store properties
    best_pairs = []
    inv_masses = []
    pt_pairs = []

    # Iterate through each event in the DataFrame
    for index, row in df.iterrows():
        # Collect particle properties in a list
        particles = [
            (row['PID1'], row['E1'], row['px1'], row['py1'], row['pz1']),
            (row['PID2'], row['E2'], row['px2'], row['py2'], row['pz2']),
            (row['PID3'], row['E3'], row['px3'], row['py3'], row['pz3']),
            (row['PID4'], row['E4'], row['px4'], row['py4'], row['pz4']),
        ]

        # Generate possible pairings while ensuring particles have different PIDs
        possible_pairings = [
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2))
        ]

        # Filter pairings to include only those with different PIDs within each pair
        filtered_pairings = []
        for pairing in possible_pairings:
            (i1, j1), (i2, j2) = pairing
            if particles[i1][0] != particles[j1][0] and particles[i2][0] != particles[j2][0]:
                filtered_pairings.append(pairing)

        # Initialize variables to find the best pairing
        best_pairing = None
        min_difference = float('inf')
        best_inv_mass_1, best_inv_mass_2 = None, None
        best_pt_1, best_pt_2 = None, None

        # Try each filtered pairing and find the one closest to the Z boson mass
        for pairing in filtered_pairings:
            (i1, j1), (i2, j2) = pairing

            # Calculate invariant mass for each pair
            inv_mass_1 = inv_mass(*particles[i1][1:], *particles[j1][1:])
            inv_mass_2 = inv_mass(*particles[i2][1:], *particles[j2][1:])
            #print(inv_mass_1,inv_mass_2)
            # Calculate transverse momentum for each pair
            pt_1 = pt(particles[i1][2], particles[i1][3], particles[j1][2], particles[j1][3])
            pt_2 = pt(particles[i2][2], particles[i2][3], particles[j2][2], particles[j2][3])

            # Calculate the total difference from the Z boson mass
            diff = abs(Z_mass - inv_mass_1) + abs(Z_mass - inv_mass_2)

            # Update best pairing if this is closer to the Z boson mass
            if diff < min_difference:
                min_difference = diff
                best_pairing = pairing
                best_inv_mass_1, best_inv_mass_2 = inv_mass_1, inv_mass_2
                best_pt_1, best_pt_2 = pt_1, pt_2

        # Append the best pairing and associated properties for the current event
        best_pairs.append(best_pairing)
        inv_masses.append((best_inv_mass_1, best_inv_mass_2))
        #print(best_inv_mass_1,best_inv_mass_2)
        pt_pairs.append((best_pt_1, best_pt_2))

    # Add pairs, invariant masses, and transverse momenta to DataFrame
    df['BestPairs'] = best_pairs
    df['InvMass_Pair1'] = [masses[0] for masses in inv_masses]
    df['InvMass_Pair2'] = [masses[1] for masses in inv_masses]
    df['PT_Pair1'] = [pt[0] for pt in pt_pairs]
    df['PT_Pair2'] = [pt[1] for pt in pt_pairs]
    
    return df