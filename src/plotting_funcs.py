import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs.plotting_funcs import *
from funcs.finding_file_funcs import *
from funcs.cutting_funcs import *
from funcs.get_scalling_funcs import *
from scipy.stats import poisson



# plt.rc('xtick', labelsize=12)  # X-tick labels size
# plt.rc('ytick', labelsize=12)  # Y-tick labels size
# plt.rc('legend', fontsize=12)  # Legend font size
#plt.rc('axes', labelsize=18)  # Axis labels size
import matplotlib.style as mplstyle
style_file_path = r'C:\Users\CaioV\OneDrive - Massachusetts Institute of Technology\Paper Styles\light'
mplstyle.use(style_file_path)

def plot_histograms(dataframes, labels, real_data, scale_factors, quantities={}, stacked=True,reverse=True):
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2'   # Pink
    ]
    Nrows = 1
    Ncolumns = 1
    fig, axs = plt.subplots(Nrows, Ncolumns, figsize=(10 * Ncolumns, 10 * Nrows))
    
    def plot_accumulated_histograms(ax, quantity_name, dataframes, labels, scale_factors, func, colors, stacked, Range,nbins=50):
        accumulated_histograms = {}
        all_bins = None  # Store bins for consistency

        # Calculate and accumulate histograms by unique label
        for df, label, scale_factor in zip(dataframes, labels, scale_factors):
            data = func(df)
            counts, bins = np.histogram(data, bins=nbins, range=Range)
            scaled_counts = counts * scale_factor

            # Accumulate counts for each unique label
            if label not in accumulated_histograms:
                color = colors.pop()
                accumulated_histograms[label] = (scaled_counts, bins, color)
            else:
                
                accumulated_histograms[label] = (
                    accumulated_histograms[label][0] + scaled_counts,  # Sum counts
                    accumulated_histograms[label][1],
                    accumulated_histograms[label][2]
                )
            all_bins = bins  # Ensure bins are consistent

        # Sort accumulated histograms by total counts for stacking order (smallest first)
        if reverse:
            sorted_histograms = sorted(accumulated_histograms.items(), key=lambda x: -np.sum(x[1][0]))
        else:
            sorted_histograms = sorted(accumulated_histograms.items(), key=lambda x: np.sum(x[1][0]))

        # Prepare lists for stacking
        all_accumulated_counts = [h[1][0] for h in sorted_histograms]
        all_colors = [h[1][2] for h in sorted_histograms]
        all_labels = [h[0] for h in sorted_histograms]

        # Plot accumulated histograms
        if stacked:
            ax.hist([all_bins[:-1]] * len(all_accumulated_counts), bins=all_bins, weights=all_accumulated_counts, 
                    stacked=True, color=all_colors, label=all_labels)
        else:
            for accumulated_counts, color, label in zip(all_accumulated_counts, all_colors, all_labels):
                ax.hist(all_bins[:-1], bins=all_bins, weights=accumulated_counts, histtype='step', color=color, label=label)

    def plot_real_data(ax, real_data, func, Range,nbins=50):
        accumulated_real_histogram = None
        for real_df in real_data:
            data_real = func(real_df)
            hist, bin_edges = np.histogram(data_real, bins=nbins, range=Range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            if accumulated_real_histogram is None:
                accumulated_real_histogram = hist
            else:
                accumulated_real_histogram += hist

        if accumulated_real_histogram is not None:
            errors = np.sqrt(accumulated_real_histogram)
            ax.errorbar(bin_centers, accumulated_real_histogram, yerr=errors, color='black', fmt='o', label='Real Data')

    # Plotting for each quantity in quantities
    for j, (quantity_name, func) in enumerate(quantities.items()):
        id_row = j // Ncolumns
        id_column = j % Ncolumns
        if Ncolumns==Nrows==1:
            ax = axs
        else:
            ax = axs[id_row, id_column]
        
        Range = (0, 200)
        nbins=50
        if quantity_name == r'$\phi$' or quantity_name == r'$\eta$':
            Range = (-5, 5)

        if 'Boson Mass' in quantity_name:
            Range = (0, 130)
        if 'Pt' in quantity_name or 'pt' in quantity_name or r'$p_t$' in quantity_name:
            Range = (0, 60)
        # Plot MC histograms
        plot_accumulated_histograms(ax, quantity_name, dataframes, labels, scale_factors, func, colors, stacked, Range,nbins)

        # Plot real data
        plot_real_data(ax, real_data, func, Range,nbins)

        ax.set_xlabel(quantity_name)
        ax.set_ylabel('Counts')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_inv4mass(dataframes, labels, real_data, scale_factors, stacked=True, reverse=True):
    quantity_name = 'Invariant Mass 4l'
    func = inv_mass_4l  # Your function to calculate the invariant mass
    fig,axs = plt.subplots(1,2,figsize=(20,10))
    
    colors = plt.cm.get_cmap('tab20', len(labels)).colors
    Range = (0, 200)
    nbins = 30
    
    histograms = {}  # Dictionary to store histograms

    # Calculate real data histogram
    accumulated_real_histogram = None
    for real_df in real_data:
        data_real = func(real_df)
        hist, bin_edges = np.histogram(data_real, bins=nbins, range=Range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if accumulated_real_histogram is None:
            accumulated_real_histogram = hist
        else:
            accumulated_real_histogram += hist

    if accumulated_real_histogram is not None:
        errors = np.sqrt(accumulated_real_histogram)
        histograms['real_data'] = {
            'hist': accumulated_real_histogram,
            'bin_centers': bin_centers,
            'errors': errors
        }

    # Calculate histograms for each label in dataframes
    accumulated_histograms = {}
    for df, label, scale_factor, color in zip(dataframes, labels, scale_factors, colors):
        data = func(df)
        counts, bins = np.histogram(data, bins=nbins, range=Range)
        scaled_counts = counts * scale_factor

        if label not in accumulated_histograms:
            accumulated_histograms[label] = (scaled_counts, bins, color)
        else:
            accumulated_histograms[label] = (
                accumulated_histograms[label][0] + scaled_counts,
                bins,
                color
            )

    # Store background and Higgs histograms in histograms dictionary
    for label, (counts, bins, color) in accumulated_histograms.items():
        if 'background' in label.lower():
            histograms['background'] = {'hist': counts, 'bins': bins}
        elif 'higgs' in label.lower():
            histograms['higgs'] = {'hist': counts, 'bins': bins}

    # Sort histograms for stacking if necessary
    sorted_histograms = sorted(
        [(label, data) for label, data in histograms.items() if label not in ['real_data']],
        key=lambda x: -np.sum(x[1]['hist']) if reverse else np.sum(x[1]['hist'])
    )
    

    ax = axs[0]
    # Plot real data with error bars
    real_hist = histograms.get('real_data')
    background_hist=histograms.get('background')
    higgs_hist=histograms.get('higgs')
    if real_hist is not None:
        ax.errorbar(real_hist['bin_centers'], real_hist['hist'], yerr=real_hist['errors'], color='black', fmt='o', label='Real Data')

    # Prepare lists for stacking
    all_accumulated_counts = [h[1]['hist'] for h in sorted_histograms]
    all_colors = [accumulated_histograms[h[0]][2] for h in sorted_histograms]  # Colors from original accumulated_histograms
    all_labels = [h[0] for h in sorted_histograms]
    all_bins = accumulated_histograms[sorted_histograms[0][0]][1]  # Use bins from the first histogram as they are consistent
    

    # Plot accumulated histograms using values from histograms dictionary
    if stacked:
        ax.hist([all_bins[:-1]] * len(all_accumulated_counts), bins=all_bins, weights=all_accumulated_counts, 
                stacked=True, color=all_colors, label=all_labels)
    else:
        for accumulated_counts, color, label in zip(all_accumulated_counts, all_colors, all_labels):
            ax.hist(all_bins[:-1], bins=all_bins, weights=accumulated_counts, histtype='step', color=color, label=label)   
    

    
    # Set labels and display plot
    ax.set_xlabel(quantity_name)
    ax.set_ylabel('Counts')
    ax.legend()
    
    ax = axs[1]
    data_minus_background = real_hist['hist'] - background_hist['hist']
    errors = np.sqrt(real_hist['errors'] ** 2 + background_hist['hist'])  # Combined errors

    # Plot real data minus background with error bars
    ax.errorbar(real_hist['bin_centers'], data_minus_background, yerr=errors, color='black', fmt='o', label='Real Data - Background')

    ax.hist(background_hist['bins'][:-1], bins=background_hist['bins'], weights=higgs_hist['hist'], histtype='step', color='blue', label='Higgs')
    plt.tight_layout()
    plt.show()
    

def p_value_plot(dataframes, labels, real_data, scale_factors, stacked=True, reverse=True):
    quantity_name = 'Invariant Mass 4l'
    func = inv_mass_4l  # Your function to calculate the invariant mass
    colors = plt.cm.get_cmap('tab20', len(labels)).colors
    Range = (100, 150)
    nbins = 25
    
    bin_edges = np.linspace(Range[0], Range[1], nbins + 1)
    
    histograms = {}  # Dictionary to store histograms

    # Calculate real data histogram
    accumulated_real_histogram = None
    for real_df in real_data:
        data_real = func(real_df)
        hist, _ = np.histogram(data_real, bins=bin_edges)
        if accumulated_real_histogram is None:
            accumulated_real_histogram = hist
        else:
            accumulated_real_histogram += hist

    if accumulated_real_histogram is not None:
        errors = np.sqrt(accumulated_real_histogram)
        histograms['real_data'] = {
            'hist': accumulated_real_histogram,
            'bins': bin_edges,
            'errors': errors
        }

    # Calculate histograms for each label in dataframes
    accumulated_histograms = {}
    for df, label, scale_factor, color in zip(dataframes, labels, scale_factors, colors):
        data = func(df)
        counts, _ = np.histogram(data, bins=bin_edges)
        scaled_counts = counts * scale_factor

        if label not in accumulated_histograms:
            accumulated_histograms[label] = (scaled_counts, color)
        else:
            accumulated_histograms[label] = (
                accumulated_histograms[label][0] + scaled_counts,  # Accumulate counts
                color
            )

    # Store background and Higgs histograms in histograms dictionary
    for label, (counts, color) in accumulated_histograms.items():
        if 'background' in label.lower():
            histograms['background'] = {'hist': counts, 'bins': bin_edges}
        elif 'higgs' in label.lower():
            histograms['higgs'] = {'hist': counts, 'bins': bin_edges}


    # Example usage
    real_hist = histograms['real_data']
    background_hist = histograms['background']
    higgs_hist = histograms['higgs']

    

    def compute_p_values(real_hist, background_hist, drange):
        """
        Compute p-values using an index range instead of a mass range.
        The p-value is computed as the probability of observing at least N_data events
        given a background expectation of B using a Poisson distribution.

        Args:
            real_hist (dict): Histogram dictionary for real data.
            background_hist (dict): Histogram dictionary for background data.
            drange (int): Number of bins to sum around the current index (centered at i).

        Returns:
            bin_centers (array): Bin centers associated with the p-values.
            p_values (list): List of p-values for each bin center.
        """

        # Extract values from histograms
        real_data_counts = real_hist['hist']
        background_counts = background_hist['hist']
        bin_edges = real_hist['bins']  # Use the same bin edges for both histograms
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers from edges

        # Initialize p-values
        p_values = []

        # Calculate p-values for each bin index
        for i in range(len(bin_centers)):
            # Define the range of indices for summing counts
            min_index = max(i - drange // 2, 0)  # Avoid negative indices
            max_index = min(i + drange // 2 + 1, len(bin_centers))  # Avoid exceeding the histogram length

            # Sum counts within the index range for real data and background data
            N_data = np.sum(real_data_counts[min_index:max_index])
            B = np.sum(background_counts[min_index:max_index])
            print(N_data)
            print(B)
            # Calculate the p-value
            if B > 0:  # Avoid calculations with B = 0
                p_value = 1 - poisson.cdf(N_data - 1, B)  # P(X >= N_data)
            else:
                p_value = 1.0  # If background is 0, p-value is maximum
            p_values.append(p_value)

        return bin_centers, p_values
        
    real_hist = histograms.get('real_data')
    background_hist=histograms.get('background')
    higgs_hist=histograms.get('higgs')
    drange=3

    # Assuming `real_hist` and `background_hist` are already calculated
    bin_centers, p_values = compute_p_values(real_hist, background_hist, drange)
    
    plt.figure(figsize=(20, 12))
    plt.plot(bin_centers, p_values, marker='o', linestyle='-', color='blue', label='P-Value')
    plt.xlim(115,130)
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    plt.yscale('log')  # Use logarithmic scale for better visualization
    plt.xlabel('Mass (GeV/c²)')
    plt.ylabel('P-Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.show()