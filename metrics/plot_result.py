import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  common import get_csv_paths, getType

def plotResults(input, column, title, output):
    method, color, type, year, output1 = getType(input, title)

    # Read data from CSV
    df = pd.read_csv(input)
    data = df[column]
    
    # Create histogram with 10 bins for IOU values
    plt.figure(figsize=(10, 6))
    bins = 10

    # Calculate histogram data with 10 bins
    counts, edges, _ = plt.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')

    # Calculate total number of samples
    total_counts = int(sum(counts))
    
    # Convert frequency to percentage
    percentages = counts / counts.sum() * 100
    maxPercentages =int(max(percentages))
    # Clear previous plot
    plt.clf()

    # Plot the histogram with percentage frequency
    plt.bar((edges[:-1] + edges[1:]) / 2, percentages, width=np.diff(edges), color=color, alpha=0.7, edgecolor='black')

    # Annotate bars with percentages and IOU values
    for i in range(len(counts)):
        y_position = percentages[i] + 1
        if percentages[i] > maxPercentages:
           y_position = percentages[i] + 0.3
    
        plt.text((edges[i] + edges[i+1]) / 2, y_position, f"{percentages[i]:.1f}%", 
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

    # Set labels and title
    plt.xlabel(f"{title} Values")
    plt.ylabel("Frequency (%)")
    #Total: {total_counts} Samples, 
    plt.title(f"Histogram of {title} Values for {type} images {method} (Year: {year})", pad=10)
    plt.grid(axis='y', alpha=0.75)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Avoid overlap
    # Save the plot
    plt.savefig(output, format='png', dpi=150)

    # Show plot
    plt.show()

    # Print out bin ranges and corresponding percentages
    for i in range(len(counts)):
        print(f"{percentages[i]:.1f}% of samples have {method} in range {edges[i]:.2f} - {edges[i+1]:.2f}")
        
def plot_all_results(year):
     path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut = get_csv_paths(year)
     for column, title in [("IOU_VECTOR", "IoU"), ("F1_BOUNDARY", "F1"), ("Shape_Similarity_Index", "Shape Similarity Index")]:
        for input in [path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut]:
            plotResults(input, column, title, input.replace(".csv", f"_{title}.png"))