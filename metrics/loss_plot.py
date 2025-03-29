import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the consistent axis limits
training_loss_max = 0.5585567473899573  # highest max training loss
training_loss_min = 0.4466053412761539  # lowest min training loss
validation_loss_max = 0.6971921492367983  # highest max validation loss
validation_loss_min = 0.454997194930911  # lowest min validation loss

# Overall y-axis limits
y_min = min(training_loss_min, validation_loss_min)
y_max = max(training_loss_max, validation_loss_max)

# List of file paths
file_paths = [
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_VNIR_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_VNIR_3_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_VNIR_AUG_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_VNIR_Hypertuning_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_VNIR_mixup_cutmix_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_NDV_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_NDV_3_2022.csv',
    r'D:\Source\Test\MasterThesis\metrics\res_2022\Loss\LOSS_NDV_AUG_2022.csv'
]

def getType(input):
    method = "using all bands"
    color = 'purple'
    type = 'NDV'
    year = "2022"
    output = input.replace(".csv", ".png")
    if '2010' in input:
        year = '2010'
    if 'VNIR' in input:
        type = 'VNIR'
        color = 'lightblue'
    if 'NDV' in input:
        type = 'NDV'
        color = 'lightGreen'
    if '_3_' in input:
        method = "using 3 bands"
    elif '_AUG_' in input:
        method = "using augmentation"
    return method, color, type, year, output

def plot_all_loss():
    # Initialize subplot
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))  # Adjust the figure size as needed
    axes = axes.flatten()  # Flatten the array to make indexing easier

    # Iterate through file paths and plot data
    for i, file_path in enumerate(file_paths):
        # Read CSV file
        data = pd.read_csv(file_path)
        method, color, type, year, output = getType(file_path)
        title = f"Training and Validation Loss per Epoch \nfor {type} images {method}"
    
        # Plotting the data
        ax = axes[i]
        ax.plot(data['Current Epoch'], data['Training Loss'], label='Training Loss')
        ax.plot(data['Current Epoch'], data['Validation Loss'], label='Validation Loss')
    
        # Setting the title and labels
        ax.set_title(f'{title}', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.grid(True)

        # Set consistent y-axis limits
        ax.set_xlim(0, 50)
        ax.set_ylim([y_min, y_max])
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)

    colors = ['skyblue', 'orange']
    labels = ['Training Loss', 'Validation Loss']
    patches = [plt.matplotlib.patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=12)
    plt.savefig(os.path.join(r"D:\Source\Test\MasterThesis\metrics\res_2022\img_2022", "all_loss2.png"), format='png', dpi=300)
    plt.show()