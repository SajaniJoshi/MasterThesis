import os
import pandas as pd
import matplotlib.pyplot as plt
import const


# Define the consistent axis limits
training_loss_max = 0.5585567473899573  # highest max training loss
training_loss_min = 0.4466053412761539  # lowest min training loss
validation_loss_max = 0.6971921492367983  # highest max validation loss
validation_loss_min = 0.454997194930911  # lowest min validation loss

# Overall y-axis limits
y_min = min(training_loss_min, validation_loss_min)
y_max = max(training_loss_max, validation_loss_max)



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
    for i, file_path in enumerate(const.file_paths):
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
    savePath =os.path.join(const.result_path_2022,"img_2022", "all_loss2.png")
    plt.savefig(savePath, format='png', dpi=300)
    plt.show()