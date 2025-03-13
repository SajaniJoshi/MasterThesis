import os
import csv  

def saveAsCSV(headers, path,  data, isIOU= False): #Write each epoch loss in CSV
    # Check if the file exists
    if os.path.exists(path) and isIOU:
        # If file exists, open in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            print(f"Data appended to existing file {path}.")
    
    else:
        with open(path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            print(f"Data saved to {path}")
    
def SaveModels(path, model_list):
    os.makedirs(path, exist_ok=True)
    for model in model_list:
        model.net.save_parameters(os.path.join(path, f"model_VNIR_{model.epoch}.params"))