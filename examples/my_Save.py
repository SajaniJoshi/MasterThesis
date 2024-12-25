import os
import csv  
import const

def saveAsCSV(headers, path,  data): #Write each epoch loss in CSV
        with open(path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {path}")
    
def SaveModels(model_list):
    for model in model_list:
        path =  os.path.join(const.trained_model_path, f"model_VNIR_{model.epoch}.params")
        model.net.save_parameters(path)