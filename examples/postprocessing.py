import os
import numpy as np
import mxnet as autograd
from examples import myModel
from image_Dictionary import getImage
from myPlots import visualize_all
from decode.postprocessing.instance_segmentation import InstSegm

class predictedImages:
    def __init__(self, config_2022):
        self.config_2022 = config_2022

    def get_model_file_name(self):
        files = os.listdir(self.config_2022.output_models)  # Get all files in the folder
        if files:
            last_file = os.path.join(self.config_2022.output_models, f'model_VNIR_{len(files)-1}.params')
            print(f"This model is using: {last_file}")
            return last_file
        else:
            print("The folder is empty.")
            
    def get_img_metadata(self,id, image_dict, mask_dict, ctx):
        if self.config.year:
            img = getImage(id, image_dict, ctx)
            mask = getImage(id, mask_dict, ctx)
            currentMetadata = image_dict[id]
        else:
            img = getImage(id, image_dict, ctx)
            currentMetadata = image_dict[id]
            mask = None
        return img, mask, currentMetadata
    
    def save_predictions(self,config, ctx, val_ids, t_ext, t_bound, image_dict, mask_dict={}):
        print(f"Starting predictions with t_ext = {t_ext}, t_bound = {t_bound}")
        modelPath = rf"{self.get_model_file_name()}"
        netPredict = myModel.MyFractalResUNetcmtsk(True, modelPath, ctx)
        for id in val_ids:
            print(f"Processing image ID: {id}")
            try:
                img, mask, currentMetadata = self.get_img_metadata(id, image_dict, mask_dict, ctx)
                with autograd.predict_mode():  
                    outputs = netPredict.net(img) 
                    pred_segm  = np.array(outputs[0][0,1,:,:].asnumpy())
                    pred_bound =  np.array(outputs[1][0,1,:,:].asnumpy())
                    pred_dists =  np.array(outputs[2][0,1,:,:].asnumpy()) 
                    pred_segm = 1-pred_segm
                    inst =InstSegm(pred_segm, pred_bound, t_ext=t_ext, t_bound=t_bound)   # perform instance segmentation
                    inst = np.nan_to_num(inst, nan=0)
                    visualize_all(id, img, currentMetadata, outputs, pred_segm, pred_bound, pred_dists, inst, config.result_path)
            
            except Exception as e:
                print(f"Error processing image ID {id}: {e}")
