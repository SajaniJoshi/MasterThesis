import os
import numpy as np
import rasterio

def validate_image(filename, image):
    if image is None or image.size == 0:
        print(f"Image is empty {filename}.")
        return False
    if np.any(np.isnan(image)):
        print(f"Image contains NaN values {filename}.")
        return False
    if np.all(image == 0):
        print(f"Image contains only zero values {filename}.")
        return False
    return True

def load_file(file_path, isMask):
    with rasterio.open(file_path) as src: 
        image = src.read()
        if len(image.shape) < 2:
            print(f"Invalid dimensions for image: {file_path}")
            return None
        if not validate_image(file_path, image):
            print(f"Corrupted or invalid image: {file_path}")
            return None
        if not isMask:           
            image = np.clip(image / 10000.0, 0, 1)
            image = image.astype('float32')                
        if isMask:
            image = np.array(image)
    return image 

def load_pred_file(file_path):
    with rasterio.open(file_path) as src:  
        image = src.read(1)
        if not validate_image(file_path, image):
            print(f"Corrupted or invalid image: {file_path}")
            return None
    return  image

def load_pred_file1(file_path):
    with rasterio.open(file_path) as src:  
        image = src.read(1)
        if not validate_image(file_path, image):
            print(f"Corrupted or invalid image: {file_path}")
            return None
        image[image >= 1] = 1
        image[image <= 0] = 0  
    return  image 

class plotPredictedImage:
    def __init__(self,id, ori, mask, extent, distance, boundary, instSeg, is2022 ):
        self.id = id
        self.ori = ori
        self.mask = mask
        self.extent = extent
        self.distance = distance
        self.boundary = boundary
        self.instSeg = instSeg
        self.is2022 = is2022

def get_ori_pred_ini(id, oriImg,maskImg, extImg, boundaryImg, instImg, distImg, is2022):
    ori = load_file(oriImg, False)
    mask = None
    if is2022:
        mask = load_file(maskImg, True)
    extent = load_pred_file(extImg)
    boundary = load_pred_file(boundaryImg)
    instSeg = load_pred_file1(instImg)
    distance = load_pred_file(distImg)
    return plotPredictedImage(id, ori, mask, extent, distance, boundary, instSeg,is2022)

def get_ori_pred_img(id, oriPath, maskPath, predPath, is2022):
    year= "2022"
    maskImg = os.path.join(maskPath, f"{id}.tif")
    if not is2022:
        year = 2010
        maskImg = None
    oriImg = os.path.join(oriPath, f"{id}_LS_{year}_{year}_VNIR.tif")
    extImg = os.path.join(predPath, id, f"{id}_extend.tif")
    boundaryImg =os.path.join(predPath, id, f"{id}_boundary.tif")
    distImg = os.path.join(predPath, id, f"{id}_distance.tif")
    instImg = os.path.join(predPath, id, f"{id}_inst_seg.tif")
    return get_ori_pred_ini(id, oriImg,maskImg, extImg, boundaryImg, instImg, distImg, is2022)


def get_ori_pred(idsList, is2022):
    oriPath = rf"D:\Source\Input\Data\2022\BB\08X_Features_Multi"
    maskPath = r"D:\Source\Input\Data\2022\BB\XX_Reference_Masks_ResUNetA"
    predPath = r"E:\Master_Chemnitz\Output\Result_mix_cut_2022\VNIR\648\result"
    if not is2022:
        oriPath = r"D:\Source\Input\Data\2010\BB\08X_Features_Multi"
        predPath = r"E:\Master_Chemnitz\Output\Result_2010\VNIR"
    results = []
    for id in idsList:
         results.append(get_ori_pred_img(id,oriPath, maskPath, predPath, is2022))
    return results

def get_ori_perd_2022_2010(idsList):
    results = []
    for id in idsList:
        oriPath = rf"D:\Source\Input\Data\2022\BB\08X_Features_Multi"
        maskPath = r"D:\Source\Input\Data\2022\BB\XX_Reference_Masks_ResUNetA"
        predPath = r"E:\Master_Chemnitz\Output\Result_mix_cut_2022\VNIR\648\result"
        oriPath_2010 = r"D:\Source\Input\Data\2010\BB\08X_Features_Multi"
        predPath_2010 = r"E:\Master_Chemnitz\Output\Result_mix_cut_2010\VNIR"
        results.append(get_ori_pred_img(id,oriPath, maskPath, predPath, True))
        results.append(get_ori_pred_img(id,oriPath_2010, "", predPath_2010, False))
    return results
    
        
        
    