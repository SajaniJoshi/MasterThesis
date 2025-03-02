import os
from common import saveAsCSV, get_csv_name, get_gt_pred
from cal_iou import cal_iou_raster_shape, compute_shapefile_iou
from cal_f1 import cal_f1_boundary, calculate_f1_shp
from cal_ssi import shape_similarity_index, calculate_shape_similarity_index

def get_results(gt_vector_path, pred_path, mask):
    results =[]
    for root, dirs, files in os.walk(pred_path):
        for dir in dirs:
            predicted_raster = os.path.join(root, dir, f"{dir}_extend.tif")
            predicted_raster_boundary = os.path.join(root, dir, f"{dir}_boundary.tif")
            predicted_shp = os.path.join(root, dir, f"{dir}.shp")
            ground_truth_shp = os.path.join(gt_vector_path, f"tile_{dir}.shp")
            ground_truth_raster = os.path.join(mask, f"{dir}.tif")
            raster_iou, shapefile_iou  = cal_iou_raster_shape(ground_truth_raster, predicted_raster, ground_truth_shp, predicted_shp)
            f1 = cal_f1_boundary(ground_truth_raster, predicted_raster_boundary)
            ssi= shape_similarity_index(ground_truth_raster, predicted_raster)
            results.append({ "ID": dir,"IOU_RASTER": raster_iou, "IOU_VECTOR": shapefile_iou, "F1_BOUNDARY": f1, "Shape_Similarity_Index": ssi}) 
    return results
    
def save_result(gt_vector_paths, pred_path, mask, name):
    results = get_results(gt_vector_paths, pred_path, mask)
    csv_name = get_csv_name(pred_path, name)
    saveAsCSV(["ID", "IOU_RASTER", "IOU_VECTOR", "F1_BOUNDARY", "Shape_Similarity_Index"], csv_name, results)
    
def save_result_2022():
    mask_2022 = r"D:\Source\Input\Data\2022\BB\XX_Reference_Masks_ResUNetA"
    gt_vector_path =r"E:\Master_Chemnitz\Output\IACS_2022"
    pred_path_vnir_all_2022 = r"E:\Master_Chemnitz\Output\Result_2022\VNIR\648\result"
    pred_path_ndv_all_2022 = r"E:\Master_Chemnitz\Output\Result_2022\NDV\648\result"
    pred_path_vnir_band3_2022= r"E:\Master_Chemnitz\Output\Result_2022_band3\VNIR\648\result"
    pred_path_ndv_band3_2022= r"E:\Master_Chemnitz\Output\Result_2022_band3\NDV\648\result"
    pred_path_vnir_aug_2022= r"E:\Master_Chemnitz\Output\Result_2022_band_aug\VNIR\648\result"
    pred_path_vnir_hp_2022= r"E:\Master_Chemnitz\Output\Result_hp_2022\VNIR\648\result"
    pred_path_vnir_cut_mix_2022= r"E:\Master_Chemnitz\Output\Result_mix_cut_2022\VNIR\648\result"

    for pred in  [pred_path_vnir_all_2022,  pred_path_ndv_all_2022, pred_path_vnir_band3_2022,  pred_path_ndv_band3_2022, pred_path_vnir_aug_2022,pred_path_vnir_hp_2022,  pred_path_vnir_cut_mix_2022]:
        save_result(gt_vector_path, pred, mask_2022, "R2022_iou")
        

def get_results_2010( gt_vector_path_2010, file_2010):
    results = []
    for root, dirs, files in os.walk(file_2010):
        for dir in dirs:
            print("ID", dir)
            gt_shp= os.path.join(gt_vector_path_2010, f"tile_{dir}.shp")
            predicted_shp = os.path.join(root, dir, f"{dir}.shp")
            ground_truth, predicted = get_gt_pred(gt_shp, predicted_shp)
            if ground_truth is not None and predicted is not None:
                iou, intersection_area = compute_shapefile_iou(ground_truth, predicted)
                f1 = calculate_f1_shp(ground_truth, predicted, intersection_area)
                ssi = calculate_shape_similarity_index(ground_truth, predicted)
                results.append({ "ID": dir, "IOU_VECTOR": iou, "F1_BOUNDARY": f1, "Shape_Similarity_Index": ssi})
    return results 

def save_result_2010():
    pred_2010_file = r"E:\Master_Chemnitz\Output\Result_2010\VNIR"
    pred_2010_ndv_file = r"E:\Master_Chemnitz\Output\Result_2010\NDV"
    pred_2010_vnir_band3_file = r"E:\Master_Chemnitz\Output\Result_2010_band3\VNIR"
    pred_2010_ndv_band3_file = r"E:\Master_Chemnitz\Output\Result_2010_band3\NDV"
    pred_2010_vnir_aug_file = r"E:\Master_Chemnitz\Output\Result_2010_band_aug\VNIR"
    pred_2010_vnir_hp_file = r"E:\Master_Chemnitz\Output\Result_hp_2010\VNIR"
    pred_2010_vnir_mix_cut_file = r"E:\Master_Chemnitz\Output\Result_mix_cut_2010\VNIR"
    gt_vector_path_2010 =r"E:\Master_Chemnitz\Output\IACS_2010"
    for file_2010 in [pred_2010_file, pred_2010_ndv_file, pred_2010_vnir_band3_file, pred_2010_ndv_band3_file, pred_2010_vnir_aug_file, pred_2010_vnir_hp_file, pred_2010_vnir_mix_cut_file]:
        csv_name = get_csv_name(file_2010, "Result")
        path = os.path.join(r"D:\Source\Test\MasterThesis\metrics\res_2010", csv_name)
        results = get_results_2010(gt_vector_path_2010, file_2010)
        saveAsCSV(["ID", "IOU_VECTOR", "F1_BOUNDARY", "Shape_Similarity_Index"], path, results)

        