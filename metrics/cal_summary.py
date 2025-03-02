import os
import pandas as pd
from IPython.display import display
from common import get_csv_paths

def get_csv(year):
    path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut = get_csv_paths(year)
    csv_NDV_all= pd.read_csv(path_NDV_all)
    csv_NDV_band3= pd.read_csv(path_NDV_band3)
    csv_VNIR_all= pd.read_csv(path_VNIR_all)
    csv_VNIR_aug= pd.read_csv(path_VNIR_aug)
    csv_VNIR_band3= pd.read_csv(path_VNIR_band3)
    csv_VNIR_hp= pd.read_csv(path_VNIR_hp)
    csv_VNIR_mix_cut= pd.read_csv(path_VNIR_mix_cut)
    return csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut
     
def display_summary(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, key, title, year):
    NDV_all = csv_NDV_all[key].describe()
    NDV_band3 = csv_NDV_band3[key].describe()
    VNIR_all = csv_VNIR_all[key].describe()
    VNIR_aug = csv_VNIR_aug[key].describe()
    VNIR_band3 = csv_VNIR_band3[key].describe()
    VNIR_hp = csv_VNIR_hp[key].describe()
    VNIR_mix_cut = csv_VNIR_mix_cut[key].describe()
    
    summary = pd.DataFrame({
    f"{title} Metric {year}": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
    "NDV all bands": NDV_all.values,
    "NDV 3 bands": NDV_band3.values,
    "VNIR all bands": VNIR_all.values,
    "VNIR 3 bands":VNIR_band3.values,
    "VNIR aug": VNIR_aug.values,
    "VNIR hp": VNIR_hp.values,
    "VNIR Mixup and Cutmix": VNIR_mix_cut.values})
    display(summary)

def display_iou_f1_ssi(year):
    csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut = get_csv(year)
    for key, title in [('IOU_VECTOR', "IoU"), ('F1_BOUNDARY', "F1"), ('Shape_Similarity_Index', "Shape Similarity Index")]:
        display_summary(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, key, title, year)
