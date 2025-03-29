import os
import pandas as pd
from IPython.display import display
from common import get_csv_paths, get_tbs_path

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
    
    
def display_summary_v1(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, key, threshold, title, year=2010):
    summary_tables=[]
    model_names = ["NDV All Bands", "NDV 3 Bands", "VNIR All Bands", "VNIR Augmentation", "VNIR 3 Bands", "VNIR Hyperparameter Tuning", "VNIR  Mixup & Cutmix"]
    for model_name, df in zip(model_names, [csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut]):
        summary_table = {
        "Model Configuration": model_name,  # Add the model name as an identifier
        f"Mean {title}": round(df[key].mean(), 2),
        f"Std. Dev": round(df[key].std(), 2),
        f"Median {title} (50%)": round(df[key].median(), 2),
        f"25th Percentile": round(df[key].quantile(0.25), 2),
        f"75th Percentile": round(df[key].quantile(0.75), 2),
        f"Fraction {title} > {threshold}": f"{round((df[key] > threshold).mean() * 100, 2)}%"  # Convert to percentage
        }
        summary_tables.append(summary_table)

    summary_df = pd.DataFrame(summary_tables)
    save_path = os.path.join(rf"D:\Source\Test\MasterThesis\metrics\res_2010", f"{title}.csv")
    print(save_path)
    summary_df.to_csv(save_path, index=False)
    print(summary_df)

def display_iou_f1_ssi_v1(year):
    csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut = get_csv(year)
    for key, title, threshold in [('IOU_VECTOR', "IoU", 0.5), ('F1_BOUNDARY', "F1 Boundary", 0.7), ('Shape_Similarity_Index', "SSI", 0.7)]:
        display_summary_v1(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, key, threshold,  title, year)
    
def display_iou_f1_ssi(year):
    csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut = get_csv(year)
    for key, title in [('IOU_VECTOR', "IoU"), ('F1_BOUNDARY', "F1"), ('Shape_Similarity_Index', "Shape Similarity Index")]:
        display_summary(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, key, title, year)
        
def display_tbs():
    mainPath =r"D:\Source\Test\MasterThesis\metrics\res_2010"
    path_NDV_all, path_NDV_band3, path_VNIR_all, path_VNIR_aug, path_VNIR_band3, path_VNIR_hp, path_VNIR_mix_cut= get_tbs_path(mainPath, "2010")
    csv_NDV_all= pd.read_csv(path_NDV_all)
    csv_NDV_band3= pd.read_csv(path_NDV_band3)
    csv_VNIR_all= pd.read_csv(path_VNIR_all)
    csv_VNIR_aug= pd.read_csv(path_VNIR_aug)
    csv_VNIR_band3= pd.read_csv(path_VNIR_band3)
    csv_VNIR_hp= pd.read_csv(path_VNIR_hp)
    csv_VNIR_mix_cut= pd.read_csv(path_VNIR_mix_cut)
    #display_summary(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, "TEMPORAL_BOUNDARY_SHIFT","TBS", "2022-2010")
    display_summary_v1(csv_NDV_all, csv_NDV_band3, csv_VNIR_all, csv_VNIR_aug, csv_VNIR_band3, csv_VNIR_hp, csv_VNIR_mix_cut, "TEMPORAL_BOUNDARY_SHIFT", 60, "TBS", "2022-2010")
