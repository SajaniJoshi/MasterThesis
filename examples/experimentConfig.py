import os
import const

"The ExperimentConfig class is crucial for managing all configurations required for "
"processing and analyzing satellite images using deep learning models. Below, we explore its functionalities, "
"parameters, and how it facilitates experiment setup."

class ExperimentConfig:
    """
    Configuration class to encapsulate training setup and settings.
    """

    def __init__(self,
                 input_directory: str,
                 year: int,
                 isVnir: bool = True,
                 numberOfimages: int =10,
                 all_bands: bool = True,
                 use_hyperparameter_tuning: bool = False,
                 use_augmentation: bool = False,
                 use_mixup_cutmix: bool = False,
                 ):
        self.input_directory = input_directory
        self.image_type = "VNIR" if isVnir else "NDV"
        self.year = year
        self.numberOfimages = numberOfimages
        self.all_bands = all_bands
        self.hyperparameter_tuning = use_hyperparameter_tuning
        self.use_augmentation = use_augmentation
        self.use_mixup_cutmix = use_mixup_cutmix
        
        if self.year == 2022:
            self.output_dir=const.result_2022
            self.output_ref =const.output_ref_2022
            if not all_bands:
                self.output_dir = const.result_2022_3
            if use_augmentation:
                self.output_dir = const.result_2022_aug
            if use_hyperparameter_tuning:
                self.output_dir = const.result_hp_2022
            if use_mixup_cutmix:
                self.output_dir = const.result_mix_cut_2022
            
            self.output_dir= os.path.join(const.output_dir, self.imageType, str(numberOfimages))
            self.output_models= os.path.join(self.output_dir,"models")
            self.lossFile = os.path.join(self.output_dir,"loss.csv")
            
        if self.year == 2010:
            self.output_dir=const.result_2010
            self.output_ref =const.output_ref_2010
            if not all_bands:
                self.output_dir = const.result_2010_3
            if use_augmentation:
                self.output_dir = const.result_2010_aug
            if use_hyperparameter_tuning:
                self.output_dir = const.result_hp_2010
            if use_mixup_cutmix:
                self.output_dir = const.result_mix_cut_2010
                
            self.output_dir= os.path.join(const.output_dir, self.imageType, str(numberOfimages))
            
        self.result_path = os.path.join(self.output_dir, "result")
        
    def makeOutputDir(self):
        """
        Create output directories for results and models.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if self.year == 2022:
            os.makedirs(self.output_models, exist_ok=True)
            os.makedirs(self.result_path, exist_ok=True)
       
    
    def __str__(self):
        return (f"ExperimentConfig(image_type={self.image_type}, year={self.year}, "
                f"all_bands={self.all_bands}, hyperparameter_tuning={self.hyperparameter_tuning}, "
                f"use_augmentation={self.use_augmentation}, use_mixup_cutmix={self.use_mixup_cutmix}, "
                f"numberOfimages={self.numberOfimages})")