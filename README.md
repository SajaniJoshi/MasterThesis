# Temporal transferability of deep learning models for segmenting agricultural fields from satellite images

This project focuses on using deep learning models to segment agricultural field boundaries in the Brandenburg area from satellite imagery. The study utilizes historical and recent data to assess model effectiveness and transferability over time.

## Study Area

The study area is Brandenburg, a federal state in northeastern Germany known for its extensive agricultural activities. The region spans approximately 29,654.16 sq. km and features a diverse landscape that is ideal for testing advanced image segmentation technologies.

## Data Acquisitions

We used multi-spectral images from Landsat 5, 7, 8, and 9, covering two time periods: 2010 and 2022. The data includes VNIR and NDV images, essential for analyzing vegetation and crop health.

## Data Preparation

Data quality was ensured through checks for common image issues. Images were normalized to scale pixel values between 0 and 1, facilitating consistent training performance across varying image conditions.

## Model Training

Training was conducted using the Fractal ResUNet architecture on an NVIDIA GeForce GTX 1650 GPU. Configurations varied across different band setups and included techniques like data augmentation and hyperparameter tuning to optimize performance.

## Model Outputs and Validation

Outputs include extent, boundary, and distance maps. Validation metrics like IoU and Boundary F1-score were used to evaluate model accuracy. Visual inspections were also conducted to align predictions with ground truths.

## Application to Historical Data

The model's ability to handle historical imagery was tested by applying it to 2010 data, with validation results indicating strong temporal transferability.

### DECODE Method for Transferability

The DECODE method—**Detect, Consolidate, Delineate**—employs advanced convolutional neural networks alongside hierarchical watershed segmentation. This technique is pivotal in managing temporal variations in satellite imagery and is detailed in Waldner's publication and [GitHub repository](https://github.com/waldnerf/decode). This method systematically enhances the robustness and accuracy of field boundary detection across years, proving crucial for our temporal analysis.

This approach not only improved our model's accuracy in segmenting field boundaries from satellite imagery but also provided a scalable solution to assess and quantify temporal changes in the landscape, which is essential for long-term environmental and agricultural planning.

### Validation of Historical Data

we performed extensive validations on the outputs generated from the 2010 imagery. The comparison with the 2022 data allowed us to measure the temporal shift in agricultural practices and land use change effectively. This validation process utilized metrics such as IoU, Boundary F1-score, Shape smilarity index, ensuring that our model predictions align accurately with historical changes observed in the satellite images.

## Getting Started

This project relies on a specific Python environment managed with Anaconda. Below are the steps to set up the environment and install all required libraries to ensure compatibility and reproducibility of the results. To run this project on your local machine, follow these steps:

### Prerequisites

- Anaconda
- Python 3.6+
- MXNet
- QGIS
- CUDA Toolkit 11.1

### Installing

- Install [ANACONDA](https://www.anaconda.com/download)
- Install [QGIS](https://qgis.org/)
- If you are using GPU
  - Install [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)
  - Set the path of CUDA in Environment Variables

Clone the repository:

- git clone https://github.com/waldnerf/decode.git
- git clone https://github.com/SajaniJoshi/MasterThesis.git

Install required Python packages using Anaconda:

- Create a new environment with python version 3.6.13
- In this new enviorment
  - Install Mxnet GPU or Mxnet 1.2.1
  - Install numpy, pandas, geopandas, matplotlib, pillow, pyproj, rasterio, Shapely, tqdm

## Running the Tests

To run the tests, follow these steps:

1. **Set Input Paths**  
   Update the image and mask input paths in `const.py` located at `examples/const.py`.

2. **Open the Notebook**  
   Launch `myDecode.ipynb` located in `examples/myDecode.ipynb`.

3. **Select Environment**  
   Change the kernel and select the newly created environment.

4. **Configure the Experiment**  
   Initialize `ExperimentConfig` with your desired settings.  
   For example, to apply image augmentation, set `use_augmentation = True`.

5. **Run Cells Sequentially**  
   Execute each cell in order to start the experiment.

6. **Training Outputs**

   - Model checkpoints are saved after each epoch.
   - Training and validation losses are saved a CSV files.

7. **Validation Outputs 2022 and 2010**

- Predicted images such as Extent, Boundary, Distance, Instant segmented images are saved in Tiff format and one additon vector shape file are created for each tile.

## Running the Validation Metrics

To evaluate the validation metrics, follow these steps:

1. **Set Paths**  
   Update the correct paths in [`const.py`](metrics/const.py).

2. **Run the Notebook**  
   Open and run each cell in [`metrics_result.ipynb`](metrics/metrics_result.ipynb) sequentially.

3. **Metrics Computation**  
   The following metrics are calculated:

   - Intersection over Union (IoU)
   - F1 Boundary Score
   - Shape Similarity
   - Temporal Boundary Shift

4. **Results**  
   The computed metrics are saved as CSV and Histrogram as PNG files in the following directories:
   - `metrics/res_2022`
   - `metrics/res_2010`

## Visualizing and Comparing Results

To visualize and compare the results, follow these steps:

1. **Set Paths**  
   Update the required paths in [`const.py`](visualize/const.py).

2. **Run the Notebook**  
   Open and execute each cell in [`visualize.ipynb`](visualize/visualize.ipynb).

3. **Output**  
   The notebook will generate comparison images that include:

   - Original Image
   - Mask
   - Extent
   - Boundary
   - Distance
   - Instance Segmented Image
   - Agreement

   All components are displayed side by side to facilitate easy comparison for both the 2022 and 2010 datasets.  
   The generated images will be saved in the `visualize/result_img` directory.

## Built With

- [Python](https://python.org) - The programming language used.
- [MXNet](https://mxnet.apache.org) - Deep learning framework used.
- [QGIS](https://qgis.org) - Visualization software used.

## Authors

- **Sajani Joshi**

## License

- TODO how do i get the license

## Acknowledgments

- Thanks to the Brandenburg data providers.
- Acknowledgment to Dr. Michael Teichmann, Prof. Dr. habil. Julien Vitay (Technische Universität Chemnitz), and Dr. Gideon Okpoti Tetteh (Thünen Institute) for their exceptional guidance, feedback, and support.

## References

- [Brandenburg Data Source](https://de.wikipedia.org/wiki/Brandenburg)
- [Agricultural Practices](https://mleuv.brandenburg.de/mleuv/en/about-us/public-relations/publications/)
- [Model Comparison Study](https://link.springer.com/article/10.1007/s41064-023-00247-x)
- [DECODE](https://www.mdpi.com/2072-4292/13/11/2197) Method for Transferability
