# Temporal transferability of deep learning models for segmenting agricultural fields from satellite images

# Agricultural Field Boundary Detection Using Satellite Imagery

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

Using the DECODE methodology, we performed extensive validations on the outputs generated from the 2010 imagery. The comparison with the 2022 data allowed us to measure the temporal shift in agricultural practices and land use change effectively. This validation process utilized metrics such as IoU, Boundary F1-score, Shape smilarity index, ensuring that our model predictions align accurately with historical changes observed in the satellite images.

## Getting Started

This project relies on a specific Python environment managed with Anaconda. Below are the steps to set up the environment and install all required libraries to ensure compatibility and reproducibility of the results. To run this project on your local machine, follow these steps:

### Prerequisites

- Anaconda
- Python 3.6+
- MXNet
- QGIS

### Installing

Clone the repository:
git clone https://github.com/SajaniJoshi/MasterThesis.git

Install required Python packages:
pip install -r requirements.txt

## Running the tests

To run tests, use the following command:

- TODO

## Built With

- [Python](https://python.org) - The programming language used.
- [MXNet](https://mxnet.apache.org) - Deep learning framework used.
- [QGIS](https://qgis.org) - Visualization software used.

## Authors

- **Sajani Joshi**

## License

- TODO how do i get the license
  This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to the Brandenburg data providers.
- Acknowledgment to research team members and advisors.

## References

- Brandenburg Data Source [[WikipediaBrandenburg](https://de.wikipedia.org/wiki/Brandenburg)]
- Agricultural Practices [MLEUVPublications]
- Model Comparison Study [https://link.springer.com/article/10.1007/s41064-023-00247-x]
- DECODE Method for Transferability [https://www.mdpi.com/2072-4292/13/11/2197]
