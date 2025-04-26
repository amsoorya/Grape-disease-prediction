# Grape Disease Detection using Machine Learning

This repository contains machine learning models for the detection and classification of grape diseases using images. The project focuses on identifying four key categories: Black Rot, ESCA, Healthy, and Leaf Blight.

## Dataset

The dataset used in this project is the [Augmented Grape Disease Detection Dataset](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset) from Kaggle, which contains:

- Black Rot: 3000 images
- ESCA: 3000 images
- Healthy: 3000 images
- Leaf Blight: 3000 images

The dataset provides a balanced representation of different grape leaf conditions, making it suitable for training robust classification models.

## Project Structure

```
├── data/                      # Dataset directory
│   ├── Black_Rot/             # Black Rot images
│   ├── ESCA/                  # ESCA disease images
│   ├── Healthy/               # Healthy leaf images
│   └── Leaf_Blight/           # Leaf Blight images
├── models/                    # Saved model files
├── notebooks/                 # Jupyter notebooks for exploration and analysis
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data preprocessing functions
│   ├── feature_extraction.py  # Feature extraction code
│   ├── model_training.py      # Model training scripts
│   └── evaluation.py          # Model evaluation utilities
├── results/                   # Results and visualizations
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Features

- Image preprocessing and augmentation for enhanced model training
- Feature extraction and selection to identify the most important image characteristics
- Multiple machine learning models (Random Forest, Gradient Boosting, SVM)
- Comprehensive model evaluation with metrics like accuracy, precision, recall, and F1-score
- Visualization of results through confusion matrices, ROC curves, and precision-recall curves

## Installation

1. Clone this repository:
```bash
git clone https://github.com/amsoorya/grape-disease-detection.git
cd grape-disease-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset) and extract it to the `data/` directory.

## Usage

### Training Models

To train the models:

```bash
python src/model_training.py
```

This will:
- Load and preprocess the image data
- Split the data into training and testing sets (80/20 split)
- Train multiple classifiers (Random Forest, Gradient Boosting, SVM)
- Save the trained models to the `models/` directory

### Evaluating Models

To evaluate the trained models:

```bash
python src/evaluation.py
```

This will generate performance metrics and visualizations for each model, including:
- Confusion matrices
- ROC curves 
- Precision-recall curves
- Model comparison charts

## Results

The models achieve strong classification performance across all four grape disease categories. The best-performing model achieves an accuracy of approximately X% on the test set.

Key findings:
- Feature selection identified X key features that are most important for disease classification
- The Random Forest classifier showed the best overall performance

## Future Work

- Development of a web application for real-time disease detection
- Deployment of models on mobile devices for in-field use
- Integration with automated vineyard monitoring systems
- Expansion to include additional grape diseases and conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)
- Special thanks to all contributors and the viticulture community

## Contact

- **Contributor**: Jaya Soorya
- **GitHub**: [amsoorya](https://github.com/amsoorya)
- **Email**: amjayasoorya@gmail.com
