# Heart Disease Prediction System

## Overview
A comprehensive machine learning system to predict the likelihood of heart disease based on patient data and health indicators. This project implements both traditional machine learning models and deep learning approaches using TensorFlow and scikit-learn to analyze health metrics and provide accurate predictions of heart disease risk.

## Features
- Deep learning model implementation using TensorFlow 2.11.0
- Multiple traditional machine learning models for comparison
- Data preprocessing pipeline for categorical and numerical features
- Multi-task learning approach with dual output heads
- Binary classification for heart disease prediction
- High accuracy prediction results
- Comprehensive data visualization capabilities
- Model performance comparison and evaluation
- Cross-validation and hyperparameter tuning

## Project Structure
```
heart-disease-prediction/
├── data/                               # Dataset directory
│   ├── raw/                            # Raw data files
│   │   └── heart_2020_cleaned.csv      # Original dataset
│   └── processed/                      # Processed data files
│       └── NewData.csv                 # Preprocessed dataset
├── notebooks/                          # Jupyter notebooks
│   ├── heart-disease-prediction.ipynb  # Main analysis notebook
│   └── test.ipynb                      # Model testing notebook
├── src/                                # Source code
│   ├── GetModel.py                     # Model training implementation
│   ├── tfhearthealth.py                # TensorFlow model definition
│   ├── predict.py                      # Prediction functionality
│   ├── toNewDataa.py                   # Data preprocessing
│   ├── mymodel.h5                      # Saved model weights
│   ├── NewData.csv                     # Preprocessed dataset
│   ├── heart_2020_cleaned.csv          # Original dataset
│   └── PredictResult.csv               # Model predictions
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

## Technologies Used
### Core Dependencies
- Python 3.8+
- TensorFlow 2.11.0
- NumPy 1.21.6
- Pandas 1.3.5
- Scikit-learn 1.0.2
- Matplotlib 3.5.3
- Seaborn 0.12.2
- Jupyter Notebook 6.5.7

### Machine Learning Models
- Deep Learning (TensorFlow)
  - Custom neural network architecture
  - Multi-task learning capabilities
- Traditional Machine Learning (scikit-learn)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Gaussian Naive Bayes
  - Multi-layer Perceptron (MLP)

### Additional Tools
- joblib (Model serialization)
- OneHotEncoder (Categorical feature encoding)
- StandardScaler (Feature scaling)
- make_column_transformer (Data preprocessing)

## Setup and Installation
1. Clone the repository:
```bash
git clone https://github.com/M0M0KO/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the Heart Disease Health Indicators Dataset from the CDC's BRFSS survey. The dataset includes various health indicators that might be relevant for predicting heart disease:

### Features
- Demographic features
  - Age (categorized into age groups)
  - Sex (Male/Female)
  - Race/Ethnicity
- Health Conditions
  - Diabetes status
  - High blood pressure
  - Stroke history
  - Asthma
  - Kidney Disease
  - Skin Cancer
- Lifestyle Factors
  - Smoking status
  - Alcohol consumption
  - Physical activity level
  - Sleep time (hours per day)
- Health Metrics
  - BMI (Body Mass Index)
  - Physical health (days)
  - Mental health (days)
  - Difficulty walking
  - General health assessment

## Model Architecture

### Deep Learning Model
The TensorFlow implementation includes:
- Input layer for multiple health indicators
- Dense hidden layers with ReLU activation
- Dual output heads:
  - Binary classification with softmax activation
  - Continuous prediction with linear activation
- Custom loss function combining categorical crossentropy and MSE
- Adam optimizer with learning rate 1e-3

### Traditional Models
Multiple traditional machine learning models are implemented for comparison:
1. Logistic Regression
   - Binary classification baseline
   - L2 regularization
2. K-Nearest Neighbors
   - Non-parametric learning
   - Distance-based classification
3. Support Vector Machine
   - Non-linear classification using RBF kernel
4. Random Forest
   - Ensemble learning with multiple decision trees
   - Feature importance analysis
5. Naive Bayes
   - Probabilistic classification
6. Multi-layer Perceptron
   - Neural network with scikit-learn

## Usage

### 1. Data Preprocessing
```python
python src/toNewDataa.py
```
This script:
- Handles missing values
- Converts categorical variables to numerical format
- Performs feature scaling
- Prepares the data for model training

### 2. Model Training
```python
python src/GetModel.py
```
Trains both deep learning and traditional models:
- Splits data into training and testing sets
- Performs feature scaling
- Trains multiple models for comparison
- Saves trained models for later use

### 3. Making Predictions
```python
python src/predict.py
```
Generates predictions using the trained models and saves results to `PredictResult.csv`.

## Model Performance
The system achieves:
- High prediction accuracy on test data
- Robust performance across different demographic groups
- Effective handling of both categorical and numerical features
- Reliable risk assessment capabilities

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC curve

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author
Jiatao Yan - [GitHub](https://github.com/M0M0KO)

## Last Updated
May 2022 