# Heart Disease Prediction System

## Overview
A comprehensive machine learning system to predict the likelihood of heart disease based on patient data and health indicators. This project implements a deep learning model using TensorFlow to analyze health metrics and provide accurate predictions of heart disease risk.

## Features
- Deep learning model implementation using TensorFlow 2.11.0
- Data preprocessing pipeline for categorical and numerical features
- Multi-task learning approach with dual output heads
- Binary classification for heart disease prediction
- High accuracy prediction results
- Comprehensive data visualization capabilities

## Project Structure
```
heart-disease-prediction/
├── data/                          # Dataset directory
│   ├── raw/                      # Raw data files
│   │   └── heart_2020_cleaned.csv  # Original dataset
│   └── processed/                # Processed data files
│       └── NewData.csv            # Preprocessed dataset
├── src/                          # Source code
│   ├── GetModel.py              # Model training implementation
│   ├── tfhearthealth.py         # TensorFlow model definition
│   ├── predict.py               # Prediction functionality
│   ├── toNewDataa.py           # Data preprocessing
│   ├── mymodel.h5              # Saved model weights
│   ├── NewData.csv             # Preprocessed dataset
│   ├── heart_2020_cleaned.csv  # Original dataset
│   └── PredictResult.csv       # Model predictions
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Technologies Used
- Python 3.8+
- TensorFlow 2.11.0
- NumPy 1.21.6
- Pandas 1.3.5
- Scikit-learn 1.0.2
- Matplotlib 3.5.3
- Seaborn 0.12.2
- Jupyter Notebook 6.5.7

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

- Demographic features (age, sex)
- Health conditions (diabetes, high blood pressure)
- Lifestyle factors (smoking, physical activity)
- General health indicators
- BMI and other physical measurements

## Usage

### 1. Data Preprocessing
```python
python src/toNewDataa.py
```
This script converts categorical variables into numerical format and prepares the data for model training.

### 2. Model Training
```python
python src/GetModel.py
```
Trains the deep learning model using the preprocessed data. The model architecture includes:
- Input layer for health indicators
- Multiple dense layers with ReLU activation
- Dual output heads for classification and regression
- Custom loss function combining categorical crossentropy and MSE

### 3. Making Predictions
```python
python src/predict.py
```
Generates predictions using the trained model and saves results to `PredictResult.csv`.

## Model Architecture
The system implements a neural network with:
- Input layer for multiple health indicators
- Dense hidden layers with ReLU activation
- Dual output heads:
  - Binary classification with softmax activation
  - Continuous prediction with linear activation
- Custom loss function combining categorical crossentropy and MSE
- Adam optimizer with learning rate 1e-3

## Model Performance
The system achieves:
- High prediction accuracy on test data
- Robust performance across different demographic groups
- Effective handling of both categorical and numerical features
- Reliable risk assessment capabilities

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author
Jiatao Yan - [GitHub](https://github.com/M0M0KO)

## Last Updated
May 2022 