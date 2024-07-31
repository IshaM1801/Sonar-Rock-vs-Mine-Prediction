# Sonar Rock vs. Mine Prediction

This project aims to classify sonar returns into two categories: Rocks or Mines. The dataset used for this project is the Sonar dataset, which contains 60 attributes representing the energy levels of sonar returns.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The objective of this project is to develop a machine learning model that can accurately classify sonar returns as either rocks or mines. The dataset used for this project consists of 208 samples, each with 60 features.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/IshaM1801/Sonar-Rock-vs-Mine-Prediction.git
    cd Sonar-Rock-vs-Mine-Prediction
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the necessary libraries installed as specified in the `requirements.txt` file.
2. Run the Jupyter notebook `SONAR.ipynb` to train and evaluate the model.
3. Use the script `predict.py` to make predictions on new data.

## Dataset

The Sonar dataset is publicly available and can be found [here](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+%28sonar,+mines+vs.+rocks%29). The dataset contains 208 instances, each with 60 features and a class label (Rock or Mine).

## Model Training

The model is trained using a logistic regression classifier. The following steps are performed in the training process:

1. Data Preprocessing
2. Splitting the data into training and testing sets
3. Training the logistic regression model
4. Evaluating the model's performance

## Model Evaluation

The model's performance is evaluated using accuracy scores on both training and testing datasets. Overfitting and underfitting are checked, and appropriate measures such as regularization are applied to improve the model's generalization.

## Prediction

To make predictions on new data, use the following code snippet:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Example input data
input_data = (0.0286, 0.0453, 0.0277, 0.0174, 0.0384, 0.0990, 0.1201, 0.1833, 0.2105, 0.3039,
              0.2988, 0.4250, 0.6343, 0.8198, 1.0000, 0.9988, 0.9508, 0.9025, 0.7234, 0.5122,
              0.2074, 0.3985, 0.5890, 0.2872, 0.2043, 0.5782, 0.5389, 0.3750, 0.3411, 0.5067,
              0.5580, 0.4778, 0.3299, 0.2198, 0.1407, 0.2856, 0.3807, 0.4158, 0.4054, 0.3296,
              0.2707, 0.2650, 0.0723, 0.1238, 0.1192, 0.1089, 0.0623, 0.0494, 0.0264, 0.0081,
              0.0104, 0.0045, 0.0014, 0.0038, 0.0013, 0.0089, 0.0057, 0.0027, 0.0051, 0.0062)

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as you are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Load the trained model
model = LogisticRegression()
model.fit(X_train, Y_train)  # Ensure the model is trained

# Make a prediction
prediction = model.predict(input_data_reshaped)
print(prediction)
