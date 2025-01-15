# Car Data Analysis and Machine Learning Model

## Solution
<img width="1470" alt="Screenshot 2025-01-15 at 1 47 44 AM" src="https://github.com/user-attachments/assets/6fc2b842-86b9-4729-9bf7-94d96cf18b60" />



## Overview

This project provides an exploratory data analysis (EDA) and a machine learning classification model for predicting car acceptability based on various attributes. It integrates data manipulation, visualization, and machine learning using Python libraries such as pandas, matplotlib, and scikit-learn.

## Files

1. **car\_data\_ml\_analysis.py**: This script implements a machine learning model using the Decision Tree Classifier.
2. **Original Data Processing and Visualization Script**: The original user-provided script performs data cleaning, visualization, and exploratory analysis.

## Instructions

### Data

The dataset used is a CSV file (`cardata.csv`) containing car attributes:

- `buying`: Buying price (e.g., low, med, high, vhigh)
- `maint`: Maintenance cost
- `doors`: Number of doors (e.g., 2, 3, 4, 5more)
- `persons`: Number of persons the car can accommodate (e.g., 2, 4, more)
- `lug_boot`: Luggage boot size (small, med, big)
- `safety`: Safety level (low, med, high)
- `class`: Acceptability of the car (unacc, acc, good, vgood)

### 1. Data Processing and Visualization

#### Steps

1. **Load and clean data**: Missing values and duplicates are removed.
2. **Replace values for readability**:
   - `doors` replaces '5more' with 5.
   - `persons` replaces 'more' with 6.
   - `class` replaces 'unacc' with 'unacceptable' and 'acc' with 'acceptable'.
3. **Create new features**:
   - `type`: Classifies cars based on `doors` into 'coupe', 'hatchback', 'sedan', etc.
4. **Visualize data**:
   - Bar plot for car type distribution.
   - Line plot comparing maintenance prices for four-door cars.
   - Pie charts for buying price analysis and acceptability.

#### Outputs

- Visualizations provide insights into car types, maintenance costs, and acceptability.
- Statistical summaries include mean, median, mode, and standard deviation for numeric features.

### 2. Machine Learning Model

#### Steps

1. **Data Preparation**:
   - Encode categorical variables using `LabelEncoder`.
   - Split data into features (X) and target (y).
   - Standardize data using `StandardScaler`.
2. **Model Training**:
   - Train a Decision Tree Classifier on the training set.
3. **Model Evaluation**:
   - Use confusion matrix, classification report, and accuracy score to evaluate performance.
   - Output feature importances to identify the most influential variables.

#### Outputs

- **Confusion Matrix**: Shows the count of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Accuracy Score**: Measures overall model accuracy.
- **Feature Importance**: Identifies key attributes that influence car acceptability.

---

## How to Run

1. Clone the repository or download the files to your local machine.
2. Ensure that `cardata.csv` is in the same directory as the scripts.
3. Install the necessary Python libraries:
   ```sh
   pip install pandas numpy matplotlib scikit-learn
   ```
4. Run the provided scripts:
   - For data processing and visualization:
     ```sh
     python data_visualization_script.py
     ```
   - For the machine learning model:
     ```sh
     python car_data_ml_analysis.py
     ```

---

## Summary

This project demonstrates:

- **Data preprocessing**: Cleaning, encoding, and scaling.
- **Exploratory visualization**: Insightful visual analysis.
- **Machine learning classification**: Building and evaluating a predictive model.

