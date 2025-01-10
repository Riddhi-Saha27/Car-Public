import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset from a CSV file
# The dataset contains car evaluation data with various attributes like buying cost, maintenance, etc.
df = pd.read_csv("cardata.csv", nrows=1040)

# Drop missing values to ensure clean data
# Drop duplicate rows to avoid redundant information
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Rename columns for easier access
# The columns represent different car features and the class (acceptability)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Replace 'class' values to improve readability
# 'unacc' becomes 'unacceptable', 'acc' becomes 'acceptable'
df['class'] = df['class'].replace({'unacc': 'unacceptable', 'acc': 'acceptable'})

# Replace '5more' with 5 for easier numeric manipulation
# Convert 'doors' to integer
# Similarly, replace 'more' with 6 in 'persons' and convert it to integer
df['doors'] = df['doors'].replace('5more', 5).astype(int)
df['persons'] = df['persons'].replace('more', 6).astype(int)

# Encode categorical columns into numeric values using LabelEncoder
# This allows machine learning models to process these columns
label_encoders = {}  # Dictionary to store encoders for later use
categorical_columns = ['buying', 'maint', 'lug_boot', 'safety', 'class']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode each column
    label_encoders[col] = le  # Store the encoder for potential inverse transformation

# Separate the features (X) and target (y)
# 'class' is the target variable to predict car acceptability
X = df.drop('class', axis=1)  # Features (all columns except 'class')
y = df['class']  # Target (class)

# Split the data into training and testing sets
# 70% training data, 30% testing data for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature data to improve model performance
# StandardScaler scales features to have mean 0 and variance 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Decision Tree Classifier to predict car acceptability
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model using confusion matrix, classification report, and accuracy score
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Output example:
# [[66  1]   -> 66 correctly classified as 'unacceptable', 1 misclassified
#  [ 0 245]] -> 0 misclassified as 'unacceptable', 245 correctly classified

print("Classification Report:\n", classification_report(y_test, y_pred))
# Detailed performance metrics like precision, recall, F1-score for each class
# Precision: Proportion of correct positive predictions
# Recall: Proportion of actual positives correctly predicted
# F1-score: Harmonic mean of precision and recall

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
# Example output: Accuracy Score: 1.00 (100% accuracy in this example)

# Analyze feature importance to understand which attributes influence the model most
feature_importances = clf.feature_importances_
columns = X.columns
for feature, importance in zip(columns, feature_importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")
# Output example:
# Feature: safety, Importance: 0.2663 -> 'safety' has the highest influence on predictions
