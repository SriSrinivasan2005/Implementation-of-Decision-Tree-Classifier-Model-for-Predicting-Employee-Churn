# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the employee dataset using pandas.

2. Perform basic data checks and preprocess data (handle null values and encode salary using LabelEncoder).

3. Select relevant features (X) and target variable (y = left).

4. Split the dataset into training and testing sets using train_test_split.

5. Create a Decision Tree Classifier with entropy criterion and limited depth, then train it using training data.

6. Predict employee churn on test data and evaluate performance using accuracy and confusion matrix.

7. Visualize results (confusion matrix, decision tree) and use the model for new employee prediction.

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv("/content/Employee.csv")

# Basic checks
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

# Encode salary column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features
X = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]
X.head()
y = data["left"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

# Create Decision Tree model (prevent overfitting)
dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

# Train model
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Feature Importance
feature_importance = pd.Series(dt.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

# Example Prediction
sample_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample_employee)

if prediction[0] == 1:
    print("Prediction: Employee will leave")
else:
    print("Prediction: Employee will stay")

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()

```

## Output:
### data.head():
<img width="1253" height="193" alt="image" src="https://github.com/user-attachments/assets/ef4aeb52-a883-4e60-b2d7-20fdcfd2dfd3" />

### data.info():
<img width="563" height="371" alt="image" src="https://github.com/user-attachments/assets/437dde1f-a20f-4b7b-aaa8-41505491b4df" />

### data.isnull().sum():
<img width="319" height="245" alt="image" src="https://github.com/user-attachments/assets/e24e29ac-daf6-46d1-9573-3acee398bcf1" />

### data["left"].value_counts():
<img width="289" height="85" alt="image" src="https://github.com/user-attachments/assets/432c2c0c-f17a-49f4-80a4-1851b83d41d9" />

### X.head():
<img width="1156" height="208" alt="image" src="https://github.com/user-attachments/assets/4494f386-82c2-40a1-b550-e8b304088b4a" />

### Accuracy:
<img width="189" height="43" alt="image" src="https://github.com/user-attachments/assets/bda26c4e-f1b2-49ae-aa05-3401661b718c" />

### Feature Importance:
<img width="382" height="221" alt="image" src="https://github.com/user-attachments/assets/fe28bbc2-c9b7-4b99-bb46-cbd424f52c6c" />

### Plot Decision Tree:
<img width="1289" height="752" alt="image" src="https://github.com/user-attachments/assets/16f74c64-f5db-4406-8834-77c36c0ca1c5" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
