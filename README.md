# K-Nearest Neighbors (KNN) Classification Project

## Overview
This project implements the K-Nearest Neighbors (KNN) algorithm for gender classification based on height, weight, and age. Various distance metrics, including Euclidean, Manhattan, and Minkowski distances, are applied to classify individuals into gender categories (`Male` or `Female`). The project evaluates the model's performance using different values of `K` and identifies the most effective feature set for accurate predictions.

---

## Objectives
1. Implement KNN using Euclidean, Manhattan, and Minkowski distance metrics.
2. Evaluate the impact of different `K` values on classification accuracy.
3. Analyze feature importance by removing individual features and assessing model performance.
4. Use cross-validation to evaluate model robustness.

---

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - `numpy`: For numerical computations.
  - `math`: For mathematical operations.

---

## Dataset Details
- **Training Data** (`Training_Data.txt` and `Training_Data.csv`):
  - Contains height, weight, age, and gender labels (`M` for Male, `W` for Female).
  - Example:
    ```
    (( 1.6530190426733, 72.871146648479, 24), W)
    (( 1.6471384909498, 72.612785314988, 34), W)
    ```
- **Test Data** (`Test_Data.txt` and `Test_Data.csv`):
  - Contains height, weight, and age without labels for prediction.
  - Example:
    ```
    (1.62065758929, 59.376557437583, 32)
    ```

---

## Key Tasks

### **1. Implement KNN with Multiple Distance Metrics**
- Implemented KNN using the following distance metrics:
  1. **Euclidean Distance**:
     \[
     \text{distance} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
     \]
  2. **Manhattan Distance**:
     \[
     \text{distance} = \sum_{i=1}^{n} |x_i - y_i|
     \]
  3. **Minkowski Distance**:
     \[
     \text{distance} = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}
     \]

### **2. Evaluate Model for Different `K` Values**
- Tested the model with various values of `K` (e.g., 1, 3, 5, 7).
- Observed classification accuracy for each `K` using cross-validation.

### **3. Feature Importance Analysis**
- Removed features (e.g., age) to evaluate their impact on model performance.
- Discovered that removing age improved accuracy, indicating height and weight are stronger predictors of gender.

---

## Results

### **Performance Metrics**
| Distance Metric | K=1  | K=3  | K=5  | K=7  | K=9  |
|-----------------|-------|------|------|------|------|
| Euclidean       | 100%  | 95%  | 92%  | 90%  | 88%  |
| Manhattan       | 98%   | 96%  | 93%  | 91%  | 89%  |
| Minkowski (p=3) | 99%   | 96%  | 93%  | 90%  | 89%  |

### **Feature Importance**
- Removing `Age` improved model accuracy across all metrics and `K` values.
- Height and weight were found to be the most significant features for predicting gender.
