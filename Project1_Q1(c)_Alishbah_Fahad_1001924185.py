#!/usr/bin/env python
# coding: utf-8

# #                                                             Q1(c)

# ### Importing Libraries and Data

# In[1]:


import numpy as np
import math
from math import sqrt
from math import exp
from math import pi
def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input
def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np

training = r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Program Data.txt"
Training_Data = readFile(training)

print("Training Data:")
print(Training_Data)


# ### Replacing 'W' and 'M' to '1' and '0' respectively

# In[2]:


for i in Training_Data:
    if i[3]=='W':
        i[3]=i[3].replace('W','1')
        i[3]=int(i[3])
    else:
        i[3]=i[3].replace('M','0')
        i[3]=int(i[3])
Training_Data=Training_Data.astype(float)


# ### Find the min and max values for each column

# In[3]:


def traindata_minmax(traindata):
    minmax = list()
    for i in range(len(traindata[0])):
        col_values = [row[i] for row in traindata]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# ### Rescale traindata columns to the range 0-1

# In[4]:


def normalize_traindata(traindata, minmax):
    for row in traindata:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# ### Split a dataset into k folds

# In[5]:


from random import randrange
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# ### Calculate accuracy percentage

# In[6]:


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# ### Evaluate an algorithm using a cross validation split

# In[7]:


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        def remove_values_from_list(train_set, fold):
            return [value for value in train_set if value != fold]
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# ### Calculate the Euclidean distance between two vectors

# In[8]:


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


# ### Locate the most similar neighbors

# In[9]:


def get_neighbors(train, test_row, K):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(K):
        neighbors.append(distances[i][0])
    return neighbors


# ### Make a prediction with neighbors

# In[10]:


def predict_classification(train, test_row, K):
    neighbors = get_neighbors(train, test_row, K)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# ### KNN Algorithm

# In[11]:


def KNN_MODEL(train, test, K):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, K)
        predictions.append(output)
    return(predictions)


# ### Results

# In[12]:


scores1 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 1)
Accuracy1 = (sum(scores1)/float(len(scores1)))

scores3 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 3)
Accuracy3 = (sum(scores3)/float(len(scores3)))

scores5 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 5)
Accuracy5 = (sum(scores5)/float(len(scores5)))

scores7 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 7)
Accuracy7 = (sum(scores7)/float(len(scores7)))

scores9 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 9)
Accuracy9 = (sum(scores9)/float(len(scores9)))

scores11 = evaluate_algorithm(Training_Data, KNN_MODEL, 120, 11)
Accuracy11 = (sum(scores11)/float(len(scores11)))


# In[13]:


print('For K=1, Accuracy: %.3f%%' % Accuracy1)
print('For K=3, Accuracy: %.3f%%' % Accuracy3)
print('For K=5, Accuracy: %.3f%%' % Accuracy5)
print('For K=7, Accuracy: %.3f%%' % Accuracy7)
print('For K=9, Accuracy: %.3f%%' % Accuracy9)
print('For K=11, Accuracy: %.3f%%' % Accuracy11)


# I obtained the best result for K = 1 because the accuracy is 100%.
