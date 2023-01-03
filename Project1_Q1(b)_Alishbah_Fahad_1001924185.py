#!/usr/bin/env python
# coding: utf-8

# #                                                             Q1(b)

# ### Importing Libraries and Data

# In[1]:


import numpy as np
Training_Data=np.genfromtxt(r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Training Data.csv", delimiter=',', skip_header=1, usecols = (0, 1, 2))
Label=np.genfromtxt(r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Training Data.csv", delimiter=',', skip_header=1, usecols = 3, dtype=str)
Test_Data=np.genfromtxt(r"C:\Users\alish\OneDrive\Documents\Alishbah\CSE6363_Machine Learning\Project-1\axf4185_project_1\dataset\Test Data.csv", delimiter=',', skip_header=1, usecols = (0, 1, 2))
print("Training Data:")
print(Training_Data)
print()
print("Label:")
print(Label)
print()
print("Test Data:")
print(Test_Data)


# ### KNN MODEL using Euclidean Distance

# In[2]:


def KNN_MODEL_Euclidean(Train, Test, k):
### Calculating Euclidean Distance
    def EDM(A,B):
        difference = A[:,None] - B[None,:]
        power_difference = np.power(np.abs(difference), 2)
        return power_difference.sum(axis=-1) ** (1 / 2)
### Finding Closest Neighbor by using Euclidean Distance
    def closest_neighbor_EDM(Train, Test, k):
        distances = EDM(Training_Data,Test_Data)
        indices = np.argsort(distances, 0)
        distances = np.sort(distances,0)
        return indices[0:k, : ], distances[0:k, : ]
### Predicting the Gender by using Euclidean Distance
    def prediction_EDM(Train,Target,Test,k):
        indices, distances = closest_neighbor_EDM(Training_Data, Test_Data, k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
    return EDM(Training_Data,Test_Data),closest_neighbor_EDM(Training_Data,Test_Data, k),prediction_EDM(Training_Data,Label,Test_Data,k)


# ### KNN MODEL using Manhattan Distance

# In[3]:


def KNN_MODEL_Manhattan(Train, Test, k):
### Calculating Manhattan Distance
    def ManhattanDM(A, B):
        return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1]) + np.abs(A[:,2,None] - B[:,2])
### Finding Closest Neighbor by using Manhattan Distance    
    def closest_neighbor_Manhattan(Train, Test, k):
        distances_Manhattan = ManhattanDM(Training_Data,Test_Data)
        indices = np.argsort(distances_Manhattan, 0)
        distances_Manhattan = np.sort(distances_Manhattan,0)
        return indices[0:k, : ], distances_Manhattan[0:k, : ]
### Predicting the Gender by using Manhattan Distance    
    def prediction_Manhattan(Train,Target,Test,k):
        indices, distances_Manhattan = closest_neighbor_Manhattan(Train,Test,k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
    return ManhattanDM(Training_Data, Test_Data),closest_neighbor_Manhattan(Training_Data, Test_Data, k), prediction_Manhattan(Training_Data,Label,Test_Data,k)


# ### KNN MODEL using Minkowski Distance

# In[4]:


def KNN_MODEL_Minkowski(Train, Test, k):
### Calculating Minkowski Distance  
    def MinkowskiDM(A,B,p):
        difference = A[:,None] - B[None,:]
        power_difference = np.power(np.abs(difference), p)
        return power_difference.sum(axis=-1) **(1 / p)
### Finding Closest Neighbor by using Minkowski Distance    
    def closest_neighbor_Minkowski(Train, Test, k):
        distances_Minkowski = MinkowskiDM(Training_Data,Test_Data,3)
        indices = np.argsort(distances_Minkowski, 0)
        distances_Minkowski = np.sort(distances_Minkowski,0)
        return indices[0:k, : ], distances_Minkowski[0:k, : ]
### Predicting the Gender by using Minkowski Distance 
    def prediction_Minkowski(Train,Target,Test,k):
        indices, distances_Minkowski = closest_neighbor_Minkowski(Train,Test,k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
    return MinkowskiDM(Training_Data,Test_Data,3),closest_neighbor_Minkowski(Training_Data, Test_Data, k), prediction_Minkowski(Training_Data,Label,Test_Data,k)


# ### KNN MODEL using Euclidean, Manhattan, and Minkowski Distances

# In[5]:


def KNN_MODEL(Train, Test, k):
### Calculating Euclidean Distance
    def EDM(A,B):
        difference = A[:,None] - B[None,:]
        power_difference = np.power(np.abs(difference), 2)
        return power_difference.sum(axis=-1) ** (1 / 2)
### Finding Closest Neighbor by using Euclidean Distance
    def closest_neighbor_EDM(Train, Test, k):
        distances = EDM(Training_Data,Test_Data)
        indices = np.argsort(distances, 0)
        distances = np.sort(distances,0)
        return indices[0:k, : ], distances[0:k, : ]
### Predicting the Gender by using Euclidean Distance
    def prediction_EDM(Train,Target,Test,k):
        indices, distances = closest_neighbor_EDM(Train, Test, k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
### Calculating Manhattan Distance
    def ManhattanDM(A, B):
        return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1]) + np.abs(A[:,2,None] - B[:,2])
### Finding Closest Neighbor by using Manhattan Distance    
    def closest_neighbor_Manhattan(Train, Test, k):
        distances_Manhattan = ManhattanDM(Training_Data,Test_Data)
        indices = np.argsort(distances_Manhattan, 0)
        distances_Manhattan = np.sort(distances_Manhattan,0)
        return indices[0:k, : ], distances_Manhattan[0:k, : ]
### Predicting the Gender by using Manhattan Distance    
    def prediction_Manhattan(Train,Target,Test,k):
        indices, distances_Manhattan = closest_neighbor_Manhattan(Train,Test,k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
### Calculating Minkowski Distance  
    def MinkowskiDM(A,B,p):
        difference = A[:,None] - B[None,:]
        power_difference = np.power(np.abs(difference), p)
        return power_difference.sum(axis=-1) **(1 / p)
### Finding Closest Neighbor by using Minkowski Distance    
    def closest_neighbor_Minkowski(Train, Test, k):
        distances_Minkowski = MinkowskiDM(Training_Data,Test_Data,3)
        indices = np.argsort(distances_Minkowski, 0)
        distances_Minkowski = np.sort(distances_Minkowski,0)
        return indices[0:k, : ], distances_Minkowski[0:k, : ]
### Predicting the Gender by using Minkowski Distance 
    def prediction_Minkowski(Train,Target,Test,k):
        indices, distances_Minkowski = closest_neighbor_Minkowski(Train,Test,k)
        Target = Target.flatten()
        rows, columns = indices.shape
        prediction = list()
        for j in range(columns):
            T = list()
            for i in range(rows):
                cell = indices[i][j]
                T.append(Target[cell])
            prediction.append(max(T,key=T.count))
        prediction=np.array(prediction)
        return prediction
    return EDM(Training_Data,Test_Data),closest_neighbor_EDM(Training_Data,Test_Data, k),prediction_EDM(Training_Data,Label,Test_Data,k),ManhattanDM(Training_Data, Test_Data),closest_neighbor_Manhattan(Training_Data, Test_Data, k), prediction_Manhattan(Training_Data,Label,Test_Data,k), MinkowskiDM(Training_Data,Test_Data,3),closest_neighbor_Minkowski(Training_Data, Test_Data, k), prediction_Minkowski(Training_Data,Label,Test_Data,k)

