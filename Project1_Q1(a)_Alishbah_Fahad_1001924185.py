#!/usr/bin/env python
# coding: utf-8

# #                                                             Q1(a)

# ### Importing Libraries and Data

# In[2]:


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


# ### Calculating Euclidean Distance

# In[3]:


def EDM(A,B):
    difference = A[:,None] - B[None,:]
    power_difference = np.power(np.abs(difference), 2)
    return power_difference.sum(axis=-1) ** (1 / 2)
print (EDM(Training_Data,Test_Data))


# ### Finding Closest Neighbor by using Euclidean Distance

# In[4]:


def closest_neighbor_EDM(Train, Test, k):
    distances = EDM(Training_Data,Test_Data)
    indices = np.argsort(distances, 0)
    distances = np.sort(distances,0)
    return indices[0:k, : ], distances[0:k, : ]
print ("Closest Neighbor when K=1:")
print (closest_neighbor_EDM(Training_Data,Test_Data, 1))
print ()
print ("Closest Neighbor when K=3:")
print (closest_neighbor_EDM(Training_Data,Test_Data, 3))
print ()
print ("Closest Neighbor when K=7:")
print (closest_neighbor_EDM(Training_Data,Test_Data, 7))


# ### Predicting the Gender by using Euclidean Distance

# In[5]:


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
print ("Prediction for K=1:", prediction_EDM(Training_Data,Label,Test_Data,1))
print ("Prediction for K=3:", prediction_EDM(Training_Data,Label,Test_Data,3))
print ("Prediction for K=7:", prediction_EDM(Training_Data,Label,Test_Data,7))


# ### Calculating Manhattan Distance

# In[6]:


def ManhattanDM(A, B):
    return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1]) + np.abs(A[:,2,None] - B[:,2])
print (ManhattanDM(Training_Data,Test_Data))


# ### Finding Closest Neighbor by using Manhattan Distance

# In[7]:


def closest_neighbor_Manhattan(Train, Test, k):
    distances_Manhattan = ManhattanDM(Training_Data,Test_Data)
    indices = np.argsort(distances_Manhattan, 0)
    distances_Manhattan = np.sort(distances_Manhattan,0)
    return indices[0:k, : ], distances_Manhattan[0:k, : ]
print ("Closest Neighbor when K=1:")
print (closest_neighbor_Manhattan(Training_Data,Test_Data, 1))
print ()
print ("Closest Neighbor when K=3:")
print (closest_neighbor_Manhattan(Training_Data,Test_Data, 3))
print ()
print ("Closest Neighbor when K=7:")
print (closest_neighbor_Manhattan(Training_Data,Test_Data, 7))


# ### Predicting the Gender by using Manhattan Distance

# In[8]:


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
print ("Prediction for K=1:", prediction_Manhattan(Training_Data,Label,Test_Data,1))
print ("Prediction for K=3:", prediction_Manhattan(Training_Data,Label,Test_Data,3))
print ("Prediction for K=7:", prediction_Manhattan(Training_Data,Label,Test_Data,7))


# ### Calculating Minkowski Distance

# In[9]:


def MinkowskiDM(A,B,p):
    difference = A[:,None] - B[None,:]
    power_difference = np.power(np.abs(difference), p)
    return power_difference.sum(axis=-1) **(1 / p)
print (MinkowskiDM(Training_Data,Test_Data,3))


# ### Finding Closest Neighbor by using Minkowski Distance

# In[10]:


def closest_neighbor_Minkowski(Train, Test, k):
    distances_Minkowski = MinkowskiDM(Training_Data,Test_Data,3)
    indices = np.argsort(distances_Minkowski, 0)
    distances_Minkowski = np.sort(distances_Minkowski,0)
    return indices[0:k, : ], distances_Minkowski[0:k, : ]
print ("Closest Neighbor when K=1:")
print (closest_neighbor_Minkowski(Training_Data,Test_Data, 1))
print ()
print ("Closest Neighbor when K=3:")
print (closest_neighbor_Minkowski(Training_Data,Test_Data, 3))
print ()
print ("Closest Neighbor when K=7:")
print (closest_neighbor_Minkowski(Training_Data,Test_Data, 7))


# ### Predicting the Gender by using Minkowski Distance

# In[11]:


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
print ("Prediction for K=1:", prediction_Minkowski(Training_Data,Label,Test_Data,1))
print ("Prediction for K=3:", prediction_Minkowski(Training_Data,Label,Test_Data,3))
print ("Prediction for K=7:", prediction_Minkowski(Training_Data,Label,Test_Data,7))


# In[12]:


print ("For K=1, Metric=Euclidean -->",prediction_EDM(Training_Data,Label,Test_Data,1))
print ("For K=1, Metric=Manhattan -->",prediction_Manhattan(Training_Data,Label,Test_Data,1))
print ("For K=1, Metric=Minkowski -->",prediction_Minkowski(Training_Data,Label,Test_Data,1))
print ('---------------------------------------------------')
print ("For K=3, Metric=Euclidean -->",prediction_EDM(Training_Data,Label,Test_Data,3))
print ("For K=3, Metric=Manhattan -->",prediction_Manhattan(Training_Data,Label,Test_Data,3))
print ("For K=3, Metric=Minkowski -->",prediction_Minkowski(Training_Data,Label,Test_Data,3))
print ('---------------------------------------------------')
print ("For K=7, Metric=Euclidean -->",prediction_EDM(Training_Data,Label,Test_Data,7))
print ("For K=7, Metric=Manhattan -->",prediction_Manhattan(Training_Data,Label,Test_Data,7))
print ("For K=7, Metric=Minkowski -->",prediction_Minkowski(Training_Data,Label,Test_Data,1))
print ('---------------------------------------------------')


# ###### I got the same prediction from all distances, even though the distance values were slightly different.
# ###### Prediction for K=1: [' W' ' W' ' W' ' W']
# ###### Prediction for K=3: [' W' ' M' ' W' ' W']
# ###### Prediction for K=7: [' W' ' M' ' W' ' W']
