#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt


# In[76]:


# opening the 'my_csv' file to read its contents
with open('E:\\_Github\\ML_linear classifiers\\buildingDataset.csv', newline = '', encoding="utf-8-sig") as file:
    reader = csv.reader(file,
                        quoting = csv.QUOTE_ALL,
                        delimiter = ' ')
     
    # storing all the rows in an output list
    output = []
    for row in reader:
        output.append(row[:][0].split(","))   
dataset = np.asarray(output[1:]).astype(float)


def DataOrganize(Dataset):
    dataset_features = Dataset[:,1:-1]
    dataset_labels = Dataset[:,-1][np.newaxis].T
    return dataset_features, dataset_labels

def ShowGraphAnimation (Result_th, Result_thZero, Dataset, iteration):
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    print(np.where(dataset_labels[:,0] == 1))
    
    positive_features = dataset_features[np.where(dataset_labels[:,0] == 1)]
    negative_features = dataset_features[np.where(dataset_labels[:,0] == -1)]
    
    xOne = np.linspace(-1000,3500)
    xTwo = -1*(xOne*Result_th[1,0] + result_thZero[0])/Result_th[0,0]
    
    plt.clf()
    
    plt.plot(positive_features[:,0], positive_features[:,1], 'ro')
    plt.plot(negative_features[:,0], negative_features[:,1], 'bo')
    plt.plot(xOne, xTwo)
    plt.axis([-1000, 3500, -0.5, 1.5])
    plt.savefig("E:\_Github\ML_linear classifiers\images\lcResult"+ str(iteration) +'.png')
    

def Perceptron (iteration, Dataset):   
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    th = np.zeros(np.shape(dataset_features)[1])[np.newaxis].T
    thZero = 0
    
    for t in range(iteration):
        changed = False
        for i in range(dataset_labels.shape[0]):
            if dataset_labels[i]*(np.dot(initial_th.T, dataset_features[i]) + initial_thZero) <= 0:
                th += (dataset_features[i]*dataset_labels[i])[np.newaxis].T
                thZero += dataset_labels[i]
                
                ShowGraphAnimation(th, thZero, Dataset, (t+1)*(i+1))
                
                changed = True
        if changed == False:
            break
    return th, thZero

result_th, result_thZero = Perceptron(8, dataset)


# In[ ]:




