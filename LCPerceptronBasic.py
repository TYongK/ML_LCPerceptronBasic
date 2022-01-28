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

def SaveGraphAnimationImages (Result_th, Result_thZero, Dataset, iteration):
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    positive_features = dataset_features[np.where(dataset_labels[:,0] == 1)]
    negative_features = dataset_features[np.where(dataset_labels[:,0] == -1)]
    
    xOne = np.linspace(-1000,3500)
    xTwo = -1*(xOne*Result_th[0,0] + Result_thZero[0])/Result_th[1,0]
    
    plt.clf()
    
    plt.plot(positive_features[:,0], positive_features[:,1], 'ro')
    plt.plot(negative_features[:,0], negative_features[:,1], 'bo')
    plt.plot(xOne, xTwo)
    plt.axis([-1000, 3500, -500, 1500])
    plt.savefig("E:\_Github\ML_linear classifiers\images\lcResult"+ str(iteration) +'.png')
    
def ShowGraph (Result_th, Result_thZero, Dataset):
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    positive_features = dataset_features[np.where(dataset_labels[:,0] == 1)]
    negative_features = dataset_features[np.where(dataset_labels[:,0] == -1)]
    
    xOne = np.linspace(-1000,3500)
    xTwo = -1*(xOne*Result_th[0,0] + Result_thZero[0])/Result_th[1,0]
    
    plt.clf()
    plt.plot(positive_features[:,0], positive_features[:,1], 'ro')
    plt.plot(negative_features[:,0], negative_features[:,1], 'bo')
    plt.plot(xOne, xTwo)
    plt.axis([-1000, 3500, -500, 1500])
    

def Perceptron (iteration, Dataset, SaveImages):   
    dataset_features, dataset_labels = DataOrganize(Dataset)
    
    th = np.zeros(np.shape(dataset_features)[1])[np.newaxis].T
    thZero = 0
    
    for t in range(iteration):
        changed = False
        for i in range(dataset_labels.shape[0]):
            if dataset_labels[i:i+1,:]*(np.dot(th.T, dataset_features[i:i+1,:].T) + thZero) <= 0:
                th = th + (dataset_features[i:i+1,:]*dataset_labels[i:i+1,:]).T
                thZero = thZero + dataset_labels[i:i+1,:]
                if SaveImages == True:
                    SaveGraphAnimationImages(th, thZero, Dataset, (t+1)*(i+1))
                changed = True
                
        if changed == False:
            print("Found Seperation")
            break
    return th, thZero

result_th, result_thZero = Perceptron(3, dataset, True)

ShowGraph(result_th, result_thZero, dataset)

# In[ ]:




