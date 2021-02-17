import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression





# This function looks at how many zero elements there are in the data.
# If there are more than $threshold$ zeros. The data is removed.
# At 420, no data is removed, at -1, alla data is removed.

def cleanup_data(x_data, y_data):
    delete_idx = []
    threshold = 420
    for idx, row in enumerate(x_data):
        nbr_of_zeros = np.count_nonzero(row == 0)
        if nbr_of_zeros > 420:
            delete_idx.append(nbr_of_zeros)

    new_x = np.delete(x_data, delete_idx, axis = 0)
    new_y = np.delete(y_data, delete_idx, axis = 0)
    return new_x, new_y

# Some data only contain zeros. Remove them
def remove_zero_data(x_data, y_data):

    delete_idx = []
    for idx, x in enumerate(x_data):
        if np.linalg.norm(x) == 0:
            delete_idx.append(idx)
        
    new_x = np.delete(x_data, delete_idx, axis = 0)
    new_y = np.delete(y_data, delete_idx, axis = 0)
    return new_x, new_y

# Normalize data
def normalize_data(x_data):
    x_mean = np.mean(x_data)
    sigma = np.std(x_data)
    x_min = np.min(x_data) # This is just zero in our case
    # x_data = (x_max-x_data)/x_max
    x_data = (x_data-x_mean)/sigma
    return x_data

# Action 3 was often classed as action 4. So a weight was added to focus on correct predictions on class 3
def train(x_data, y_data):
    weight = {3:4}#, 4:2,9:2} # Adds importance to action 3. Improved performance.
    clf = svm.SVC(class_weight= weight)
    clf2 = LogisticRegression(random_state=0)
    accuracies = []
    accuracies2 = []
    confusion_matrices = []
    confusion_matrices2 = []
    N = 25
    for k in range(N):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, stratify = y_data)

        clf.fit(x_train, y_train)
        clf2.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        y_pred2 = clf2.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy2 = metrics.accuracy_score(y_test, y_pred2)

        accuracies.append(accuracy)
        accuracies2.append(accuracy2)

        c_matrix = confusion_matrix(y_test, y_pred)
        c_matrix2 = confusion_matrix(y_test, y_pred2)
        confusion_matrices.append(c_matrix)
        confusion_matrices2.append(c_matrix2)
    print("\nAverage accuracy after " + str(N) +  " runs with SVM:",np.mean(accuracies))
    print("\nAverage accuracy after " + str(N) +  " runs with LogRegression:",np.mean(accuracies2))
    return accuracies, accuracies2, confusion_matrices, confusion_matrices2


def read_data():
    x_data = pd.read_csv('training_data_x.csv')
    y_data = pd.read_csv('training_data_y.csv')

    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    x_data = transpose_x_data(x_data)
    y_data = np.transpose(y_data)[0]
    
    return x_data, y_data


def transpose_x_data(x_data):
    for i, mat in enumerate(x_data):
        mat = mat.reshape((-1,12))
        mat=np.array([np.array(xi) for xi in mat])
        x_data[i] = mat.flatten('F')
    return x_data

# Action 9 seemed to "take over" a lot of classifications so here it is removed
def remove_action(nbr, x_data, y_data):
    delete_idx = np.where(y_data==nbr)
    new_x = np.delete(x_data, delete_idx, axis = 0)
    new_y = np.delete(y_data, delete_idx, axis = 0)
    return new_x, new_y


def plot_distribution(y_data):
    plt.hist(y_data, bins = len(y_data));
    plt.show()

def visual_data_example(nbr: int, x_data, y_data):
    instances = np.where(y_data==nbr)
    for i in range(3):
        x_plot = x_data[instances[0][i]]
        # plt.hist(x_plot)
        plt.plot(np.linspace(1,len(x_plot), len(x_plot)), x_plot)
        plt.title(nbr)
        plt.show()

def remove_data(nbr, amount, x_data, y_data):
    instances = np.where(y_data==nbr)
    delete_idx = instances[0][0:amount]
    new_x = np.delete(x_data, delete_idx, axis = 0)
    new_y = np.delete(y_data, delete_idx, axis = 0)
    return x_data, y_data

def run(x_data, y_data):
    accuracies, accuracies2, confusion_matrices, confusion_matrices2 = train(x_data, y_data)
    val = np.min(accuracies)
    i = np.argmin(accuracies)
    print("Lowest accuracy: " + str(val))
    print('\n')
    print(confusion_matrices[i])
    return accuracies, accuracies2, confusion_matrices, confusion_matrices2

def main():
    x_data, y_data = read_data()
    x_data, y_data = remove_zero_data(x_data, y_data)
    # x_data, y_data = cleanup_data(x_data, y_data)
    # x_data, y_data = remove_data(9, 60, x_data, y_data)
    # x_data, y_data = remove_data(4, 50, x_data, y_data)
    # plot_distribution(y_data)
    x_data = normalize_data(x_data)
        # visual_data_example(i, x_data, y_data)
    # for i in range(len(y_data)):
    # x_data, y_data = remove_action(9, x_data, y_data)
    # x_data, y_data = remove_action(0, x_data, y_data)
    # x_data, y_data = remove_action(4, x_data, y_data)
    accuracies, accuracies2, confusion_matrices, confusion_matrices2 = run(x_data, y_data)
    

    

main()












