#Import required packages
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn



#Load Iris dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#Using train_test_split to create train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


#Create datafram from data in X_train
#label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#Introducing KNeighbor

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#Calling the fit methoda and making a prediction based on new data

X_new = np.array([[5, 2.9, 1, 0.2]])
knn.fit(X_train, y_train)

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))

#Evaluating the model using the test set

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


#Developed by Eddie Odanga


