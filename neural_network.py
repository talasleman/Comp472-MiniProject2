# Import necessary libraries
from sklearn.neural_network import MLPClassifier
import numpy as np

#training data
X_train = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1]
])

#output of training data
y_train = np.array([
    1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

#testing data
X_test = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1]
])

#output of testing data
y_test = np.array([
    1, 1, 0, 0, 1, 1, 1, 1
])

#MLPClassifier model
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=5000, activation='relu', solver='adam')

#Train the MLPClassifier model
clf.fit(X_train, y_train)

#Check accuracy of model, expect a high score out of 100
accuracy = clf.score(X_test, y_test)
print(f'Test set accuracy: {accuracy * 100}%')
