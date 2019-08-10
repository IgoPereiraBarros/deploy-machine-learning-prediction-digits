# https://cloudxlab.com/blog/deploying-machine-learning-model-in-production/

import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


np.random.seed(42)
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Train SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

# Print the accuracy of SGDClassifier
y_pred = sgd_clf.predict(X_test)
sgd_accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(sgd_accuracy))

# Dump the model to the file
pickle.dump(sgd_clf, open('trained_models/mnist_model.pkl', 'wb'))