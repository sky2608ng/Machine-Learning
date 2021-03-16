from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target

# print(np.unique(y).shape) 3 types of Iris
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train) # Calculate standard deviation and sample mean
x_train_std = sc.transform(x_train) # Standardize the data
x_test_std = sc.transform(x_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter = 40, eta0 = 0.01, random_state = 0)
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
print("Misclassified samples: %d" %(y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(x, y, classifier, test_idx = None, resolution = 0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    x_test, y_test = x[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0], y = x[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

    # highlight test samples
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1], c = '', alpha = 1.0, linewidth = 1, marker = 'o', s = 55, label = 'test set')

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(x = x_combined_std, y = y_combined, classifier = ppn, test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('sepal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, 1.0, facecolor = '1.0', alpha = 1.0, ls = 'dotted')
plt.axhspan(y = 0.5, ls = 'dotted', color = 'k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (x) $')
plt.show()