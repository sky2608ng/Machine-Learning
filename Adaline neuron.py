import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    """ Adaptive Linear Neuron Classifier.

    Parameters
    ----------
    eta: float
        Learning Rate between 0.0 and 1.0
    n_iter: int
        Passes over the training dataset

    Attributes
    ----------
    w_: 1-d array
        Weights after fitting
    errors_: list
        Number of misclassification in every epoch

    """
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        """Fit Training Data

        Parameters
        ----------
        x: {array-like}, shape = [n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples and n_features is the number of features
        y: array-like, shape = [n_samples]
            Target values

        Returns
        -------
        self: object

        """
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.errors_.append(cost)
        return self

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        """Compute linear activation"""
        return self.net_input(x)

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.activation(x) >= 0.0, 1, -1)

def plot_decision_regions(x, y, classifier, resolution = 0.02):
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

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0,2]].values

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(x, y)
ax[0].plot(range(1, len(ada1.errors_) + 1), ada1.errors_, marker = 'o') 
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(x, y)
ax[1].plot(range(1, len(ada2.errors_) + 1), ada2.errors_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

x_std = np.copy(x)
x_std[:,0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std() 
x_std[:,1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std() 

ada=AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(x_std, y)
plot_decision_regions(x_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.xlabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1, len(ada.errors_) + 1), ada.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

from numpy.random import seed

class AdalineSGD(object):
    """Adaptive linear neuron classifier

    Parameter
    ----------
    eta: floating
        Learning rate between 0.0 and 1.0
    n_iter: int
        Passes over the training dataset

    Attributes
    ----------
    w_: 1-d array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch
    shuffle: bool (default: True)
        Shuffles training data every epoch
        if True to prevent cycles
    random_state: int (default: None)
        Set random satet for shuffling and initializing the weights
    """
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, x, y):
        """Fit training dat

        Parameters
        ----------
        x: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features
        y: array-like, shape = [n_samples]
            Target values

        Returns
        -------
        self: object
        """

        self._initialize_weights(x.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            error = []
            for xi, target in zip(x,y):
                error.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.errors_.append(avg_cost)
        return self
    
    def partial_self(self, x, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    def _shuffle(self, x, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zero"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    
    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        """Compute linear activation"""
        return self.net_input(x)

    def predict(self, x):
        """return class label after unit step"""
        return np.where(self.activation(x) >= 0.0, 1, -1)

ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
ada.fit(x_std, y)
plot_decision_regions(x_std, y, classifier = ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel('Sepal length [standardized')
plt.ylabel('Petal length [standardized')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1, len(ada.errors_) + 1), ada.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()