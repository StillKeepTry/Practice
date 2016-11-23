from sklearn import datasets, linear_model

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

import numpy as np

def plot_decision_boundary(pred_func, x, y):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z.shape = xx.shape

    pyplot.xlim([x_min, x_max])
    pyplot.ylim([y_min, y_max])
    pyplot.contourf(xx, yy, z, cmap = pyplot.cm.Spectral)
    pyplot.scatter(x[:, 0], x[:, 1], s = 20, c = y, cmap = pyplot.cm.Spectral)
    pyplot.savefig("Logistic_Regression.jpg")
    pyplot.close()

def generate_data():
    np.random.seed(0)
    x, y = datasets.make_moons(200, noise = 0.20)
    return x, y

def main():
    clf = linear_model.LogisticRegressionCV()
    x, y = generate_data()
    clf.fit(x, y)
    plot_decision_boundary(lambda x : clf.predict(x), x, y)


if __name__ == "__main__":
    main()
