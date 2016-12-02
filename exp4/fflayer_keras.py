import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

from sklearn import datasets
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np

nb_classes = 5

def generate_data():
    np.random.seed(0)
    x, y = datasets.make_gaussian_quantiles(n_samples = nb_classes * 100, n_classes=nb_classes, shuffle=True)
    return x, y

def plot_decision_boundary(model, x, y):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z.shape = xx.shape

    pyplot.xlim(x_min, x_max)
    pyplot.ylim(y_min, y_max)
    pyplot.contourf(xx, yy, Z, cmap = pyplot.cm.Spectral)
    pyplot.scatter(x[:, 0], x[:, 1], c = y, cmap = pyplot.cm.Spectral)
    pyplot.savefig("fflayer_keras.jpg")
    pyplot.close()

def make_model():
    model = Sequential()
    model.add(Dense(50, input_shape=(2,)))
    model.add(Activation('tanh'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.01, clipvalue = 0.5)

    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def main():
    x, y = generate_data()
    Y = np_utils.to_categorical(y, nb_classes=nb_classes)
    model = make_model()
    model.fit(x, Y, batch_size=20, nb_epoch=1000)
    plot_decision_boundary(model, x, y)

if __name__ == '__main__':
    main()
