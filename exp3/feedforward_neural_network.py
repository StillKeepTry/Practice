import matplotlib
matplotlib.use("Agg")

from sklearn import datasets
import numpy as np
from matplotlib import pyplot

import IPython

class FFconfig:
    nn_input_dim = 2
    nn_output_dim = 3

    epsilon = 0.01
    regularization = 0.01

def generate_data():
    np.random.seed(0)
#    x, y = datasets.make_moons(200, noise = 0.20)
    x, y = datasets.make_gaussian_quantiles(n_samples=300, n_classes=3)
    return x, y

def plot_decision_boundary(pred_func, x, y):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    h = 0.01
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z.shape = xx.shape

    pyplot.xlim(x_min, x_max)
    pyplot.ylim(y_min, y_max)
    pyplot.contourf(xx, yy, Z, cmap = pyplot.cm.Spectral)
    pyplot.scatter(x[:, 0], x[:, 1], c = y, cmap = pyplot.cm.Spectral)
    pyplot.savefig("ffnn.jpg")
    pyplot.close()

def calculate_loss(model, x, y):
    sample_len = len(x)

    w1, b1, w2, b2 = model['W1'], model['B1'], model['W2'], model['B2']
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    cross_lobprobs = - np.log(probs[range(sample_len), y])
    data_loss = np.mean(cross_lobprobs)
    return data_loss

def predict(model, x):
    w1, b1, w2, b2 = model['W1'], model['B1'], model['W2'], model['B2']
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
    

def build_model(x, y, nn_hdim, epoch=1000, print_loss=None):
    sample_len = len(x)

    np.random.seed(0)
    w1 = np.random.randn(FFconfig.nn_input_dim, nn_hdim) / np.sqrt(FFconfig.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, FFconfig.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, FFconfig.nn_output_dim))

    model = {}
    
    loss = []

    for i in range(epoch):

        # forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        #IPython.embed()
        delta3[range(sample_len), y] -= 1
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = (1 - a1 * a1) * (delta3.dot(w2.T))
        dw1 = (x.T).dot(delta2)
        db1 = np.sum(delta2, axis=0)

        # add regularization
        dw2 += FFconfig.regularization * w2
        dw1 += FFconfig.regularization * w1

        w1 = w1 - FFconfig.epsilon * dw1
        b1 = b1 - FFconfig.epsilon * db1
        w2 = w2 - FFconfig.epsilon * dw2
        b2 = b2 - FFconfig.epsilon * db2

        model = {'W1' : w1, 'B1' : b1,
                 'W2' : w2, 'B2' : b2}
 
        if print_loss is not None and (i % 50) == 0:
            loss.append(print_loss(model, x, y))
   
    pyplot.plot(loss, 'o')
    pyplot.savefig("ffnn_loss_common.jpg")
    pyplot.close()
    return model

def main():
    x, y = generate_data()
    model = build_model(x, y, 10, epoch=1000, print_loss=calculate_loss)
    plot_decision_boundary(lambda x : predict(model, x), x, y)

if __name__ == "__main__":
    main()
