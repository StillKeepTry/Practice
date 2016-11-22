import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import numpy as np

import theano.tensor as T
import theano

import IPython

fx = np.loadtxt('ex4Data/ex4x.dat')
fy = np.loadtxt('ex4Data/ex4y.dat')

def Newton(X, Y, epochs=5):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y.shape = (Y.shape[0], 1)

    gradient_descent = []
    
    x = T.dmatrix("x")
    y = T.dmatrix("y")

    w = theano.shared(value = np.matrix(np.random.randn(3)).transpose(), name = "w")
    
    hypothesis = 1. / (1. + T.exp(- T.dot(x, w)))
    cost = (-y * T.log(hypothesis) - (1 - y) * T.log(1 - hypothesis)).mean()
    
    gw = T.grad(cost, w)

    H, updates = theano.scan(
                    fn = lambda i, g, w : T.grad(T.sum(g[i]), w),
                    sequences = T.arange(gw.shape[0]),
                    non_sequences = [gw, w]
                )

    newton = theano.function(
                inputs = [x, y],
                outputs = [gw, cost, H],
                updates = updates)
    
    p1 = T.dmatrix("p1")
    p2 = T.dmatrix("p2")
    
    p3 = T.dot(p1, p2)

    updates_model = [(w, w - p3)]

    train = theano.function(
                inputs = [p1, p2],
                outputs = [p3],
                updates = updates_model
            )
    
    for i in range(epochs):
        result = newton(X, Y)
        hessian_matrix = result[2].reshape(result[2].shape[0], result[2].shape[1])
        train_model = train(np.linalg.inv(hessian_matrix), result[0])
        gradient_descent.append(result[1])

    return w.get_value(), gradient_descent

def quasi_Newton():
    pass

def Logistic_Regression(X, Y, epochs=5000, learning_rate=0.05):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y.shape = (Y.shape[0], 1)

    gradient_descent = []
    
    w = theano.shared(value = np.matrix(np.random.randn(3)).transpose(), name = "w")

    x = T.dmatrix("x")
    y = T.dmatrix("y")

    hypothesis = 1. / (1. + T.exp(- T.dot(x, w)))
    xcent = -y * T.log(hypothesis) - (1 - y) * T.log(1 - hypothesis)
    loss = xcent.mean()
    g_w = T.grad(loss, w)

    updates = [(w, w - learning_rate * g_w)]

    train_model = theano.function(
                    inputs = [x, y],
                    outputs = [loss],
                    updates = updates
                )
    for i in range(epochs):
        cost = train_model(X, Y)
        gradient_descent.append(cost[0])
    return w.get_value(), gradient_descent[::2]

def draw_split_line(theta, line_type):
    x = np.arange(2)
    y = - (theta[0] * x + theta[2]) / theta[1]
    pyplot.plot(x, y, line_type)

def draw_gradient(gd, savefile=None):
    pyplot.plot(gd, 'o')
    if savefile != None:
        pyplot.savefig(savefile)
        pyplot.close()

def main():
    print theano.config.device
    
    fy_idx = np.arange(len(fy))
    fx_min, fx_max = np.array([fx[:, 0].min(), fx[:, 1].min()]), \
                     np.array([fx[:, 0].max(), fx[:, 1].max()])
    fx_normalized = (fx - fx_min) / (fx_max - fx_min)

    fx_normalized_p = fx_normalized[fy_idx[fy == 1]]
    fx_normalized_n = fx_normalized[fy_idx[fy == 0]]

    fy_normalized = fy
    
    pyplot.plot(fx_normalized_p[:, 0], fx_normalized_p[:, 1], '+')
    pyplot.plot(fx_normalized_n[:, 0], fx_normalized_n[:, 1], 'o')
    
    theta, Logistic_regression_gd = Logistic_Regression(fx_normalized, fy_normalized)
    draw_split_line(theta, 'b-')

    theta, newton_gd = Newton(fx_normalized, fy_normalized)
    draw_split_line(theta, 'r-')

    pyplot.savefig('split.jpg')
    pyplot.close()

    draw_gradient(newton_gd, "newton_gd.jpg")
    draw_gradient(Logistic_regression_gd, "logistic_regression.jpg")

if __name__ == '__main__':
    main()
