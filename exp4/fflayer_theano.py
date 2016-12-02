import matplotlib
matplotlib.use("Agg")

from sklearn import datasets
import numpy as np
from matplotlib import pyplot

import theano.tensor as T
from theano import shared, function
import time

class FeedForward:
    
    def __init__(self, nn_hdim, pic_name="test.jpg"):
        np.random.seed(0)
        self.X, self.Y = datasets.make_moons(200, noise=0.20)
        # self.X, self.Y = datasets.make_gaussian_quantiles(n_samples=300, n_classes=3)
        self.n = len(self.X)
        self.nn_hdim = nn_hdim
        self.savefile = pic_name

        self.nn_input_dim = 2
        self.nn_output_dim = 2

        self.epsilon = 0.01
        self.regularization = 0.01

        self.x = T.dmatrix('x')
        self.y = T.lvector('y')

        self.w1 = shared(np.random.randn(self.nn_input_dim, nn_hdim), name='w1')
        self.b1 = shared(np.zeros((nn_hdim)), name='b1')
        self.w2 = shared(np.random.randn(nn_hdim, self.nn_output_dim), name='w2')
        self.b2 = shared(np.zeros((self.nn_output_dim)), name='b2')

        self.z1 = self.x.dot(self.w1) + self.b1
        self.a1 = T.tanh(self.z1)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.probs = T.nnet.softmax(self.z2)

        self.loss = T.nnet.categorical_crossentropy(self.probs, self.y).mean()
        self.prediction = T.argmax(self.probs, axis=1)
        self.dw2 = T.grad(self.loss, self.w2)
        self.dw1 = T.grad(self.loss, self.w1)
        self.db2 = T.grad(self.loss, self.b2)
        self.db1 = T.grad(self.loss, self.b1)

        self.gradient_step = function(
            inputs = [self.x, self.y],
            outputs = [self.loss, self.w2, self.dw2],
            updates = ((self.w2, self.w2 - self.epsilon * self.dw2),
                       (self.w1, self.w1 - self.epsilon * self.dw1),
                       (self.b2, self.b2 - self.epsilon * self.db2),
                       (self.b1, self.b1 - self.epsilon * self.db1))
        )

        self.loss_func = function(
            inputs = [self.x, self.y], 
            outputs = [self.loss]
        )
        
        self.predict = function(
            inputs = [self.x],
            outputs = self.prediction
        )

    def plot_decision_boundary(self):
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z.shape = xx.shape

        pyplot.xlim(x_min, x_max)
        pyplot.ylim(y_min, y_max)
        pyplot.contourf(xx, yy, z, cmap=pyplot.cm.Spectral)
        pyplot.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=pyplot.cm.Spectral)
        pyplot.savefig(self.savefile)

    def train(self, epoch = 5000):
        for i in range(epoch):
            result = self.gradient_step(self.X, self.Y)
            if i % 50 == 0:
                print result[0]

def main():
    start_t = time.clock()
    ff_model = FeedForward(50, pic_name="fflayer_theano.jpg")
    ff_model.train()
    ff_model.plot_decision_boundary()
    end_t = time.clock()

    print ("the program cost %.4f second") % (end_t - start_t)

if __name__ == '__main__':
    main()
