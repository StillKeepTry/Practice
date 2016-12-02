import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

import cPickle, gzip, IPython
import numpy as np
import h5py

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

nb_classes = 10
batch_size = 50
nb_epochs = 20

def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    return train_set, valid_set, test_set

def save_image(image_mat, image_shape, im_file='test.png'):
    im = np.array(image_mat)
    im *= 256
    im.shape = image_shape
    pyplot.imsave(im_file, im, cmap='gray')
    pyplot.close()

def make_model():
    model = Sequential()
    
    model.add(Reshape((1, 28, 28), input_shape=(784, )))
    model.add(Convolution2D(4, 5, 5, border_mode='valid'))
    model.add(Activation('tanh'))
    
    model.add(Convolution2D(8, 3, 3, border_mode = 'valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Convolution2D(16, 3, 3, border_mode = 'valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('tanh'))
    model.add(Dense(10, init='normal'))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9)
    model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
    return model

def main():
    train_set, valid_set, test_set = load_data()
    train_image, train_label = train_set
#    train_label = np_utils.to_categorical(train_label, nb_classes)
    

#    model = make_model()
#    model.fit(train_image, train_label, 
#              batch_size = batch_size,
#              nb_epoch = nb_epochs, 
#              validation_split = 0.2,
#              shuffle=True)
#    model.save('model.h5')
    model = load_model('model.h5')
    result = model.predict(train_image)
    IPython.embed()

if __name__ == '__main__':
    main()
