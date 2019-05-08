import logging
import numpy as np
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Conv2D


class MyGenerator(Sequence):
    """
    Generator class to read examples straight from file.
    """

    def __init__(self, x_filenames, y_filenames, n_channels, n_points, n_padding, batch_size):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.n_points = n_points
        self.n_padding = n_padding
        self.n_channels = n_channels

    def __len__(self):
        return int(np.ceil(len(self.x_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.x_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_batch = self.y_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        x = np.empty((len(x_batch), self.n_points+2*self.n_padding, self.n_points+2*self.n_padding, self.n_channels))
        y = np.empty((len(y_batch), self.n_points, self.n_points, self.n_channels))
        for i, xbatch in enumerate(x_batch):
            x_tmp = np.load(xbatch)['X_train']
            for j in range(self.n_channels):
                x[i, :, :, j] = np.pad(x_tmp[:, :, j],
                                       ((self.n_padding, self.n_padding), (self.n_padding, self.n_padding)), 'wrap')
            y[i] = np.load(y_batch[i])['y_train']
        return x, y


class MyKerasCNN():
    """
    Class of Convolutional Neural Nework (CNN).
    """

    def __init__(self, n_points, activation_function):
        self.n_points = n_points
        self.n_channels = 3
        self.act_fun = activation_function
        self.model = None

    def deconv_cnn_model(self, loss_func, opt, n_kernels, kernel_size):
        """Simple Deconvolutional Convolutional Neural Network.
        :param loss_func: loss function
        :param opt: optimizer
        :param n_kernels: number of kernels
        :param kernel_size: tuple of sizes of kernels
        :return: model
        """
        n_padding = int((kernel_size[0] + kernel_size[1] - 2)/2)
        print(n_padding)
        input_shape = (2*n_padding+self.n_points, 2*n_padding+self.n_points, self.n_channels)
        model = Sequential()  # Initialize model
        # Deconv CNN
        model.add(Conv2D(n_kernels, kernel_size=(1, kernel_size[0]), activation=self.act_fun,
                         kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(n_kernels, kernel_size=(kernel_size[0], 1), activation=self.act_fun,
                         kernel_initializer='he_normal', padding='valid'))
        model.add(Conv2D(3, kernel_size=(kernel_size[1], kernel_size[1]), activation=self.act_fun,
                         kernel_initializer='he_normal', padding='valid'))
        # compile model
        model.compile(loss=loss_func, optimizer=opt)
        logging.info(model.summary())
        self.model = model
        return self.model

    def deconv_cnn_noise_model(self, loss_func, opt, n_kernels, kernel_size):
        """ Deconvolutional Convolutional Neural Network joined with denoising CNN.
        :param loss_func: loss function
        :param opt: optimizer
        :param n_kernels: number of kernels
        :param kernel_size: tuple of sizes of kernels
        :return: model
        """
        # create model
        input_shape = (int(3 / 2 * self.n_points), int(3 / 2 * self.n_points), self.n_channels)
        model = Sequential()  # Initialize model
        # # Deconv CNN
        model.add(Conv2D(n_kernels, kernel_size=(1, kernel_size[0]), activation=self.act_fun, kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(n_kernels, kernel_size=(kernel_size[0], 1), activation=self.act_fun, kernel_initializer='he_normal',
                         padding='valid'))
        model.add(Conv2D(512, kernel_size=(kernel_size[1], kernel_size[1]), kernel_initializer='he_normal',
                         padding='valid'))
        # Denoising CNN
        model.add(Conv2D(512, kernel_size=(1, 1), activation=self.act_fun, kernel_initializer='he_normal',
                         padding='valid'))
        model.add(Conv2D(3, kernel_size=(8, 8), activation=self.act_fun, kernel_initializer='he_normal',
                         padding='valid'))

        # compile model
        model.compile(loss=loss_func, optimizer=opt)
        logging.info(model.summary())
        self.model = model
        return self.model
