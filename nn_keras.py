import logging
import os
from datetime import datetime
import numpy as np
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from time import time
import utils


class MyGenerator(Sequence):

    def __init__(self, X_filenames, y_filenames, batch_size):
        self.X_filenames = X_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.X_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_batch = self.y_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((len(x_batch), 384, 384, 3))
        y = np.empty((len(y_batch), 256, 256, 3))
        for i in range(len(x_batch)):
            X_tmp = np.load(x_batch[i])['X_train']
            for j in range(3):
                X[i, :, :, j] = np.pad(X_tmp[:, :, j], ((64, 64), (64, 64)), 'wrap')
            y[i] = np.load(y_batch[i])['y_train']
        return X, y


class MyKerasCNN():

    def __init__(self, n_points, activation_function):
        self.n_points = n_points
        self.n_channels = 3
        self.act_fun = activation_function
        self.predictions = []
        self.true = []
        self.mse = []
        self.model = None

    def deconv_cnn_model(self, loss_func, opt):
        # create model
        input_shape = (int(3 / 2 * self.n_points), int(3 / 2 * self.n_points), self.n_channels)
        model = Sequential()  # Initialize model
        # Deconv CNN
        model.add(Conv2D(38, kernel_size=(1, 129), activation=self.act_fun, kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(38, kernel_size=(129, 1), activation=self.act_fun, kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(3, kernel_size=(1, 1), activation=self.act_fun, kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))

        # compile model
        model.compile(loss=loss_func, optimizer=opt)
        logging.info(model.summary())
        self.model = model
        return self.model

    def deconv_cnn_noise_model(self, loss_func, opt):
        # create model
        input_shape = (int(3 / 2 * self.n_points), int(3 / 2 * self.n_points), self.n_channels)
        model = Sequential()  # Initialize model
        # # Deconv CNN
        model.add(Conv2D(38, kernel_size=(1, 129), activation=self.act_fun, kernel_initializer='he_normal',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(38, kernel_size=(129, 1), activation=self.act_fun, kernel_initializer='he_normal',
                         padding='valid'))
        model.add(Conv2D(512, kernel_size=(16, 16), kernel_initializer='he_normal',
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

    # def cfd_cnn_model(self, loss_func, opt):
    #     # create model
    #     input_shape = (int(3 / 2 * self.n_points), int(3 / 2 * self.n_points), self.n_channels)
    #     model = Sequential()  # Initialize model
    #     # Encoder
    #     model.add(Conv2D(38, kernel_size=(1, 129), activation=self.act_fun, kernel_initializer='he_normal',
    #                      input_shape=input_shape, padding='valid'))
    #     model.add(Conv2D(38, kernel_size=(129, 1), activation=self.act_fun, kernel_initializer='he_normal',
    #                      input_shape=input_shape, padding='valid'))
    #     model.add(Conv2D(3, kernel_size=(1, 1), activation=self.act_fun, kernel_initializer='he_normal',
    #                      input_shape=input_shape, padding='valid'))
    #
    #     # compile model
    #     model.compile(loss=loss_func, optimizer=opt)
    #     logging.info(model.summary())
    #     self.model = model
    #     return self.model


    # def evaluate_model(self, X_train, y_train, X_validation, y_validation, plot_folder):
    #     seed = 12
    #     seed = np.random.seed(seed)
    #
    #     # Keras requirese to pass blank function to `build_fn`
    #     if two_layer:
    #         self.estimator = KerasRegressor(build_fn=self.two_layer_model, epochs=self.epochs, batch_size=64, verbose=2)
    #     else:
    #         self.estimator = KerasRegressor(build_fn=self.baseline_model, epochs=self.epochs, batch_size=64, verbose=2)
    #     start_training = time()
    #     self.estimator_trained = self.estimator.fit(X_train, y_train, validation_data = (X_validation, y_validation))
    #     end_training = time()
    #     utils.save_loss_per_epoch(plot_folder, self.estimator_trained.history["loss"], self.estimator_trained.history["val_loss"])
    #     self.training_time = utils.timer(start_training, end_training, 'Training time')
    #     self.plot_loss_per_epoch(plot_folder, two_layer)
    #
    # def plot_loss_per_epoch(self, plot_folder, two_layer):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    #     ax.plot(range(1,self.epochs+1), self.estimator_trained.history['loss'], color="steelblue", marker="o", label="training")
    #     ax.plot(range(1,self.epochs+1), self.estimator_trained.history['val_loss'], color="green", marker="o", label="validation")
    #     ax.grid(alpha=0.25)
    #     ax.legend(loc="best", fontsize=10)
    #     plt.xticks(range(1,self.epochs+2,2))
    #     ax.set_xlabel("epoch", fontsize=16)
    #     ax.set_ylabel("loss", fontsize=16)
    #     if two_layer:
    #         title = "Two Layer $[{}, {}]$ Neurons".format(self.num_neurons, self.num_neurons_L2)
    #         ax.set_title(title)
    #         plot_folder = os.path.join(plot_folder, 'FF_2L_{}_{}_neurons_loss_per_epoch.png'.format(str(self.num_neurons),
    #                                                                                     str(self.num_neurons_L2)))
    #     else:
    #         title = "Single Layer {} Neurons".format(self.num_neurons)
    #         ax.set_title(title)
    #         plot_folder = os.path.join(plot_folder, 'FF_1L_{}_neurons_loss_per_epoch.png'.format(str(self.num_neurons)))
    #     plt.savefig(plot_folder)
    #
    # def evaluate_test_sets(self, X_test_list, y_test_list):
    #     """ Return training examples as observations (rows) x features (columns)
    #     :param X_test_list: list of len=3 with each element as an array with rows == (256*256*3 | 64*64*64*3) and
    #                         columns == (9 | 25 | 27)
    #     :param y: list of len=3 with each element as a single column array with same num rows as X_test_list
    #     :return: none - but append predictions and mse
    #     """
    #
    #     for sigma in range(len(X_test_list)):
    #         logging.info("Evaluating test set: {}".format(sigma))
    #         prediction = self.estimator.predict(X_test_list[sigma])
    #         error = mean_squared_error(y_test_list[sigma], prediction)
    #         self.predictions.append(prediction)
    #         self.true.append(y_test_list[sigma])
    #         self.mse.append(error)
    #     # self.plot_mse()
    #
    # def plot_mse(self):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    #     sigma = [1, 1.1, 0.9]
    #     labels = [r'$\sigma = 1$', r'$\sigma = 1.1$', r'$\sigma = 0.9$']
    #     colors = ['steelblue','green','black']
    #     for mse in range(len(self.mse)):
    #         ax.scatter(sigma[mse], self.mse[mse], color=colors[mse], marker="o", label=labels[mse])
    #     ax.grid(alpha=0.25)
    #     ax.legend(loc="best", fontsize=10)
    #     bottom = min(self.mse) - 0.1*min(self.mse)
    #     top = max(self.mse) + 0.1*min(self.mse)
    #     ax.set_ylim(bottom, top)
    #     # plt.xticks(range(1,self.epochs+2,2))
    #     ax.set_xlabel("sigma", fontsize=16)
    #     ax.set_ylabel("MSE", fontsize=16)
    #     plt.savefig('plots/' + datetime.now().strftime('%Y-%m-%d %H_%M') + '_MSE.png')

# LEFTOVERS
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)    