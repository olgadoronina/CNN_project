import logging
import sys
import os
import numpy as np
from keras.models import model_from_json
import nn_keras
import plotting
import utils


PLOT_FOLDER_BASE = './plots/'
DATA_FOLDER = '/Users/olgadorr/Classes/2019_spring/Nueral Nets'


def main():
    np.random.seed(1234)

    plot_folder = PLOT_FOLDER_BASE

    # Set logging configuration
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))


    ########################## DEFINE DATA ##########################
    # Define base variables
    n_points = 256
    n_channels = 3    # for 'u', 'v', 'w' velocity

    # Select filter type to use: gaussian, median, or noise
    filter_type = "noise"
    assert filter_type in ("gaussian", "median", "noise", "fourier_sharp", "physical_sharp"), \
        'Incorrect filter type: {}'.format(filter_type)

    logging.info('Filter type is {}'.format(filter_type))
    plot_folder = os.path.join(plot_folder, filter_type)
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    ########################## DEFINE MODEL ##########################
    # model_type = 'CNN'
    valid_frac = 0.1
    batch_size = 10
    epochs = 1
    number_of_examples = 100
    number_of_valid = int(valid_frac*number_of_examples)
    number_of_training = number_of_examples - number_of_valid
    logging.info("number of training examples is {}".format(number_of_training))
    logging.info("number of validation examples is {}".format(number_of_valid))

    # Define activation function
    activation_function = 'tanh'
    assert activation_function in ('relu', 'tanh', 'sigmoid'), \
        'Incorrect activation function: {}'.format(activation_function)

    loss_func = 'mean_squared_error'
    optimizer = 'adam'
    n_kernels = 3
    kernel_size = (29, 5)
    ########################## INIT MODEL ##########################
    logging.info('CNN with {} for {} epochs'.format(number_of_examples, epochs))
    cnn = nn_keras.MyKerasCNN(n_points, activation_function)
    model = cnn.deconv_cnn_model(loss_func, optimizer, n_kernels, kernel_size)
    n_padding = int((kernel_size[0] + kernel_size[1] - 2)/2)
    ########################## RUN MODEL ##########################
    # Data generators
    y_foldername = os.path.join(DATA_FOLDER, 'y_train')
    x_foldername = os.path.join(DATA_FOLDER, 'X_train_{}'.format(filter_type))
    y_filenames = [os.path.join(y_foldername, '{}.npz'.format(i)) for i in range(number_of_examples)]
    x_filenames = [os.path.join(x_foldername, '{}.npz'.format(i)) for i in range(number_of_examples)]
    training_batch_generator = nn_keras.MyGenerator(x_filenames[:number_of_training], y_filenames[:number_of_training],
                                                    n_channels, n_points, n_padding, batch_size)
    validation_batch_generator = nn_keras.MyGenerator(x_filenames[number_of_training:], y_filenames[number_of_training:],
                                                      n_channels, n_points, n_padding, batch_size)

    training = model.fit_generator(generator=training_batch_generator,
                                   epochs=epochs,
                                   verbose=1,
                                   validation_data=validation_batch_generator,
                                   use_multiprocessing=True,
                                   workers=1,
                                   max_queue_size=1)
    # Save model into json file
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model.h5")      # serialize weights to HDF5
    # logging.info("Saved model to disk")

    print(training.history.keys())
    utils.save_loss_per_epoch(plot_folder, training.history['loss'], training.history['val_loss'])
    plotting.plot_loss_per_epoch(plot_folder, epochs, training.history)

    # later...

    # # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights("model.h5")              # load weights into new model
    # logging.info("Loaded model from disk")
    # model.compile(loss=loss_func, optimizer=optimizer)
    # logging.info(model.summary())

    y_test = np.load(os.path.join(DATA_FOLDER, 'y_test.npz'))['y_test']
    x_tmp = np.load(os.path.join(DATA_FOLDER, 'X_test_{}.npz'.format(filter_type)))['X_test']
    x_test = np.empty((x_tmp.shape[0], n_points+2*n_padding, n_points+2*n_padding, n_channels))
    for i in range(x_tmp.shape[0]):
        for j in range(3):
            x_test[i, :, :, j] = np.pad(x_tmp[i, :, :, j], ((n_padding, n_padding), (n_padding, n_padding)), 'wrap')
    new_shape_x = (1,) + x_test[0].shape
    new_shape_y = (1,) + y_test.shape

    # # Evaluate model, validating on same test set key as trained on
    test_eval = model.evaluate(x_test[0].reshape(new_shape_x), y_test.reshape(new_shape_y))
    test_eval_smaller = model.evaluate(x_test[1].reshape(new_shape_x), y_test.reshape(new_shape_y))
    test_eval_bigger = model.evaluate(x_test[2].reshape(new_shape_x), y_test.reshape(new_shape_y))
    logging.info('Test loss same filter: {}'.format(test_eval))
    logging.info('Test loss smaller filter: {}'.format(test_eval_smaller))
    logging.info('Test loss bigger filter: {}'.format(test_eval_bigger))
#
    y_predict = model.predict(x_test)
    np.savez(os.path.join(plot_folder, 'y_predict'),
             y_predict0=y_predict[0], y_predict1=y_predict[1], y_predict2=y_predict[2])
    print(y_predict)
    plotting.plot_velocities_and_spectra(x_test[:, n_padding:-n_padding, n_padding:-n_padding, :],
                                         y_test, y_predict, plot_folder)


if __name__ == '__main__':
    main()
