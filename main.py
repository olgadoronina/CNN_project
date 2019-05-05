import data
import logging
import numpy as np
import sys
import os
import plotting
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from keras.models import model_from_json
import nn_keras

plot_folder_base = './plots/'
data_folder = '/home/olga/data/examples/32_bit/'


def main():
    np.random.seed(1234)

    data_output_folder_base = './data_output/'
    plot_folder_base = './plots/'
    data_output_folder = data_output_folder_base
    plot_folder = plot_folder_base

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
    assert filter_type == "gaussian" \
           or filter_type == "median" \
           or filter_type == "noise" \
           or filter_type == "fourier_sharp" \
           or filter_type == "physical_sharp", \
        'Incorrect filter type: {}'.format(filter_type)

    logging.info('Filter type is {}'.format(filter_type))
    plot_folder = os.path.join(plot_folder_base, filter_type)
    if not os.path.isdir(plot_folder): os.makedirs(plot_folder)
    ########################## DEFINE MODEL ##########################
    # model_type = 'CNN'
    valid_frac = 0.1
    batch_size = 20
    epochs = 5
    number_of_examples = 2000
    number_of_valid = int(valid_frac*number_of_examples)
    number_of_training = number_of_examples - number_of_valid
    logging.info("number of training examples is {}".format(number_of_training))
    logging.info("number of validation examples is {}".format(number_of_valid))

    # Define activation function to use for 'FF_1L' and 'FF_2L'
    activation_function = 'sigmoid'
    assert activation_function == 'relu' \
           or activation_function == 'tanh' \
           or activation_function == 'sigmoid', 'Incorrect activation function: {}'.fotmat(activation_function)

    loss_func = 'mean_squared_error'
    optimizer = 'adam'
    # ########################## INIT MODEL ##########################
    logging.info('CNN with {} for {} epochs'.format(number_of_examples, epochs))
    cnn = nn_keras.MyKerasCNN(n_points, activation_function)
    model = cnn.deconv_cnn_model(loss_func, optimizer)
    ########################## RUN MODEL ##########################
    # Data generators
    y_foldername = os.path.join(data_folder, 'y_train')
    x_foldername = os.path.join(data_folder, 'X_train_{}'.format(filter_type))
    y_filenames = [os.path.join(y_foldername, '{}.npz'.format(i)) for i in range(number_of_examples)]
    X_filenames = [os.path.join(x_foldername, '{}.npz'.format(i)) for i in range(number_of_examples)]
    my_training_batch_generator = nn_keras.MyGenerator(X_filenames[:number_of_training],
                                                       y_filenames[:number_of_training], batch_size)
    my_validation_batch_generator = nn_keras.MyGenerator(X_filenames[number_of_training:],
                                                         y_filenames[number_of_training:], batch_size)

    training = model.fit_generator(generator=my_training_batch_generator,
                                   epochs=epochs,
                                   verbose=1,
                                   validation_data=my_validation_batch_generator,
                                   use_multiprocessing=True,
                                   workers=1,
                                   max_queue_size=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    plotting.plot_loss_per_epoch(plot_folder, epochs, training.history)

    # evaluate the model
    # scores = model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # serialize model to JSON


    # # later...
    #
    # # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # logging.info(model.summary())


    # # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)

    #     new_shape = (1,) + X_test[0].shape
    #     # # Evaluate model, validating on same test set key as trained on
    #     test_eval = model.evaluate(X_test[0].reshape(new_shape), y_test[0].reshape(new_shape), verbose=0)
    #     test_eval_smaller = model.evaluate(X_test[1].reshape(new_shape), y_test[1].reshape(new_shape), verbose=0)
    #     test_eval_bigger = model.evaluate(X_test[2].reshape(new_shape), y_test[2].reshape(new_shape), verbose=0)
    #     print('Test loss same filter:', test_eval)
    #     print('Test loss smaller filter:', test_eval_smaller)
    #     print('Test loss bigger filter:', test_eval_bigger)
    # #
    #     y_predict = model.predict(X_test)
    #
    #     plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)

    # # Record training time for each model
    # training_time.append("{}epochs {}neurons: {}".format(epochs, neurons, model.training_time))
    #
    # # Predict on each of the test sets and plot MSE:
    # # MSE plotting currently not working
    # model.evaluate_test_sets(X_test_final, y_test_final)
    #
    # # print(model.predictions.shape, type(model.predictions))
    # untransformed_predictions = []
    # for p in model.predictions:
    #     untransformed_predictions.append(untransform(p, shape))
    # plotting.plot_velocities_and_spectra(X_test, y_test, untransformed_predictions, plot_folder)
    # predictions.append(model.predictions)
    # mse.append(model.mse)
    #
    # save_results(plot_folder, model.predictions, model.true, model.mse)


if __name__ == '__main__':
    main()




    # input_shape = (int(3/2*n_points), int(3/2*n_points), n_channels)
    #
    # model = Sequential()        # Initialize model
    # # Encoder
    # model.add(Conv2D(38, kernel_size=(1, 129), activation=activation_function, kernel_initializer='he_normal',
    #                  input_shape=input_shape, padding='valid'))
    # model.add(Conv2D(38, kernel_size=(129, 1), activation=activation_function, kernel_initializer='he_normal',
    #                  input_shape=input_shape, padding='valid'))
    # model.add(Conv2D(3, kernel_size=(1, 1), activation=activation_function, kernel_initializer='he_normal',
    #                  input_shape=input_shape, padding='valid'))

    # # model.add(MaxPooling2D((2, 2), padding='same'))
    # # model.add(Conv2D(32, (3, 3),
    # #                  activation=activation_function, kernel_initializer='normal', padding='same'))
    # # model.add(MaxPooling2D((2, 2), padding='same'))
    #
    #
    # # Decoder
    # # model.add(Conv2D(32, (3, 3), activation=activation_function, kernel_initializer='normal', padding='same'))
    # # model.add(UpSampling2D((2, 2)))
    # # model.add(Conv2D(32, (3, 3), activation=activation_function, kernel_initializer='normal', padding='same'))
    # # model.add(UpSampling2D((2, 2)))
    # # model.add(Conv2D(16, (3, 3), activation='sigmoid'))
    # # model.add(UpSampling2D((2, 2)))
    # # model.add(Conv2D(3, (3, 3), activation=activation_function, kernel_initializer='normal', padding='same'))