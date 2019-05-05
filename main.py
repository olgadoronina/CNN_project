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

plot_folder = './plots/'
data_folder = '/home/olga/data/examples/'

def main():
    np.random.seed(1234)

    # Define base variables
    Npoints_coarse2D = 256
    # Npoints_coarse3D = 64
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

    ########################## DEFINE MODEL ##########################
    # model_type = 'CNN'
    # Define the number of inputs to be used for creating the feature vectors.  See below for requirements:

    # Define activation function to use for 'FF_1L' and 'FF_2L'
    activation_function = 'sigmoid'
    assert activation_function == 'relu' \
           or activation_function == 'tanh' \
           or activation_function == 'sigmoid', 'Incorrect activation function: {}'.fotmat(activation_function)

    ########################## FORMAT TRAINING AND TESTING ##########################
    # Select number of dimensions to use for analysis: 2 or 3
    dimension = 2
    assert dimension == 2 or dimension == 3, 'Incorrect number of dimensions: {}'.format(dimension)
    # Select filter type to use: gaussian, median, or noise
    filter_type = "noise"
    assert filter_type == "gaussian" \
           or filter_type == "median" \
           or filter_type == "noise" \
           or filter_type == "fourier_sharp" \
           or filter_type == "physical_sharp", \
        'Incorrect filter type: {}'.format(filter_type)

    # Define arguments based on required dimensions
    if dimension == 2:
        Npoints_coarse = Npoints_coarse2D
        size = (Npoints_coarse2D, Npoints_coarse2D)

    # Load in data
    # data.load_data(dimension)
    number_of_examples = 20
    N_channels = 3

    # # traiing data
    # data_all = np.empty((number_of_examples,) + size + (N_channels,))
    # data_all[:4040] = np.load('/home/olga/data/examples/data_CNNx_y_slice.npz')['data_train']
    # print('x_y')
    # data_all[4040:8080] = np.load('/home/olga/data/examples/data_CNNx_z_slice.npz')['data_train']
    # print('x_z')
    # data_all[8080:] = np.load('/home/olga/data/examples/data_CNNy_z_slice.npz')['data_train']
    # print('y_z')
    # print(data_all.shape)
    # # # shuffle indeces
    # ind = np.arange(number_of_examples)
    # np.random.shuffle(ind)
    # data.save_shuffled_truth_data(data_all, ind)
    # data.filter_and_save_shuffled_data(data_all, ind, filter_type)
    # del data_all



    # #    # test data
    # data_test = np.empty((3333,) + size + (N_channels,))
    # data_test[:1111] = np.load('/home/olga/data/examples/data_CNNx_y_slice.npz')['data_test']
    # print('x_y')
    # data_test[1111:2222] = np.load('/home/olga/data/examples/data_CNNx_z_slice.npz')['data_test']
    # print('x_z')
    # data_test[2222:] = np.load('/home/olga/data/examples/data_CNNy_z_slice.npz')['data_test']
    # print('y_z')
    # print(data_test.shape)
    #
    # x_test = data.filter_and_save_test_data(data_test, filter_type)
    # print(x_test)
    # exit()

    # for i in range(number):
    #     print(i)
    #     data_new = data[ind[404*i:404*(i+1)]]
    #     np.savez('/home/olga/data/examples/data'+str(i)+'.npz', data=data_new)
    #     del data_new
    # for i in range(30):
    #     print(i)
    #     data_tmp = np.load('/home/olga/data/examples/data'+str(i)+'.npz')['data']
    #     for j, example in enumerate(data_tmp):  # Below just applies filter, keeping it in shapes of [256, 256]
    #         filtered = data.filtering(filter_type, example, dimension)
    #         np.savez('/home/olga/data/examples/y_train/{}.npz'.format(i * 404 + j), y_train=data_tmp[j])
    #         np.savez('/home/olga/data/examples/X_train_noise/{}.npz'.format(i * 404 + j), X_train=filtered)
    #     del data_tmp


    # ########################## RUN MODEL ##########################

    epochs = 10
    N_channels = 3    # for 'u', 'v', 'w' velocity
    logging.info('Run CNN for {} epochs'.format(epochs))
    #
    input_shape = (int(3/2*Npoints_coarse), int(3/2*Npoints_coarse), N_channels)
    # Initialize model
    model = Sequential()
    # Encoder
    model.add(Conv2D(38, kernel_size=(1, 129),
                     activation=activation_function, kernel_initializer='normal',
                     input_shape=input_shape, padding='valid'))
    model.add(Conv2D(38, kernel_size=(129, 1),
                     activation=activation_function, kernel_initializer='he_normal',
                     input_shape=input_shape, padding='valid'))
    model.add(Conv2D(3, kernel_size=(1, 1),
                     activation=activation_function, kernel_initializer='he_normal',
                     input_shape=input_shape, padding='valid'))

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

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    logging.info(model.summary())
#
    logging.info('{}'.format(filter_type))
    plot_folder = os.path.join(plot_folder_base, filter_type)
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    y_filenames = ['{}y_train/{}.npz'.format(data_folder, i) for i in range(number_of_examples)]
    X_filenames = ['{}X_train_{}/{}.npz'.format(data_folder, filter_type, i) for i in range(number_of_examples)]
    valid_frac = 0.1
    number_of_valid = int(valid_frac*number_of_examples)
    number_of_training = number_of_examples - number_of_valid
    logging.info("number of training examples is {}".format(number_of_training))
    logging.info("number of validation examples is {}".format(number_of_valid))
    batch_size = 1
    my_training_batch_generator = nn_keras.MyGenerator(X_filenames[:number_of_training],
                                                       y_filenames[:number_of_training], batch_size)
    my_validation_batch_generator = nn_keras.MyGenerator(X_filenames[number_of_training:],
                                                         y_filenames[number_of_training:], batch_size)

    training = model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(number_of_training // batch_size),
                        epochs=epochs,
                        verbose=1,
                        validation_data=my_validation_batch_generator,
                        validation_steps=(number_of_valid // batch_size),
                        use_multiprocessing=True,
                        workers=1,
                        max_queue_size=1)


    plotting.plot_loss_per_epoch(plot_folder, epochs, training.history)

    # evaluate the model
    # scores = model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

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