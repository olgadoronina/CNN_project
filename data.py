import numpy as np
import logging
import scipy.ndimage as ndimage
import utils
import filters
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
data_folder = '/home/olga/data/JohnHopkins/2d_256_tstep2/'
# data_folder = '/Users/olgadorr/data/'
read_from_cvs = 1

#
# def vti_to_numpy(filename):
#     print(filename)
#     reader = vtk.vtkXMLImageDataReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     im = reader.GetOutput()
#     print(im)
#     x, y, z = im.GetDimensions()
#     print(x, y, z)
#     sc = im.GetFieldData()
#     print(sc)
#     a = vtk_to_numpy(sc)
#     print(a.shape)
#     # a = a.reshape(rows, cols, -1)
#     print(a.shape)
#     assert a.shape == im.GetDimensions()
#     return x


def load_data(dimension=2):
    """
    Load data from .csv
    :param dimension: 2 or 3
    :return: list of dictionaries of 'u', 'v', 'w' data array (256^2)
    """
    logging.info('Load csv data')

    number_of_examples = 0
    number_of_test = 0
    filenames = np.arange(0, 101, 2)

    for slices in ['x_y_slice/', 'y_z_slice/', 'x_z_slice/']:
        data_train = []
        data_test = []
        for i in filenames:
            print(i)
            filename = data_folder + slices + 'cutout.' + str(i)
            velocity = np.empty((256, 256, 3))
            data = np.genfromtxt(filename + '.csv', delimiter=',')[1:]
            # Normalize
            data /= np.max(np.abs(data))
            # split velocities
            for j in range(3):
                velocity[:, :, j] = data[:, j].reshape((256, 256))
            # reflect
            if i % 10 == 1:
                for j in range(3):
                    velocity[:, :, j] = np.fliplr(velocity[:, :, j])
            if i % 10 == 2:
                for j in range(3):
                    velocity[:, :, j] = np.flipud(velocity[:, :, j])
            # rotate
            if i % 10 == 3:
                for j in range(3):
                    velocity[:, :, j] = np.rot90(velocity[:, :, j], k=1)
            if i % 10 == 6:
                for j in range(3):
                    velocity[:, :, j] = np.rot90(velocity[:, :, j], k=2)
            if i % 10 == 9:
                for j in range(3):
                    velocity[:, :, j] = np.rot90(velocity[:, :, j], k=3)
            # rotate and reflect
            if i % 10 == 4:
                for j in range(3):
                    velocity[:, :, j] = np.fliplr(np.rot90(velocity[:, :, j], k=1))
            if i % 10 == 5:
                for j in range(3):
                    velocity[:, :, j] = np.flipud(np.rot90(velocity[:, :, j], k=1))
            if i % 10 == 7:
                for j in range(3):
                    velocity[:, :, j] = np.fliplr(np.rot90(velocity[:, :, j], k=2))
            if i % 10 == 8:
                for j in range(3):
                    velocity[:, :, j] = np.flipud(np.rot90(velocity[:, :, j], k=2))

            if i < 80:
                data_train.append(velocity)
                for sh in np.arange(10, 256, 10):
                    v_shifted = np.roll(velocity, shift=sh, axis=1)     # shifting horizontally
                    data_train.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=sh, axis=0)     # shifting vertically
                    data_train.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(sh, sh), axis=(1, 0))  # shifting diagonally
                    data_train.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(-sh, sh), axis=(1, 0))  # shifting diagonally
                    data_train.append(v_shifted)
            else:
                data_test.append(velocity)
                for sh in np.arange(10, 256, 10):
                    v_shifted = np.roll(velocity, shift=sh, axis=1)     # shifting horizontally
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=sh, axis=0)     # shifting vertically
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(sh, sh), axis=(1, 0))  # shifting diagonally
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(-sh, sh), axis=(1, 0))  # shifting diagonally
                    data_test.append(v_shifted)
        number_of_examples += len(data_train)
        number_of_test += len(data_test)

        np.savez('/home/olga/data/examples/data_CNN' + slices[:-1] + '.npz', data_train=data_train, data_test=data_test)
        logging.info('Data of {} saved'.format(slices[:-1]))

    logging.info('Number of training and test examples are {} and {}'.format(number_of_examples, number_of_test))

    return number_of_examples, number_of_test


def save_shuffled_truth_data(data_all, ind):
    number_of_examples = len(ind)
    for i in range(number_of_examples):
        print(i)
        np.savez('/home/olga/data/examples/y_train/{}.npz'.format(i), y_train=data_all[ind[i]])
    return


def filter_and_save_shuffled_data(data_all, ind, filter_type):
    number_of_examples = len(ind)
    for i in range(number_of_examples):
        print(i)
        filtered = filtering(filter_type, data_all[ind[i]])
        np.savez('/home/olga/data/examples/X_train_noise/{}.npz'.format(i), X_train=filtered)
    return


def filter_and_save_test_data(data_test, filter_type):

    number_of_examples = len(data_test)
    filtered = np.empty((number_of_examples, 3, 256, 256, 3))
    for i in range(number_of_examples):
        filtered[i] = filtering_test(filter_type, data_test[i])
    np.savez('/home/olga/data/examples/X_test_noise.npz', X_test=filtered)
    return filtered


def filtering(filter_type, data_train, dimension=2):
    """
    Filter example of train set with given filter type.
    :param filter_type: type of filter
    :param data_train: np.array of one train example
    :param dimension: dimension of velocity field (2 or 3)
    :return: np.array of filtered fields
    """

    filtered_train = np.empty_like(data_train)

    if filter_type == 'gaussian':
        sigma = [1, 1.1, 0.9]
        for key in range(3):
            filtered_train[:, :, key] = ndimage.gaussian_filter(data_train[:, :, key], sigma=sigma[0],  mode='wrap',
                                                                truncate=500)

    elif filter_type == 'median':
        s = [4, 5, 3]
        for key in range(3):
            filtered_train[:, :, key] = ndimage.median_filter(data_train[:, :, key], size=s[0],  mode='wrap')

    elif filter_type == 'noise':
        mu = [0.2, 0.22, 0.18]
        for key in range(3):
            kappa = np.random.normal(0, 1, size=data_train[:, :, key].shape)
            filtered_train[:, :, key] = data_train[:, :, key] + mu[0]*kappa

    elif filter_type == 'fourier_sharp' or filter_type == 'physical_sharp':
        if dimension == 3:
            k = [4, 5, 3]
        else:
            k = [15, 16, 14]
        for key in range(3):
            filtered_train[:, :, key] = filters.filter_sharp_array(data_train[:, :, key], filter_type=filter_type, scale_k=k[0])

    else:
        logging.error('Filter type is not defined.')

    return filtered_train


def filtering_test(filter_type, data_test, dimension=2):
    """
    Filter 3 examples of test set with different filter scales.
    :param filter_type: type of filter
    :param data_test: np.array of 3 test examples
    :param dimension: dimension of velocity field (2 or 3)
    :return: np.array of filtered fields
    """
    filtered_test = np.empty((3, ) + data_test.shape)

    if filter_type == 'gaussian':
        sigma = [1, 1.1, 0.9]
        for i in range(3):
            for key in range(3):
                filtered_test[i, :, :, key] = ndimage.gaussian_filter(data_test[:, :, key], sigma=sigma[i],
                                                                      mode='wrap', truncate=500)
    elif filter_type == 'median':
        s = [4, 5, 3]
        for i in range(3):
            for key, value in data_test[i].items():
                filtered_test[i, :, :, key] = ndimage.median_filter(data_test[:, :, key], size=s[i], mode='wrap')

    elif filter_type == 'noise':
        mu = [0.2, 0.22, 0.18]
        for i in range(3):
            for key in range(3):
                kappa = np.random.normal(0, 1, size=data_test[:, :, key].shape)
                filtered_test[i, :, :, key] = data_test[:, :, key] + mu[i]*kappa

    elif filter_type == 'fourier_sharp' or filter_type == 'physical_sharp':
        if dimension == 3:
            k = [4, 5, 3]
        else:
            k = [15, 16, 14]
        for i in range(3):
            for key in range(3):
                filtered_test[i, :, :, key] = filters.filter_sharp_array(data_test[:, :, key],
                                                                         filter_type=filter_type, scale_k=k[i])
    else:
        logging.error('Filter type is not defined.')

    return filtered_test
