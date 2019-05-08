import os
import logging
import numpy as np
import scipy.ndimage as ndimage
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy

# import utils
import filters

DATA_FOLDER_SLICES = '/home/olga/data/JohnHopkins/2d_256_tstep2/'
DATA_FOLDER_BASE = '/Users/olgadorr/Classes/2019_Spring/Nueral Nets/'

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


def load_data(folder, n_bit):
    """
    Load data from .csv and populate with rotations, reflections an d shifting
    :param folder: path to save populated slices in .npz format
    :param n_bit: type of data (32 or 64)
    :return: (number of train examples, number of test examples)
    """
    logging.info('Load csv data')

    number_of_examples = 0
    number_of_test = 0
    filenames = np.arange(0, 101, 2)
    if n_bit == 32:
        dtype = np.float32
    elif n_bit == 64:
        dtype = np.float64
    for slices in ['x_y_slice/', 'y_z_slice/', 'x_z_slice/']:
        data_train = []
        data_test = []
        for i in filenames:
            print(i)
            filename = DATA_FOLDER_SLICES + slices + 'cutout.' + str(i)
            velocity = np.empty((256, 256, 3), dtype=dtype)
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
                for shift in np.arange(10, 256, 10):
                    v_shifted = np.roll(velocity, shift=shift, axis=1)     # shifting horizontally
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=shift, axis=0)     # shifting vertically
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(shift, shift), axis=(1, 0))  # shifting diagonally
                    data_test.append(v_shifted)
                    v_shifted = np.roll(velocity, shift=(-shift, shift), axis=(1, 0))  # shifting diagonally
                    data_test.append(v_shifted)
        number_of_examples += len(data_train)
        number_of_test += len(data_test)

        np.savez(os.path.join(folder, 'data_CNN{}.npz'.format(slices[:-1])), data_train=data_train, data_test=data_test)
        logging.info('Data of {} saved'.format(slices[:-1]))

    logging.info('Number of training and test examples are {} and {}'.format(number_of_examples, number_of_test))

    return number_of_examples, number_of_test


def save_shuffled_truth_data(data_all, ind, folder):
    """ Save truth data examples by one according to randomly shuffled indices.
    :param data_all: all truth training data
    :param ind:  randomly shuffled indices
    :param folder: path to save examples
    :return:
    """
    number_of_examples = len(ind)
    folder = os.path.join(folder, 'y_train')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for i in range(number_of_examples):
        print(i)
        np.savez(os.path.join(folder, '{}.npz'.format(i)), y_train=data_all[ind[i]])


def filter_and_save_shuffled_data(data_all, ind, filter_type, folder):
    """ Filter and save training data examples by one according to randomly shuffled indices.
    :param data_all: all truth training data
    :param ind: randomly shuffled indices
    :param filter_type: type of filtering applied
    :param folder: path to save filtered examples
    :return:
    """
    number_of_examples = len(ind)
    folder = os.path.join(folder, 'X_train_{}'.format(filter_type))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for i in range(number_of_examples):
        filtered = filtering(filter_type, data_all[ind[i]])
        print(i, np.dtype(filtered.dtype))
        np.savez(os.path.join(folder, '{}.npz'.format(i)), X_train=filtered)


def filter_and_save_test_data(data_test, filter_type, folder):
    """ Filter and save test data examples into .npz file.
    :param data_test: truth test data
    :param filter_type: type of filter applied
    :param folder: path to save filtered examples in .npz file.
    :return:
    """
    filtered = filtering_test(filter_type, data_test)
    np.savez(os.path.join(folder, 'X_test_{}.npz'.format(filter_type)), X_test=filtered)
    np.savez(os.path.join(folder, 'y_test.npz'), y_test=data_test)


def filtering(filter_type, data_train):
    """
    Filter example of train set with given filter type.
    :param filter_type: type of filter
    :param data_train: np.array of one train example
    :return: np.array of filtered fields
    """

    filtered_train = np.empty_like(data_train)
    sigma = filters.filter_size(filter_type)[0]
    if filter_type == 'gaussian':
        for key in range(3):
            filtered_train[:, :, key] = ndimage.gaussian_filter(data_train[:, :, key], sigma=sigma, mode='wrap',
                                                                truncate=500)
    elif filter_type == 'median':
        for key in range(3):
            filtered_train[:, :, key] = ndimage.median_filter(data_train[:, :, key], size=sigma, mode='wrap')

    elif filter_type == 'noise':
        for key in range(3):
            kappa = np.random.normal(0, 1, size=data_train[:, :, key].shape)
            filtered_train[:, :, key] = data_train[:, :, key] + sigma*kappa

    elif filter_type in ('fourier_sharp','physical_sharp'):
        for key in range(3):
            filtered_train[:, :, key] = filters.filter_sharp_array(data_train[:, :, key], filter_type=filter_type,
                                                                   scale_k=sigma)

    else:
        logging.error('Filter type is not defined.')

    return filtered_train


def filtering_test(filter_type, data_test):
    """
    Filter test example with three different filter scales.
    :param filter_type: type of filter
    :param data_test: np.array test example
    :return: np.array of 3 filtered examples
    """

    filtered_test = np.empty((3, ) + data_test.shape)
    print(filtered_test.shape)
    sigma = filters.filter_size(filter_type)
    if filter_type == 'gaussian':
        for i in range(3):
            for key in range(3):
                filtered_test[i, :, :, key] = ndimage.gaussian_filter(data_test[:, :, key], sigma=sigma[i],
                                                                      mode='wrap', truncate=500)
    elif filter_type == 'median':
        for i in range(3):
            for key in range(3):
                filtered_test[i, :, :, key] = ndimage.median_filter(data_test[:, :, key], size=sigma[i], mode='wrap')

    elif filter_type == 'noise':
        for i in range(3):
            for key in range(3):
                kappa = np.random.normal(0, 1, size=data_test[:, :, key].shape)
                filtered_test[i, :, :, key] = data_test[:, :, key] + sigma[i]*kappa

    elif filter_type in ('fourier_sharp','physical_sharp'):
        for i in range(3):
            for key in range(3):
                filtered_test[i, :, :, key] = filters.filter_sharp_array(data_test[:, :, key],
                                                                         filter_type=filter_type, scale_k=sigma[i])
    else:
        logging.error('Filter type is not defined.')

    return filtered_test


def load_slices_data(data_folder, size, n_channels, dtype):
    """ Load data from three big .npz files (xy, xz, yz, slices)
    :param data_folder: path to .npx files
    :param size: size of 2D slice, e.g. (256, 256)
    :param n_channels: number of channels, e.g. u,v,w velocity
    :param dtype: 32 of 64 bit
    :return: (np.array of loaded data, number of loaded examples)
    """
    n_examples = 4040  # number of examples in 1 slice
    data_all = np.empty((n_examples*3,) + size + (n_channels,), dtype=dtype)
    data_all[:n_examples] = np.load(os.path.join(data_folder, 'data_CNNx_y_slice.npz'))['data_train']
    print('x_y')
    data_all[n_examples:2 * n_examples] = np.load(os.path.join(data_folder, 'data_CNNx_z_slice.npz'))['data_train']
    print('x_z')
    data_all[2 * n_examples:] = np.load(os.path.join(data_folder, 'data_CNNy_z_slice.npz'))['data_train']
    print('y_z')
    print(data_all.shape)
    n_all = len(data_all)
    print(np.dtype(data_all.dtype))
    return data_all, n_all


def main():
    np.random.seed(1234)

    n_points_coarse = 256
    n_channels = 3
    size = (n_points_coarse, n_points_coarse)
    n_bit = 32
    if n_bit == 32:
        dtype = np.float32
    elif n_bit == 64:
        dtype = np.float64

    data_folder = DATA_FOLDER_BASE
    # data_folder = os.path.join(data_folder, '{}_bit'.format(n_bit))

    # Load in data from scv format
    # number_of_examples, number_of_test = load_data(data_folder, n_bit)

    data_all, n_all = load_slices_data(data_folder, size, n_channels, dtype)
    # shuffle indices
    ind = np.arange(n_all)
    np.random.shuffle(ind)
    # logging.info('Saving truth data')
    # save_shuffled_truth_data(data_all, ind, data_folder)

    filter_type = "gaussian"
    assert filter_type in ("gaussian", "median", "noise", "fourier_sharp", "physical_sharp"), \
        'Incorrect filter type: {}'.format(filter_type)
    logging.info('Saving filtered data')
    filter_and_save_shuffled_data(data_all, ind, filter_type, data_folder)
    del data_all
    tmp = np.load(os.path.join(data_folder, 'y_train/0.npz'))['y_train']
    print(np.dtype(tmp.dtype))

    # # test data
    # data_test = np.empty((3,) + size + (n_channels,))
    data_folder = os.path.join(data_folder, 'y_train')
    data_test = np.load(os.path.join(data_folder, '10000.npz'))['y_train']
    # print('x_y')
    # data_test[1111:2222] = np.load('/home/olga/data/examples/data_CNNx_z_slice.npz')['data_test']
    # print('x_z')
    # data_test[2222:] = np.load('/home/olga/data/examples/data_CNNy_z_slice.npz')['data_test']
    # print('y_z')
    print(data_test.shape)

    filter_and_save_test_data(data_test, filter_type, data_folder)


if __name__ == '__main__':
    main()
