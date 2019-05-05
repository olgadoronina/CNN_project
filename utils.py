import sys
import numpy as np
from numpy.fft import fftfreq, fft2, fftn
import logging
import nn_keras as nnk
import plotting
import os


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>2}:{:05.2f} \t {}".format(int(minutes), seconds, label))
    return "{:0>2}:{:05.2f}".format(int(minutes), seconds)


def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def calc_vorticity_magnitude(vel_dict):
    shape = vel_dict['u'].shape
    assert len(shape) == 3, "Incorrect dimension for vorticity calculation"
    dx = np.divide([np.pi] * 3, np.array(shape))

    vorticity = np.empty((3, shape[0], shape[1], shape[2]))
    du_dx, du_dy, du_dz = np.gradient(vel_dict['u'], dx[0], dx[1], dx[2])
    dv_dx, dv_dy, dv_dz = np.gradient(vel_dict['v'], dx[0], dx[1], dx[2])
    dw_dx, dw_dy, dw_dz = np.gradient(vel_dict['w'], dx[0], dx[1], dx[2])
    vorticity[0] = dw_dy - dv_dz
    vorticity[1] = du_dz - dw_dx
    vorticity[2] = dv_dx - du_dy
    vorticity /= np.max(np.abs(vorticity))
    # vort_magnitude = np.sqrt(vorticity[0] ** 2 + vorticity[1] ** 2 + vorticity[2] ** 2)
    return vorticity


def shell_average(spect, N_points, k):
    """ Compute the 1D, shell-averaged, spectrum of the 2D or 3D Fourier-space
    variable.
    :param spect: 2D or 3D complex or real Fourier-space scalar
    :return: 1D, shell-averaged, spectrum
    """
    i = 0
    F_k = np.zeros(tuple(N_points)).flatten()
    k_array = np.empty_like(F_k)
    for ind_x, kx in enumerate(k[0]):
        for ind_y, ky in enumerate(k[1]):
            if len(N_points) == 2:
                k_array[i] = round(np.sqrt(kx**2 + ky**2))
                F_k[i] = np.pi*k_array[i]*spect[ind_x, ind_y]
                i += 1
            else:
                for ind_z, kz in enumerate(k[2]):
                    k_array[i] = round(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2))
                    F_k[i] = 2 * np.pi * k_array[i] ** 2 * spect[ind_x, ind_y, ind_z]
                    i += 1

    all_F_k = sorted(list(zip(k_array, F_k)))

    x, y = [all_F_k[0][0]], [all_F_k[0][1]]
    n = 1
    for k, F in all_F_k[1:]:
        if k == x[-1]:
            n += 1
            y[-1] += F
        else:
            y[-1] /= n
            x.append(k)
            y.append(F)
            n = 1
    return x, y


def spectral_density(vel_array, fname):
    """
    Write the 1D power spectral density of var to text file. Method
    assumes a real input in physical space.
    """
    N_points = np.array(vel_array.shape[:-1])
    dx = 2*np.pi/N_points
    spectrum = 0
    if len(N_points) == 2:
        k = 2 * np.pi * np.array([fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1])])
        for key in range(3):
            fft_array = fft2(vel_array[:, :, key])
            spectrum += np.real(fft_array * np.conj(fft_array))
    elif len(N_points) == 3:
        k = 2 * np.pi * np.array(
            [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])])
        for key in range(3):
            fft_array = fftn(vel_array[:, :, :, key])
            spectrum += np.real(fft_array * np.conj(fft_array))

    # logging.debug('done transform')
    x, y = shell_average(spectrum, N_points, k)
    # logging.debug('done shell average')

    fh = open(fname + '.spectra', 'w')
    fh.writelines(["%s\n" % item for item in y])
    fh.close()


def save_results(plot_folder, predictions, true, mse):
    header = "sigma=0.9, sigma=1.0, sigma=1.1"
    prediction_file = os.path.join(plot_folder, 'y_predictions.txt')
    true_file = os.path.join(plot_folder, 'y_actual.txt')
    mse_file = os.path.join(plot_folder, 'mse.txt')
    for i in range(len(true)):
        predictions[i] = predictions[i].flatten()
        true[i] = true[i].flatten()
    np.savetxt(prediction_file, predictions, header=header)
    np.savetxt(true_file, true, header=header)
    np.savetxt(mse_file, mse, header=header)


def save_loss_per_epoch(plot_folder, train_loss, val_loss):
    header = "train_loss, val_loss"
    train_loss_array = np.array(train_loss)
    val_loss_array = np.array(val_loss)
    final = np.column_stack((train_loss_array, val_loss_array))
    epoch_file = os.path.join(plot_folder, "loss_per_epoch.txt")
    np.savetxt(epoch_file, final, header=header)

