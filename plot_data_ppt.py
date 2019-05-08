import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

import data
import utils

mpl.use('pdf')
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1. * plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1


data_folder = '../y_train/'
filename_npz = data_folder + '0.npz'
plot_folder = '../plots/'
if not os.path.isdir(plot_folder): os.makedirs(plot_folder)
velocity = np.load(filename_npz)['y_train']
print(velocity.shape)

cmap = plt.cm.jet

# plot full size (256, 256) truth data
for i, v in enumerate(['U', 'V', 'W']):
    axis = [0, 2 * np.pi, 0, 2 * np.pi]
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title('{} velocity'.format(v))
    im = ax.imshow(velocity[:, :, i], origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
    c_ax=plt.colorbar(im, fraction=0.0465, pad=0.044)
    c_ax.yaxis.set_ticks_position('left')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(plot_folder, v))

#plot zoomed (32, 32) size truth data
axis = [0, np.pi/8, 0, np.pi/8]
fig = plt.figure(figsize=(3, 3))
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
im = ax.imshow(velocity[:32, :32, 0], origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
# plt.colorbar(im, fraction=0.05, pad=0.04)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1)
fig.savefig(os.path.join(plot_folder, 'u_32'))

# filter data
x_noise = data.filtering('noise', velocity)
x_gaussian = data.filtering('gaussian', velocity)
x_sharp = data.filtering('fourier_sharp', velocity)
# calculate and save spectrum for truth and filtered data
utils.spectral_density(velocity, os.path.join(plot_folder, 'truth_data'))
utils.spectral_density(x_noise, os.path.join(plot_folder, 'noise_added'))
utils.spectral_density(x_gaussian, os.path.join(plot_folder, 'gaussian_filtered'))
utils.spectral_density(x_sharp, os.path.join(plot_folder, 'fourier_sharp_filtered'))

# plot spectra together
fig = plt.figure(figsize=(4.5, 3.2))
ax = plt.gca()
files = ['truth_data.spectra', 'noise_added.spectra', 'gaussian_filtered.spectra', 'fourier_sharp_filtered.spectra']
labels = ['truth data', 'noise', 'gaussian', 'fourier sharp']   #, 'fourier sharp',]
width = 1.5*np.ones(len(files))
width[0] = 3
for k in range(len(files)):
    f = open(os.path.join(plot_folder, files[k]), 'r')
    data = np.array(f.readlines()).astype(np.float)
    x = np.arange(len(data))
    ax.loglog(x, data, '-', linewidth=width[k], label=labels[k])
# y = 1e9 * np.power(x[1:], -5./3)
# ax.loglog(x[1:], y, 'r--', label=r'$-5/3$ slope')
ax.set_title('Spectra')
ax.set_ylabel(r'$E$')
ax.set_xlabel(r'k')
ax.axis(ymin=1e-2)
plt.legend(loc=0)
fig.subplots_adjust(left=0.17, right=0.95, bottom=0.2, top=0.87)
fig.savefig(os.path.join(plot_folder, 'spectra_filters'))
plt.close('all')

# plot zoomed examples of filtered filds
files = ['noise_added', 'gaussian_filtered', 'fourier_sharp_filtered']
for i, v in enumerate([x_noise, x_gaussian, x_sharp]):
    cmap = plt.cm.jet
    axis = [0, 2*np.pi, 0, 2*np.pi]
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title('{} velocity'.format(v))
    im = ax.imshow(v[:32, :32, 0], origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
    # c_ax=plt.colorbar(im, fraction=0.0465, pad=0.044)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(os.path.join(plot_folder, files[i]))

