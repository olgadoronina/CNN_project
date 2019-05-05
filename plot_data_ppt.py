import numpy as np
import logging
import scipy.ndimage as ndimage
import plotting
import utils
import filters
import os

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('dark_background')

mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
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
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1




def imagesc(Arrays, titles, name=None, limits=None):
    axis = [0, np.pi/8, 0, np.pi/8]

    cmap = plt.cm.jet  # define the colormap
    # cmap = plt.cm.binary
    if limits==None:
        norm = mpl.colors.Normalize(vmin=-0.7, vmax=0.7)
    else:
        norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])

    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(10, 4))
        # fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(4, 2.5))

        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            ax.set_adjustable('box-forced')
            ax.set_xlabel(r'$x$')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            k += 1
        axes[0].set_ylabel(r'$y$')
        cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        fig.subplots_adjust(left=0.07, right=0.87, wspace=0.1, bottom=0.05, top=0.98)
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        axis = [0, np.pi / 8, 0, np.pi / 8]
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(titles[0])
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
        plt.colorbar(im, fraction=0.05, pad=0.04)
    if name:
        fig.savefig(name)
    del ax, im, fig, cmap
    plt.close()




data_folder = '../data/'
filename_npz = data_folder + 'isotropic2048.npz'
plot_folder = data_folder + 'plots/'
velocity = np.load(filename_npz)
print(velocity.keys())

cmap = plt.cm.jet
axis = [0, 2*np.pi, 0, 2*np.pi]
fig = plt.figure(figsize=(2.5, 2.8))
ax = plt.gca()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('velocity')
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.yaxis.set_major_formatter(plt.NullFormatter())
im = ax.imshow(velocity['v'], origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
# plt.colorbar(im, fraction=0.05, pad=0.04)
fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.88)
fig.savefig(os.path.join(plot_folder, 'u_2048'))


cmap = plt.cm.jet
axis = [0, np.pi/8, 0, np.pi/8]
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
im = ax.imshow(velocity['v'][-int(2048/16):, -int(2048/16):], origin='lower', cmap=cmap, interpolation="nearest", extent=axis)
# plt.colorbar(im, fraction=0.05, pad=0.04)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1)
fig.savefig(os.path.join(plot_folder, 'u_128'))