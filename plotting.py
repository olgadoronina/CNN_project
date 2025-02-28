import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.use('pdf')
import gc
import logging
import utils
import os
import filters


fig_width_pt = 469.75502  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

mpl.rcParams['figure.figsize'] = fig_size
# plt.rcParams['figure.autolayout'] = True

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


def plot_loss_per_epoch(plot_folder, epochs, history):
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax.semilogy(range(1, epochs + 1), history['loss'], color="steelblue", marker="o", label="training")
    ax.semilogy(range(1, epochs + 1), history['val_loss'], color="green", marker="o", label="validation")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_xticks(range(1, epochs + 2, 2))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.95)

    fig.savefig(os.path.join(plot_folder, 'loss_per_epoch'))

    plt.close('all')


def plot_velocities_and_spectra(x_test, y_test, y_predict, plot_folder):
    logging.info('Plot predicted velocities')
    dimension = len(y_test[:, :, 0].shape)
    if dimension == 2:
        for test_example in range(3):
            imagesc([y_test[0:, 0:, 0],
                     x_test[test_example, :, :, 0],
                     y_predict[test_example, :, :, 0]],
                    [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'],
                    os.path.join(plot_folder, 'u_{}'.format(test_example)))
            imagesc([y_test[0:32, 0:32, 1],
                     x_test[test_example, 0:32, 0:32, 1],
                     y_predict[test_example, 0:32, 0:32, 1]],
                    [r'$v_{true}$', r'$v_{filtered}$', r'$v_{predicted}$'],
                    os.path.join(plot_folder, 'v_{}'.format(test_example)))
            imagesc([y_test[0:32, 0:32, 2],
                     x_test[test_example, 0:32, 0:32, 2],
                     y_predict[test_example, 0:32, 0:32, 2]],
                    [r'$w_{true}$', r'$w_{filtered}$', r'$w_{predicted}$'],
                    os.path.join(plot_folder, 'w_{}'.format(test_example)))
    # else:
    #     for test_example in range(3):
    #         imagesc([y_test[test_example]['u'][0:32, 0:32, 32],
    #                  x_test[test_example]['u'][0:32, 0:32, 32],
    #                  y_predict[test_example]['u'][0:32, 0:32, 32]],
    #                 [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'],
    #                 os.path.join(plot_folder, 'u_{}'.format(test_example)))
    #         imagesc([y_test[test_example]['v'][0:32, 0:32, 32],
    #                  x_test[test_example]['v'][0:32, 0:32, 32],
    #                  y_predict[test_example]['v'][0:32, 0:32, 32]],
    #                 [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'],
    #                 os.path.join(plot_folder, 'v_{}'.format(test_example)))
    #         imagesc([y_test[test_example]['w'][0:32, 0:32, 32],
    #                  x_test[test_example]['w'][0:32, 0:32, 32],
    #                  y_predict[test_example]['w'][0:32, 0:32, 32]],
    #                 [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'],
    #                 os.path.join(plot_folder, 'w_{}'.format(test_example)))

    logging.info('Calculate and plot spectra')
    for test_example in range(3):
        utils.spectral_density(y_test,
                               os.path.join(plot_folder, 'true{}'.format(test_example)))
        utils.spectral_density(x_test[test_example],
                               os.path.join(plot_folder, 'filtered{}'.format(test_example)))
        utils.spectral_density(y_predict[test_example],
                                os.path.join(plot_folder, 'predicted{}'.format(test_example)))

        spectra(plot_folder, os.path.join(plot_folder, 'spectra{}'.format(test_example)), test_example)







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
    gc.collect()
    plt.close()


def spectra(folder, fname, ind):

    ind = str(ind)
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    if ind == '':
        files = ['coarse_grid.spectra', 'gaussian.spectra', 'median.spectra', 'noise.spectra', 'physical_sharp.spectra']  #, 'fourier_sharp.spectra', ]
        labels = ['true data', 'gaussian', 'median', 'noise', 'physical sharp']   #, 'fourier sharp',]
        width = 1.5*np.ones(len(files))
        width[0] = 3
    else:
        files = ['predicted' + ind + '.spectra', 'filtered' + ind + '.spectra', 'true' + ind + '.spectra']
        labels = ['predicted', 'filtered', 'true']
        width = [2, 2, 2]
    for k in range(len(files)):
        f = open(os.path.join(folder, files[k]), 'r')
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

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(fname)
    plt.close('all')


def plot_tau(x_test, y_test, y_predict, plot_folder, filter_type, y_predict2=None):
    """Plot Reynolds stresses tau fields and pdf."""
    dimension = len(y_test[0]['u'].shape)
    k = filters.filter_size(filter_type, dimension)

    if len(y_test[0]['u'].shape) == 3:
        logging.info('Plot tau')
        for test_example in range(3):
            tau = dict()
            tau2 = dict()
            tau_true = dict()
            for u_i in ['u', 'v', 'w']:
                for u_j in ['u', 'v', 'w']:
                    tmp = y_predict[test_example][u_i]*y_predict[test_example][u_j]
                    tmp = filters.filter(tmp, filter_type, k=k[test_example])
                    tau[u_i+u_j] = x_test[test_example][u_i]*x_test[test_example][u_j] - tmp
                    tmp = y_test[test_example][u_i]*y_test[test_example][u_j]
                    tmp = filters.filter(tmp, filter_type, k=k[test_example])
                    tau_true[u_i + u_j] = x_test[test_example][u_i] * x_test[test_example][u_j] - tmp
                    if y_predict2:
                        tmp = y_predict2[test_example][u_i]*y_predict2[test_example][u_j]
                        tmp = filters.filter(tmp, filter_type, k=k[test_example])
                        tau2[u_i + u_j] = x_test[test_example][u_i] * x_test[test_example][u_j] - tmp

            imagesc([tau_true['uu'][:, :, 32], tau_true['uv'][:, :, 32], tau_true['uw'][:, :, 32]],
                    titles=[r'$\tau^{true}_{11}$', r'$\tau^{true}_{12}$', r'$\tau^{true}_{13}$'],
                    name=os.path.join(plot_folder, 'tau_true{}'.format(test_example)), limits=[-0.07, 0.07])
            imagesc([tau['uu'][:, :, 32], tau['uv'][:, :, 32], tau['uw'][:, :, 32]],
                    titles=[r'$\tau_{11}$', r'$\tau_{12}$', r'$\tau_{13}$'],
                    name=os.path.join(plot_folder, 'tau{}'.format(test_example)), limits=[-0.07, 0.07])

            fig, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(6.5, 2.4))
            titles = [r'$\tau_{11}$', r'$\tau_{12}$', r'$\tau_{13}$']
            for ind, key in enumerate(['uu', 'uv', 'uw']):
                x, y = utils.pdf_from_array_with_x(tau_true[key], bins=100, range=[-0.1, 0.1])
                axarr[ind].semilogy(x, y, 'r-', lw=2, label='true')
                x, y = utils.pdf_from_array_with_x(tau[key], bins=100, range=[-0.1, 0.1])
                axarr[ind].semilogy(x, y, 'b-', label='modeled')
                if y_predict2:
                    x, y = utils.pdf_from_array_with_x(tau2[key], bins=100, range=[-0.1, 0.1])
                    axarr[ind].semilogy(x, y, 'g-', label='modeled 2')
                axarr[ind].set_xlabel(titles[ind])
            axarr[0].axis(xmin=-0.1, xmax=0.1, ymin=1e-3)
            axarr[0].set_ylabel('pdf')
            axarr[0].set_yscale('log')
            plt.legend(loc=0)
            fig.subplots_adjust(left=0.1, right=0.95, wspace=0.16, bottom=0.2, top=0.9)
            fig.savefig(os.path.join(plot_folder, 'tau_pdf{}'.format(test_example)))
            # print(os.path.join(plot_folder, 'tau_pdf{}'.format(test_example)))
            plt.close('all')


def plot_vorticity_pdf(x_test, y_test, y_predict, plot_folder, y_predict2=None):
    """ Plot normalized vorticity pdf."""
    shape = y_predict[0]['u'].shape
    dimension = len(shape)
    dx = np.divide([np.pi]*3, np.array(shape))
    logging.info('Plot vorticity pdf')
    for test_example in range(3):
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()

        if dimension == 2:
            _, du_dy = np.gradient(y_test[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(y_test[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='true')

            _, du_dy = np.gradient(x_test[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(x_test[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='filtered')

            _, du_dy = np.gradient(y_predict[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(y_predict[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='modeled')

            if y_predict2:
                _, du_dy = np.gradient(y_predict2[test_example]['u'], dx[0], dx[1])
                dv_dx, _ = np.gradient(y_predict2[test_example]['v'], dx[0], dx[1])
                vorticity = dv_dx - du_dy
                vorticity /= np.max(np.abs(vorticity))
                x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
                ax.semilogy(x, y, label='modeled 2')

        elif dimension == 3:
            vort = utils.calc_vorticity_magnitude(y_test[test_example])
            x, y = utils.pdf_from_array_with_x(vort.flatten(), bins=100, range=[-1, 1])
            ax.semilogy(x, y, 'r-', label='true')

            vort = utils.calc_vorticity_magnitude(x_test[test_example])
            x, y = utils.pdf_from_array_with_x(vort.flatten(), bins=100, range=[-1, 1])
            ax.semilogy(x, y, 'y-', label='filtered')

            vort = utils.calc_vorticity_magnitude(y_predict[test_example])
            x, y = utils.pdf_from_array_with_x(vort.flatten(), bins=100, range=[-1, 1])
            ax.semilogy(x, y, 'b-', label='modeled')

            if y_predict2:
                vort = utils.calc_vorticity_magnitude(y_predict2[test_example])
                x, y = utils.pdf_from_array_with_x(vort.flatten(), bins=100, range=[-1, 1])
                ax.semilogy(x, y, 'g-', label='modeled 2')

        # ax.set_title('Vorticity pdf')
        ax.set_ylabel('pdf')
        ax.set_xlabel(r'$\omega$')
        ax.axis(ymin=1e-3)
        plt.legend(loc=0)

        fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.95)
        fig.savefig(os.path.join(plot_folder, 'omega_{}'.format(test_example)))
        print(os.path.join(plot_folder, 'omega_{}'.format(test_example)))
        plt.close('all')








########################################################################################################################





