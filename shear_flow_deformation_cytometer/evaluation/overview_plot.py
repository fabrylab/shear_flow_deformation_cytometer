# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in
# the file 'all_data.txt' located at the same directory as this script
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd
from scipy.stats.mstats import gmean

from shear_flow_deformation_cytometer.evaluation.helper_functions import get2Dhist_k_alpha_err
from shear_flow_deformation_cytometer.evaluation.helper_plot import plot_density_scatter
from shear_flow_deformation_cytometer.evaluation.helper_plot import plot_velocity_fit, plot_density_hist, \
    plot_density_levels, plot_binned_data

def overview_plot(datafile, data, config):
    plt.close('all')
    plt.figure(0, (14, 8))

    plt.subplot(2, 4, 1)
    plt.cla()
    plot_velocity_fit(data,config)
    plt.text(0.9, 0.9, f"$\\eta_0$ {data.eta0[0]:.2f}\n$\\delta$ {data.delta[0]:.2f}\n$\\tau$ {data.tau[0]:.2f}",
             transform=plt.gca().transAxes, va="top", ha="right")
    plt.text(0.1, 0.1, f"p {data.pressure[0]:.2f} bar", transform=plt.gca().transAxes, va="bottom", ha="left")

    plt.subplot(2, 4, (2, 3))
    plt.cla()
    #plot_density_scatter(data.stress, data.epsilon)
    #plot_binned_data(data.stress, data.epsilon, bins=np.arange(0, 300, 10))
    plot_density_scatter(data.stress, data.strain)
    plot_binned_data(data.stress, data.strain, bins=np.arange(0, 300, 10))
    plt.text(0.9, 0.9, f"#cells {data.shape[0]}", transform=plt.gca().transAxes, va="top", ha="right")

    plt.xlabel("stress [Pa]")
    plt.ylabel("strain ")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.subplot(2, 4, 4)
    plt.cla()
    plot_density_scatter(data.radial_position, data.angle)
    plot_binned_data(data.radial_position, data.angle, bins=np.arange(-300, 300, 10))
    plt.xlabel("radial position (µm)")
    plt.ylabel("angle (deg)")

    plt.subplot(2, 4, 5)
    plt.loglog(data.omega, data.Gp1, "o", alpha=0.25)
    plt.loglog(data.omega, data.Gp2, "o", alpha=0.25)
    plt.ylabel("G' / G''")
    plt.xlabel("angular frequency")

    plt.subplot(2, 4, 6)
    plt.cla()
    plt.xlim(0, 4)
    plot_density_hist(np.log10(data.k), color="C0")
    plt.xlabel("log10(k)")
    plt.ylabel("relativ density")
    plt.text(0.1, 0.9,
             f"$\\overline{{k}}_g$ {gmean(data.k):.2f}\nstd(log10(k)) {np.std(np.log10(data.k)):.2f}\n$\overline{{k}}$ {np.mean(data.k):.2f}\nstd(k) {np.std(data.k):.2f}\n",
             transform=plt.gca().transAxes, va="top", ha="left")

    plt.subplot(2, 4, 7)
    plt.cla()
    plt.xlim(0, 1)
    plot_density_hist(data.alpha, color="C1")
    plt.xlabel("alpha")
    plt.text(0.9, 0.9,
             f"mean($\\alpha$) {np.mean(data.alpha):.2f}\nstd($\\alpha$) {np.std(data.alpha):.2f}\n",
             transform=plt.gca().transAxes, va="top", ha="right")

    plt.subplot(2, 4, 8)
    plt.cla()
    plot_density_scatter(data.k, data.alpha, logx=True)
    bootstrap_repitions = 10
    k, k_err, alpha, alpha_err = get2Dhist_k_alpha_err(data, bootstrap_repetitions=bootstrap_repitions)
    plt.xlabel("stiffness k (Pa)")
    plt.ylabel("fluidity $\\alpha$")
    plt.xlim(left=np.min(data.k))
    plt.ylim(0, 1)
    plt.semilogx()
    plt.axhline(alpha, linestyle='-', color='black', linewidth=0.5)
    plt.axvline(k, linestyle='-', color='black', linewidth=0.5)
    plt.text(0.9, 0.9,
             f"$\\alpha$  {alpha:.2f}$\\pm${alpha_err / np.sqrt(bootstrap_repitions):.2f}\nk {k:.2f}$\\pm${k_err / np.sqrt(bootstrap_repitions):.2f}",
             transform=plt.gca().transAxes, va="top", ha="right")

    # import matplotlib
    # locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
    # ax.xaxis.set_major_locator(locmaj)
    # locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    # ax.xaxis.set_minor_locator(locmin)
    # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.tight_layout()
    datafile = str(datafile)
    if datafile.endswith(".csv"):
        datafile = datafile[:-4] + ".pdf"
        print(datafile[-4])
    if datafile.endswith(".tif"):
        datafile = datafile[:-4] + "_evaluated.pdf"
    try:
        plt.savefig(datafile)
    except PermissionError:
        pass