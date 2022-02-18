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

from shear_flow_deformation_cytometer.evaluation.helper_plot import plot_density_scatter
from shear_flow_deformation_cytometer.evaluation.helper_plot import plot_velocity_fit, plot_density_hist, \
    plot_density_levels, plot_binned_data


def overview_plot(datafile, data):

    plt.figure(0, (10, 8))
    plt.clf()

    plt.subplot(2, 3, 1)
    plt.cla()
    plot_velocity_fit(data)
    plt.text(0.9, 0.9, f"{data.pressure[0]:.2f} bar\n$\\eta_0$ {data.eta0[0]:.2f}\n$\\delta$ {data.delta[0]:.2f}\n$\\tau$ {data.tau[0]:.2f}", transform=plt.gca().transAxes, va="top", ha="right")

    plt.subplot(2, 3, 2)
    plt.cla()
    plot_density_scatter(data.stress, data.strain)
    plot_binned_data(data.stress, data.strain, bins=np.arange(0, 300, 10))
    plt.xlabel("stress (Pa)")
    plt.ylabel("strain")

    plt.subplot(2, 3, 3)
    plt.cla()
    plot_density_scatter(data.radial_position, data.angle)
    plot_binned_data(data.radial_position, data.angle, bins=np.arange(-300, 300, 10))
    plt.xlabel("radial position (µm)")
    plt.ylabel("angle (deg)")

    plt.subplot(2, 3, 4)
    plt.loglog(data.omega, data.Gp1, "o", alpha=0.25)
    plt.loglog(data.omega, data.Gp2, "o", alpha=0.25)
    plt.ylabel("G' / G''")
    plt.xlabel("angular frequency")

    plt.subplot(2, 3, 5)
    plt.cla()
    plt.xlim(0, 4)
    plot_density_hist(np.log10(data.k), color="C0")
    plt.xlabel("log10(k)")
    plt.ylabel("relative density")
    plt.text(0.9, 0.9, f"mean(log10(k)) {np.mean(np.log10(data.k)):.2f}\nstd(log10(k)) {np.std(np.log10(data.k)):.2f}\nmean(k) {np.mean(data.k):.2f}\nstd(k) {np.std(data.k):.2f}\n", transform=plt.gca().transAxes, va="top", ha="right")

    plt.subplot(2, 3, 6)
    plt.cla()
    plt.xlim(0, 1)
    plot_density_hist(data.alpha, color="C1")
    plt.xlabel("alpha")
    plt.text(0.9, 0.9, f"mean($\\alpha$) {np.mean(data.alpha):.2f}\nstd($\\alpha$) {np.std(data.alpha):.2f}\n", transform=plt.gca().transAxes, va="top", ha="right")

    plt.tight_layout()
    datafile = str(datafile)
    if datafile.endswith(".csv"):
        datafile = datafile[:-4] + ".pdf"
    if datafile.endswith(".tif"):
        datafile = datafile[:-4] + "_evaluated.pdf"
    try:
        plt.savefig(datafile)
    except PermissionError:
        pass
