import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from shear_flow_deformation_cytometer.evaluation.helper_functions import bootstrap_error
from shear_flow_deformation_cytometer.includes.fit_velocity import getFitXY


def plot_density_scatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, s=5, levels=None, loglog=False, ax=None):
    ax = plt.gca() if ax is None else ax
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    filter = ~np.isnan(x) & ~np.isnan(y)
    if loglog is True:
        filter &= (x > 0) & (y > 0)
    x = x[filter]
    y = y[filter]
    if loglog is True:
        xy = np.vstack([np.log10(x), np.log10(y)])
    else:
        xy = np.vstack([x, y * y_factor])
    try:
        kde = gaussian_kde(xy)
        kd = kde(xy)
        idx = kd.argsort()
        x, y, z = x[idx], y[idx], kd[idx]
    except ValueError as err:
        print(err, file=sys.stderr)
        z = np.ones_like(x)
    ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap)  # plot in kernel density colors e.g. viridis

    if levels != None:
        X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        XY = np.dstack([X, Y * y_factor])
        Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
        ax.contour(X, Y, Z, levels=1)

    if loglog is True:
        ax.loglog()
    return ax


def plot_density_levels(x, y, skip=1, y_factor=1, levels=None, cmap="viridis", colors=None):
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    filter = ~np.isnan(x) & ~np.isnan(y)
    x = x[filter]
    y = y[filter]
    xy = np.vstack([x, y * y_factor])
    kde = gaussian_kde(xy)
    kd = kde(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]

    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    XY = np.dstack([X, Y * y_factor])
    Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
    plt.contour(X, Y, Z, levels=levels, cmap=cmap, colors=colors)

    #


def plot_binned_data(x, y, bins, bin_func=np.median, error_func=None, color="black", xscale="normal", **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    if xscale == "log":
        x = np.log10(x)
    strain_av = []
    stress_av = []
    strain_err = []
    for i in range(len(bins) - 1):
        index = (bins[i] < x) & (x < bins[i + 1])
        yy = y[index]
        yy = yy[~np.isnan(yy)]
        if len(yy) == 0:
            continue
        strain_av.append(bin_func(yy))
        # yy = yy[yy>0]
        # strain_err.append(np.std(np.log(yy)) / np.sqrt(len(yy)))
        if error_func is None:
            strain_err.append(bootstrap_error(yy, bin_func))  # np.quantile(yy, [0.25, 0.75]))
        elif error_func == "quantiles":
            strain_err.append(np.abs(np.quantile(yy, [0.25, 0.75]) - bin_func(yy)))  # np.quantile(yy, [0.25, 0.75]))

        stress_av.append(np.median(x[index]))
    plot_kwargs = dict(marker='s', mfc="white", mec=color, ms=7, mew=1, lw=0, ecolor='black', elinewidth=1, capsize=3)
    plot_kwargs.update(kwargs)
    x, y, yerr = np.array(stress_av), np.array(strain_av), np.array(strain_err).T
    index = ~np.isnan(x) & ~np.isnan(y)
    x = x[index]
    y = y[index]
    if xscale == "log":
        yerr = yerr[index]
        x = 10 ** x
    plt.errorbar(x, y, yerr=yerr, **plot_kwargs)

    return x, y


def all_plots_same_limits():
    xmin = np.min([ax.get_xlim()[0] for ax in plt.gcf().axes])
    xmax = np.max([ax.get_xlim()[1] for ax in plt.gcf().axes])
    ymin = np.min([ax.get_ylim()[0] for ax in plt.gcf().axes])
    ymax = np.max([ax.get_ylim()[1] for ax in plt.gcf().axes])
    for ax in plt.gcf().axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def plot_velocity_fit(data, color=None, parameters=None, color_fit="k"):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXY(config, np.mean(pressure), p)
        return x, y

    maxima = []
    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d.delta = d.delta.round(8)
        d.tau = d.tau.round(10)
        d.eta0 = d.eta0.round(8)
        d = d.set_index(["eta0", "delta", "tau"]).copy()
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p if parameters is None else parameters)
            line, = plt.plot(np.abs(dd.radial_position), dd.measured_velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x >= 0] * 1e+6, y[x >= 0] * 1e2, color=color_fit)
            maxima.append(np.nanmax(y[x > 0] * 1e2))
    try:
        plt.ylim(top=np.nanmax(maxima) * 1.1)
    except ValueError:
        pass
    plt.xlabel("position in channel (µm)")
    plt.ylabel("velocity (cm/s)")
    plt.ylim(bottom=0)


def plot_density_hist(x, orientation='vertical', do_stats=True, only_kde=False, ax=None, bins=50, **kwargs):
    ax = ax if not ax is None else plt.gca()
    from scipy import stats
    x = np.array(x)
    x = x[np.isfinite(x)]
    if len(x) != 0:
        kde = stats.gaussian_kde(x)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
        if orientation == 'horizontal':
            l, = ax.plot(kde(xx), xx, **kwargs)
        else:
            l, = ax.plot(xx, kde(xx), **kwargs)
        if not only_kde:
            ax.hist(x, bins=bins, density=True, color=l.get_color(), alpha=0.5, orientation=orientation)
    else:
        l, = ax.plot([], [], **kwargs)
    return l


def plot_joint_density(x, y, label=None, only_kde=False, color=None, growx=1, growy=1, offsetx=0):
    ax = plt.gca()
    x1, y1, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height

    wf, hf = ax.figure.get_size_inches()
    gap = 0.05
    fraction = 0.2
    width_of_hist = np.mean([(w * wf * fraction), (h * hf * fraction)])
    hist_w = width_of_hist / wf
    hist_h = width_of_hist / hf

    h *= growy
    w *= growx
    if getattr(ax, "ax2", None) is None:
        ax.ax2 = plt.axes([x1 + offsetx * w, y1 + h - hist_h + gap / hf, w - hist_w, hist_h], sharex=ax,
                          label=ax.get_label() + "_top")
        # ax.ax2.set_xticklabels([])
        ax.ax2.spines['right'].set_visible(False)
        ax.ax2.spines['top'].set_visible(False)
        ax.ax2.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        ax.ax2.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax2.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.sca(ax.ax2)
    plot_density_hist(x, color=color, only_kde=only_kde)
    if getattr(ax, "ax3", None) is None:
        ax.ax3 = plt.axes([x1 + offsetx * w + w - hist_w + gap / wf, y1, hist_w, h - hist_h], sharey=ax,
                          label=ax.get_label() + "_right")
        # ax.ax3.set_yticklabels([])
        ax.set_position([x1 + offsetx * w, y1, w - hist_w, h - hist_h])
        ax.ax3.spines['right'].set_visible(False)
        ax.ax3.spines['top'].set_visible(False)
        ax.ax3.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax3.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        # ax.ax3.spines['left'].set_visible(False)
        ax.ax3.spines['bottom'].set_visible(False)

        ax.spines['right'].set_visible(False)
    plt.sca(ax.ax3)
    l = plot_density_hist(y, color=color, orientation=u'horizontal', only_kde=only_kde)
    plt.sca(ax)
    plot_density_levels(x, y, levels=1, colors=[l.get_color()], cmap=None)
    plt.plot([], [], color=l.get_color(), label=label)


def split_axes(ax=None, join_x_axes=False, join_title=True):
    if ax is None:
        ax = plt.gca()
    x1, y1, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height
    gap = w * 0.02
    if getattr(ax, "ax2", None) is None:
        ax.ax2 = plt.axes([x1 + w * 0.5 + gap, y1, w * 0.5 - gap, h], label=ax.get_label() + "_twin")
        ax.set_position([x1, y1, w * 0.5 - gap, h])
        # ax.ax2.set_xticklabels([])
        ax.ax2.spines['right'].set_visible(False)
        ax.ax2.spines['top'].set_visible(False)
        # ax.ax2.spines['left'].set_visible(False)
        ax.ax2.tick_params(axis='y', colors='gray', which="both", labelcolor="none")
        ax.ax2.spines['left'].set_color('gray')
        if join_title is True:
            ax.set_title(ax.get_title()).set_position([1.0 + gap, 1.0])
        if join_x_axes is True:
            t = ax.set_xlabel(ax.get_xlabel())
            t.set_position([1 + gap, t.get_position()[1]])
    plt.sca(ax.ax2)


def joined_hex_bin(x1, y1, x2, y2, loglog=True):
    import matplotlib
    def filter(x, y):
        filter = ~np.isnan(x) & ~np.isnan(y)
        if loglog is True:
            filter &= (x > 0) & (y > 0)
        return x[filter], y[filter]

    from scipy.stats import gaussian_kde
    d = np.array(filter(x1, y1))
    d2 = np.array(filter(x2, y2))

    def darken(color, f, a=None):
        from matplotlib import colors
        c = np.array(colors.to_rgba(color))
        if f > 0:
            c2 = np.zeros(3)
        else:
            c2 = np.ones(3)
        c[:3] = c[:3] * (1 - np.abs(f)) + np.abs(f) * c2
        if a is not None:
            c[3] = a
        return c

    d0 = np.hstack((d, d2))
    p = plt.hexbin(d0[0], d0[1], gridsize=200, mincnt=1, xscale="log", yscale="log")
    if loglog is False:
        points = np.array([np.mean(p.vertices, axis=0) for p in p.get_paths()]).T
        kde1 = gaussian_kde(d)(points)
        kde2 = gaussian_kde(d2)(points)
    else:
        points = np.array([np.power(10, np.mean(np.log10(p.vertices), axis=0)) for p in p.get_paths()]).T
        kde1 = gaussian_kde(np.log10(d))(np.log10(points))
        kde2 = gaussian_kde(np.log10(d2))(np.log10(points))

    kde1 /= np.max(kde1)
    kde2 /= np.max(kde2)

    color1max = np.array(matplotlib.colors.to_rgba("C0"))
    color1min = darken(color1max, -0.75)
    color2max = np.array(matplotlib.colors.to_rgba("C1"))
    color2min = darken(color2max, -0.75)

    # colormap("C0", -0.75, 0), colormap("C1", -0.75, 0)

    color12max = color1max + color2max
    color12max /= np.max(color12max)
    color12min = color1min + color2min
    color12min /= np.max(color12min)

    c1 = color1min[None] * (1 - kde1[:, None]) + color1max[None] * kde1[:, None]
    c2 = color2min[None] * (1 - kde2[:, None]) + color2max[None] * kde2[:, None]

    kde12 = kde1 + kde2
    kde12 /= np.max(kde12)
    c12 = color12min[None] * (1 - kde2[:, None]) + color12max[None] * kde2[:, None]

    r1 = kde1 / (kde2 + 1e-6)
    r2 = kde2 / (kde1 + 1e-6)
    r1 = r1[:, None]
    r2 = r2[:, None]
    c = (c12 * r1 + c2 * (1 - r1)) * (r1 < 1) + (c1 * (1 - r2) + c12 * r2) * (r2 < 1)
    c[c > 1] = 1
    c[c < 0] = 0

    p.set_color(c)
    p.set_array(None)
