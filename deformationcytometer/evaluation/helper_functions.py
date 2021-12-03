import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def stressfunc(radial_position: np.ndarray, pressure: np.ndarray,
               channel_length: np.ndarray, channel_height: np.ndarray) -> np.ndarray:
    radial_position = np.asarray(radial_position)
    G = pressure / channel_length  # pressure gradient
    pre_factor = (4 * (channel_height ** 2) * G) / np.pi ** 3
    # sum only over odd numbers
    n = np.arange(1, 100, 2)[None, :]
    u_primy = pre_factor * np.sum(((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * channel_height)) \
                                  * (np.sinh((n * np.pi * radial_position[:, None]) / channel_height) / np.cosh(n * np.pi / 2)), axis=1)

    stress = np.abs(u_primy)
    return stress


def getVelocity(data: pd.DataFrame, config: dict):
    """ match cells in consecutive frames to measure their velocity """
    velocities = np.zeros(data.shape[0]) * np.nan
    cell_id = data.index.to_numpy()
    velocity_partner = np.zeros(data.shape[0], dtype="<U100")
    for i in data.index[:-10]:
        for j in range(10):
            try:
                data.radial_position[i + j]
            except KeyError:
                continue
            if np.abs(data.radial_position[i] - data.radial_position[i + j]) < 1 \
                    and data.frame[i + j] - data.frame[i] == 1 \
                    and np.abs(data.long_axis[i + j] - data.long_axis[i]) < 1 \
                    and np.abs(data.short_axis[i + j] - data.short_axis[i]) < 1 \
                    and np.abs(data.angle[i + j] - data.angle[i]) < 5:

                dt = data.timestamp[i + j] - data.timestamp[i]
                v = (data.x[i + j] - data.x[i]) * config["pixel_size"] / dt  # in mm/s
                if v > 0:
                    velocities[i] = v
                    velocity_partner[
                        i] = f"{i}, {i + j}, {dt}, {data.x[i + j] - data.x[i]}, {data.frame[i]}, {data.long_axis[i]}, {data.short_axis[i]} -> {data.frame[i + j]}, {data.long_axis[i + j]}, {data.short_axis[i + j]}"
                    cell_id[i + j] = cell_id[i]
    data["velocity"] = velocities
    data["velocity_partner"] = velocity_partner
    data["cell_id"] = cell_id


def getStressStrain(data: pd.DataFrame, config: dict):
    """ calculate the stress and the strain of the cells """
    r = np.sqrt(data.long_axis / 2 * data.short_axis / 2) * 1e-6
    data["stress"] = 0.5 * stressfunc(data.radial_position * 1e-6 + r, -config["pressure_pa"],
                                      config["channel_length_m"], config["channel_width_m"]) \
                     + 0.5 * stressfunc(data.radial_position * 1e-6 - r, -config["pressure_pa"],
                                        config["channel_length_m"],
                                        config["channel_width_m"])
    # data["stress_center"] = stressfunc(data.radial_position * 1e-6, -config["pressure_pa"], config["channel_length_m"],
    #                                  config["channel_width_m"])

    data["strain"] = (data.long_axis - data.short_axis) / np.sqrt(data.long_axis * data.short_axis)


def filterCenterCells(data: pd.DataFrame):
    """ remove the cells in the center of the flow profile """
    r = np.sqrt(data.long_axis / 2 * data.short_axis / 2) * 1e-6
    data = data[np.abs(data.radial_position * 1e-6) > r]
    return data


def filterCells(data: pd.DataFrame, solidity_threshold=0.96, irregularity_threshold=1.06) -> pd.DataFrame:
    """ filter cells acording to solidity and irregularity """
    data = data.query(f"solidity > {solidity_threshold} and irregularity < {irregularity_threshold}")
    data.reset_index(drop=True, inplace=True)
    return data


def fit_func_velocity(config):
    if "vel_fit" in config:
        p0, p1, p2 = config["vel_fit"]
        p2 = 0
    else:
        p0, p1, p2 = None, None, None

    def velfit(r, p0=p0, p1=p1, p2=p2):  # for stress versus strain
        R = config["channel_width_m"] / 2 * 1e6
        return p0 * (1 - np.abs((r + p2) / R) ** p1)

    return velfit


def fit_func_velocity_gradient(config):
    if "vel_fit" in config:
        p0, p1, p2 = config["vel_fit"]
        p2 = 0
    else:
        p0, p1, p2 = None, None, None

    def getVelGrad(r, p0=p0, p1=p1, p2=p2):
        p0 = p0 * 1e3
        R = config["channel_width_m"] / 2 * 1e6
        return - (p0 * p1 * (np.abs(r) / R) ** p1) / r

    return getVelGrad


def correctCenter(data, config):
    if not "velocity" in data:
        getVelocity(data, config)
    d = data[np.isfinite(data.velocity)]
    y_pos = d.radial_position
    vel = d.velocity

    if len(vel) == 0:
        raise ValueError("No velocity values found.")

    vel_fit, pcov = curve_fit(fit_func_velocity(config), y_pos, vel,
                              [np.nanpercentile(vel, 95), 3, -np.mean(y_pos)])  # fit a parabolic velocity profile
    y_pos += vel_fit[2]
    # data.y += vel_fit[2]
    data.radial_position += vel_fit[2]

    config["vel_fit"] = list(vel_fit)
    config["center"] = vel_fit[2]

    # data["velocity_gradient"] = fit_func_velocity_gradient(config)(data.radial_position)
    # data["velocity_fitted"] = fit_func_velocity(config)(data.radial_position)
    # data["imaging_pos_mm"] = config["imaging_pos_mm"]


def bootstrap_error(data, func=np.median, repetitions=1000):
    data = np.asarray(data)
    if len(data) <= 1:
        return 0
    medians = []
    for i in range(repetitions):
        medians.append(func(data[np.random.random_integers(len(data) - 1, size=len(data))]))
    return np.nanstd(medians)


from deformationcytometer.includes.fit_velocity import fit_velocity_pressures


def apply_velocity_fit(data2):
    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
    p0, vel, vel_grad = fit_velocity_pressures(data2, config, x_sample=100)
    eta0, delta, tau = p0
    eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)

    data2["vel_fit_error"] = np.sqrt(np.sum(((vel - data2.velocity) / data2.velocity) ** 2))

    data2["vel"] = vel
    data2["vel_grad"] = vel_grad
    data2["eta"] = eta
    data2["eta0"] = eta0
    data2["delta"] = delta
    data2["tau"] = tau
    return data2, p0


def get_cell_properties(data):
    import scipy.special
    from deformationcytometer.includes.RoscoeCoreInclude import getAlpha1, getAlpha2, getMu1, getEta1, eq41, \
        getRoscoeStrain

    alpha1 = getAlpha1(data.long_axis / data.short_axis)
    alpha2 = getAlpha2(data.long_axis / data.short_axis)

    epsilon = getRoscoeStrain(alpha1, alpha2)

    mu1 = getMu1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.stress)
    eta1 = getEta1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.eta)

    if "tt_omega" in data:
        omega = np.abs(data.tt_omega)

        ttfreq = - eq41(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), np.abs(data.vel_grad))
        # omega = ttfreq

        # omega = data.freq * 2 * np.pi

        Gp1 = mu1
        Gp2 = eta1 * np.abs(omega)
        alpha_cell = np.arctan(Gp2 / Gp1) * 2 / np.pi
        k_cell = Gp1 / (omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell))

        mu1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell)
        eta1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.sin(
            np.pi / 2 * alpha_cell) / omega

        # data["omega"] = omega
        data["tt_mu1"] = mu1
        data["tt_eta1"] = eta1
        data["tt_Gp1"] = Gp1
        data["tt_Gp2"] = Gp2
        data["tt_k_cell"] = k_cell
        data["tt_alpha_cell"] = alpha_cell
        data["tt_epsilon"] = epsilon

    def func(x, a, b):
        return x / 2 * 1 / (1 + a * x ** b)

    x = [0.113, 0.45]

    omega_weissenberg = func(np.abs(data.vel_grad), *x)
    w_Gp1 = mu1
    w_Gp2 = eta1 * np.abs(omega_weissenberg)
    w_alpha_cell = np.arctan(w_Gp2 / w_Gp1) * 2 / np.pi
    w_k_cell = w_Gp1 / (omega_weissenberg ** w_alpha_cell * scipy.special.gamma(1 - w_alpha_cell) * np.cos(
        np.pi / 2 * w_alpha_cell))

    data["omega"] = omega_weissenberg
    data["Gp1"] = w_Gp1
    data["Gp2"] = w_Gp2
    data["k"] = w_k_cell
    data["alpha"] = w_alpha_cell

    return omega, mu1, eta1, k_cell, alpha_cell, epsilon


def match_cells_from_all_data(data, config, image_width=720):
    timestamps = {i: d.timestamp for i, d in data.groupby("frame").mean().iterrows()}
    for i, d in data.iterrows():
        x = d.x
        v = d.vel * 1e3 / config["pixel_size"]
        t = d.timestamp
        for dir in [1]:
            for j in range(1, 10):
                frame2 = d.frame + j * dir
                try:
                    dt = timestamps[frame2] - t
                except KeyError:
                    continue
                x2 = x + v * dt
                # if we ran out of the image, stop
                if not (0 <= x2 <= image_width):
                    break
                d2 = data[data.frame == frame2]
                # if the cell is already present in the frame, to not do any matching
                if len(d2[d2.cell_id == d.cell_id]):
                    continue
                d2 = d2[np.abs(d2.radial_position - d.radial_position) < 10]
                d2 = d2[np.abs(d2.x - x2) < 15]
                # if it survived the filters, merge
                if len(d2):
                    data.loc[data['cell_id'] == d2.iloc[0].cell_id, 'cell_id'] = d.cell_id


def get_mode(x):
    """ get the mode of a distribution by fitting with a KDE """
    from scipy import stats
    x = np.array(x)
    x = x[~np.isnan(x)]

    kde = stats.gaussian_kde(x)
    return x[np.argmax(kde(x))]


def get_mode_stats(x, do_plot=False):
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    from scipy import stats

    x = np.array(x)
    x = x[~np.isnan(x)]

    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]

    mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=10)
    if do_plot is True:
        def string(x):
            if x > 1:
                return str(round(x))
            else:
                return str(round(x, 2))

        plt.text(0.5, 1, string(mode) + "$\pm$" + string(err), transform=plt.gca().transAxes, ha="center", va="top")
    return mode, err, len(x)


def bootstrap_match_hist(data_list, bin_width=25, max_bin=300, property="stress", groupby=None):
    import pandas as pd
    if groupby is not None and isinstance(data_list, pd.DataFrame):
        data_list = [d for name, d in data_list.groupby(groupby)]
    # create empty lists
    data_list2 = [[] for _ in data_list]
    # iterate over the bins
    for i in range(0, max_bin, bin_width):
        # find the maximum
        counts = [len(data[(i < data[property]) & (data[property] < (i + bin_width))]) for data in data_list]
        max_count = np.max(counts)
        min_count = np.min(counts)
        # we cannot upsample from 0
        if min_count == 0:
            continue
        # iterate over datasets
        for index, data in enumerate(data_list):
            # get the subset of the data that is in this bin
            data_subset = data[(i < data[property]) & (data[property] < (i + bin_width))]
            # sample from this subset
            data_list2[index].append(data_subset.sample(max_count, replace=True))

    # concatenate the datasets
    for index, data in enumerate(data_list):
        data_list2[index] = pd.concat(data_list2[index])

    if groupby is not None:
        return pd.concat(data_list2)

    return data_list2


def get2Dhist_k_alpha(data):
    return get2Dhist_k_alpha_err(data, bootstrap_repetitions=0)


def get2Dhist_k_alpha_err(data, bootstrap_repetitions=10):
    from scipy import stats
    x = np.array(data[["k", "alpha"]]).T
    x[0] = np.log10(x[0])

    def get_mode(x):
        kde = stats.gaussian_kde(x)
        mode = x[..., np.argmax(kde(x))]
        mode[0] = 10 ** mode[0]
        return mode

    def bootstrap_error(data, func, repetitions):
        medians = []
        for i in range(repetitions):
            medians.append(func(data[..., np.random.randint(data.shape[-1] - 1, size=data.shape[-1])]))
        return np.nanstd(np.array(medians), axis=0)

    mode = get_mode(x)
    if bootstrap_repetitions == 0:
        return pd.Series(mode, index=["k", "alpha"])
    err = bootstrap_error(x, get_mode, repetitions=bootstrap_repetitions)

    return pd.Series([mode[0], err[0], mode[1], err[1]], index=["k", "k_err", "alpha", "alpha_err"])


def getGp1Gp2fit_k_alpha(data):
    from scipy.special import gamma
    data = data.query("w_Gp1 > 0 and w_Gp2 > 0")

    def fit(omega, k, alpha):
        omega = np.array(omega)
        G = k * (1j * omega) ** alpha * gamma(1 - alpha)
        return np.real(G), np.imag(G)

    def cost(p):
        Gp1, Gp2 = fit(data.omega_weissenberg, *p)
        # return np.sum(np.abs(np.log10(data.w_Gp1) - np.log10(Gp1))) + np.sum(
        #    np.abs(np.log10(data.w_Gp2) - np.log10(Gp2)))
        return np.median((np.log10(data.w_Gp1) - np.log10(Gp1)) ** 2) + np.median(
            (np.log10(data.w_Gp2) - np.log10(Gp2)) ** 2)

    from scipy.optimize import minimize
    res = minimize(cost, [np.median(data.w_k_cell), np.mean(data.w_alpha_cell)],
                   method="nelder-mead")  # , bounds=([0, np.inf], [0, 1]))
    print(res)

    return res.x[0], res.x[1]


def getGp1Gp2fit3_k_alpha(data):
    from scipy.special import gamma
    k0, alpha0 = getGp1Gp2fit_k_alpha(data)
    data = data.query("w_Gp1 > 0 and w_Gp2 > 0")

    def fit(omega, k, alpha, mu):
        omega = np.array(omega)
        G = k * (1j * omega) ** alpha * gamma(1 - alpha) + 1j * omega * mu
        return np.real(G), np.imag(G)

    def cost(p):
        Gp1, Gp2 = fit(data.omega_weissenberg, *p)
        # return np.sum(np.abs(np.log10(data.w_Gp1) - np.log10(Gp1))) + np.sum(
        #    np.abs(np.log10(data.w_Gp2) - np.log10(Gp2)))
        return np.median((np.log10(data.w_Gp1) - np.log10(Gp1)) ** 2) + np.median(
            (np.log10(data.w_Gp2) - np.log10(Gp2)) ** 2)

    from scipy.optimize import minimize
    res = minimize(cost, [k0, alpha0, 0], method="nelder-mead")  # , bounds=([0, np.inf], [0, 1]))
    print(res)

    return res.x[0], res.x[1], res.x[2]


def stress_strain_fit(data, k_cell, alpha_cell):
    from deformationcytometer.includes.RoscoeCoreInclude import getRatio
    from deformationcytometer.includes.fit_velocity import getFitXYDot
    import scipy
    eta0 = data.iloc[0].eta0
    alpha = data.iloc[0].delta
    tau = data.iloc[0].tau

    count = 10

    pressure = data.iloc[0].pressure

    def func(x, a, b):
        return x / 2 * 1 / (1 + a * x ** b)

    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXYDot(config, np.mean(pressure), p, count=count * 2)
        return x, y

    channel_pos, vel_grad = getFitLine(pressure, [eta0, alpha, tau])
    vel_grad = -vel_grad
    vel_grad = vel_grad[channel_pos > 0]
    channel_pos = channel_pos[channel_pos > 0]

    omega = func(np.abs(vel_grad), *[0.113, 0.45])

    mu1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell)
    eta1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.sin(np.pi / 2 * alpha_cell) / omega

    ratio, alpha1, alpha2, strain, stress, theta, ttfreq, eta, vdot = getRatio(eta0, alpha, tau, vel_grad, mu1_, eta1_)

    return stress, strain
