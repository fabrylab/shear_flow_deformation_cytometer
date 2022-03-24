from scipy.interpolate import interp1d
from scipy.optimize import minimize

from scipy import optimize
from scipy import integrate
import scipy.integrate
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from shear_flow_deformation_cytometer.includes.quad import quad, newton


def getVelocity(eta0, alpha, tau, H, W, P, L, x_sample=100):
    #print([eta0, alpha, tau, H, W, P, L])
    pi = np.pi
    n = np.arange(1, 99, 2)[None, :]

    def getBeta(y):
        return 4 * H ** 2 * P / (pi ** 3 * L) * np.sum(
            (-1) ** ((n - 1) / 2) * pi / (n ** 2 * H) * np.sinh((n * pi * y) / H) / np.cosh((n * pi * W) / (2 * H)), axis=1)

    tau_alpha = tau ** alpha

    def f(beta):
        def f2(vdot):
            return 1 - eta0 / beta * vdot + tau_alpha * vdot ** alpha
        return f2


    def f_fprime(beta):
        beta += 1e-10
        def f2(vdot):
            #if vdot == 0:
            #    return 0, 0
            return 1 - eta0 / beta * vdot + tau_alpha * vdot ** alpha, - eta0 / beta + alpha * tau_alpha * vdot ** (alpha-1)
        return f2

    def f_max(beta):
        return (eta0 / (tau_alpha * alpha * beta + 1e-4))**(1/(alpha-1))

    ys = np.arange(1e-6, H/2+1e-6, 1e-6)
    ys = np.linspace(0, H/2, x_sample)
    if 0:
        vdot = np.zeros_like(ys)
        for i, y in enumerate(ys):
            if y == 0:
                vdot[i] = 0
                continue
            beta = getBeta(y)
            sol = optimize.root_scalar(f(beta), bracket=[0, 1000_000], method='brentq')
            vdot[i] = sol.root
        v = np.cumsum(vdot)*np.diff(ys)[0]
        v -= v[-1]

    if 0:
        def getVdot(y):
            if y == 0:
                return 0
            beta = getBeta(y)
            sol = optimize.root_scalar(f(beta), bracket=[0, 1e7], method='brentq')
            return sol.root

        getVdot = np.vectorize(getVdot)

    def getVdot(y):
        beta = getBeta(y[:, None])
        start = f_max(beta)
        yy = newton(f_fprime(beta), start+0.00001)
        yy[y==0] = 0
        return yy

    if 0:
        x = np.arange(-10, 300, 0.01)
        beta = getBeta(ys[1:][:, None])
        xx = f(beta[80])(x)
        plt.plot(x, xx)
        plt.show()

    v = quad(getVdot, H / 2, ys)
    vdot = getVdot(ys)

    return ys, v, vdot

def getFitFunc(x, eta0, alpha, tau, H, W, P, L, x_sample=100):
    ys, v, vdot = getVelocity(eta0, alpha, tau, H, W, P, L, x_sample)
    return -interpolate.interp1d(ys, v, fill_value="extrapolate")(np.abs(x))

def getFitFuncDot(x, eta0, alpha, tau, H, W, P, L):
    ys, v, vdot = getVelocity(eta0, alpha, tau, H, W, P, L)
    return -interpolate.interp1d(ys, vdot, fill_value="extrapolate")(np.abs(x))

def fit_velocity(data, config, p=None, channel_width=None):
    if channel_width is None:
        H = config["channel_width_m"]
        W = config["channel_width_m"]
    else:
        H = channel_width
        W = channel_width
    P = config["pressure_pa"]
    L = config["channel_length_m"]

    x, y = data.radial_position * 1e-6, data.measured_velocity * 1e-3
    i = np.isfinite(x) & np.isfinite(y)
    x2 = x[i]
    y2 = y[i]

    def getAllCost(p):
        cost = 0
        cost += np.sum((getFitFunc(x2, p[0], p[1], p[2], H, W, P, L) - y2) ** 2)
        return cost

    if p is None:
        res = optimize.minimize(getAllCost, [3.8, 0.64, 0.04], method="TNC", options={'maxiter': 200, 'ftol': 1e-16},
                                bounds=[(0, np.inf), (0, 0.9), (0, np.inf)])

        p = res["x"]
    eta0, alpha, tau = p
    return p, getFitFunc(x, eta0, alpha, tau, H, W, P, L), getFitFuncDot(x, eta0, alpha, tau, H, W, P, L)


def fit_velocity_pressures(data, config, p=None, channel_width=None, pressures=None, x_sample=100, start_params=[3.8, 0.64, 0.04], method="TNC"):
    if channel_width is None:
        H = config["channel_width_m"]
        W = config["channel_width_m"]
    else:
        H = channel_width
        W = channel_width
    L = config["channel_length_m"]

    x, y = data.radial_position * 1e-6, data.measured_velocity * 1e-3
    i = np.isfinite(x) & np.isfinite(y)
    x2 = x[i]
    y2 = y[i]

    all_pressures = np.unique(data.pressure)
    if pressures is None:
        fit_pressures = all_pressures
    else:
        fit_pressures = pressures
    press = np.array(data.pressure)
    press2 = press[i]

    def getAllCost(p):
        cost = 0
        for P in fit_pressures:
            difference = getFitFunc(x2[press2 == P], p[0], p[1], p[2], H, W, P*1e5, L, x_sample) - y2[press2 == P]
            c = np.abs(difference)
            #c = difference**2
            if np.sum(np.isnan(c)):
                return np.Inf
            # clip cost for possible outliers
            c = np.clip(c, 0, np.percentile(c, 95))
            cost += np.sum(c)
        return cost

    if p is None:
        res = optimize.minimize(getAllCost, start_params, method=method, options={'maxiter': 200, 'ftol': 1e-16},
                                bounds=[(0, np.inf), (0, 0.9), (0, np.inf)])

        p = res["x"]

    eta0, alpha, tau = p
    y = np.zeros_like(x)
    yy = np.zeros_like(x)
    for P in all_pressures:
        indices = press == P
        y[indices] = getFitFunc(x[indices], eta0, alpha, tau, H, W, P*1e5, L)
        yy[indices] = getFitFuncDot(x[indices], eta0, alpha, tau, H, W, P*1e5, L)
    return p, y, yy


def getFitXY(config, pressure, p):
    H = config["channel_width_m"]
    W = config["channel_width_m"]
    L = config["channel_length_m"]
    x = np.linspace(-W/2, W/2, 1000)
    eta0, alpha, tau = p
    y = getFitFunc(x, eta0, alpha, tau, H, W, pressure * 1e5, L)
    return x, y


def getFitXYDot(config, pressure, p, count=1000):
    H = config["channel_width_m"]
    W = config["channel_width_m"]
    L = config["channel_length_m"]
    x = np.linspace(-W/2, W/2, count)
    eta0, alpha, tau = p
    y = getFitFuncDot(x, eta0, alpha, tau, H, W, pressure * 1e5, L)
    return x, y




def improved_fit(data, config, plot=False):
    curve_height = np.mean(data.query("-5 < radial_position < 5").measured_velocity) * 1e-3

    x0 = np.array(data.radial_position)
    y0 = np.array(data.measured_velocity * 1e-3)
    i = np.isfinite(x0) * np.isfinite(y0)
    x0 = x0[i]
    y0 = y0[i]

    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}

    def getFitLine(pressure, p):
        #config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXY(config, np.mean(pressure), p)
        return x * 1e+6, y

    def getEta0(delta, tau):
        def cost(eta0):
            x, y = getFitLine(data.iloc[0].pressure, [eta0, delta, tau])
            return np.abs(curve_height - np.max(y))

        res = minimize(cost, [1], method="Nelder-Mead", bounds=[(0, np.inf)])
        eta0 = res["x"][0]
        return eta0

    def getEta0Error(delta, tau):
        def cost(eta0):
            x, y = getFitLine(data.iloc[0].pressure, [eta0, delta, tau])
            return np.abs(curve_height - np.max(y))

        res = minimize(cost, [1], method="Nelder-Mead", bounds=[(0, np.inf)])
        eta0 = res["x"][0]
        x, y = getFitLine(data.iloc[0].pressure, [eta0, delta, tau])
        f = interp1d(x, y)
        y2 = f(x0)
        y_true = y0
        difference = y2 - y_true

        c = np.abs(difference)
        c = np.clip(c, 0, np.percentile(c, 95))
        error = np.sum(c)
        return error, (eta0, delta, tau)

    import time
    t = time.time()
    errors = [getEta0Error(i, j) for i in np.arange(0.01, 0.8, 0.1) for j in [1, 0.3, 0.1, 0.03, 0.01]]
    eta0, delta, tau = errors[np.argmin([e[0] for e in errors])][1]
    #print(time.time()-t)
    #data, p0 = apply_velocity_fit(data, start_params=[eta0, delta, tau], method="Nelder-Mead")

    def getCostAll(p):
        eta0, delta, tau = p

        x, y = getFitLine(data.iloc[0].pressure, [eta0, delta, tau])
        f = interp1d(x, y)
        y2 = f(data.radial_position)
        y_true = data.measured_velocity * 1e-3
        difference = y2 - y_true

        #error = np.sqrt(np.sum(difference**2))
        error = np.sum(np.abs(difference)**2)
        return error

    res = minimize(getCostAll, [eta0, delta, tau], method="Nelder-Mead", bounds=[(0, np.inf), (0, 0.9), (0, np.inf)])
    eta0, delta, tau = res["x"]

    H = config["channel_width_m"]
    W = config["channel_width_m"]
    L = config["channel_length_m"]
    P = data.iloc[0].pressure
    vel = getFitFunc(data.radial_position * 1e-6, eta0, delta, tau, H, W, P*1e5, L)
    vel_grad = getFitFuncDot(data.radial_position * 1e-6, eta0, delta, tau, H, W, P*1e5, L)

    eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)

    data["vel_fit_error"] = np.sqrt(np.sum(((vel - data.measured_velocity) / data.measured_velocity) ** 2))

    data["vel"] = vel
    data["vel_grad"] = vel_grad
    data["eta"] = eta
    data["eta0"] = eta0
    data["delta"] = delta
    data["tau"] = tau

    if plot:
        def getFitLine(pressure, p):
        #    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
            x, y = getFitXY(config, np.mean(pressure), p)
            return x * 1e+6, y
        plt.plot(x0, y0, "o")
        plt.plot(*getFitLine(data.iloc[0].pressure, [eta0, delta, tau]))
        #plt.plot(*getFitLine(data.iloc[0].pressure, p0))
        plt.plot(data.radial_position, vel, "+")
        plt.show()
