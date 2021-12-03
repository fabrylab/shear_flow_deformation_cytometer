import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar as fzero
import scipy.integrate
import scipy.optimize
from pathlib import Path

def integral(*args):
    return quad(*args)[0]


def eq78(alpha1, alpha2):
    # eq. (28)
    alpha3 = 1 / (alpha1 * alpha2)

    # eq. (18)
    deltaP = lambda lambd: np.sqrt((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * (alpha3 ** 2 + lambd))
    fung1 = lambda lambd: lambd / ((alpha2 ** 2 + lambd) * (alpha3 ** 2 + lambd) * deltaP(lambd))
    g1pp = integral(fung1, 0, np.Inf)
    fung2 = lambda lambd: lambd / ((alpha1 ** 2 + lambd) * (alpha3 ** 2 + lambd) * deltaP(lambd))
    g2pp = integral(fung2, 0, np.Inf)
    fung3 = lambda lambd: lambd / ((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * deltaP(lambd))
    g3pp = integral(fung3, 0, np.Inf)

    # eq. (39)
    I = 2 / 5 * (g1pp + g2pp) / (g2pp * g3pp + g3pp * g1pp + g1pp * g2pp)

    # eq. (40)
    J = 2 / 5 * (g1pp - g2pp) / (g2pp * g3pp + g3pp * g1pp + g1pp * g2pp)

    # eq. (78)
    diff = (alpha1 ** 2 + alpha2 ** 2 - 2 / (alpha1 ** 2 * alpha2 ** 2)) / (alpha1 ** 2 - alpha2 ** 2) - J / I

    return diff

def getAlpha2(alpha1):
    if alpha1 == 1:
        return 1
    fun = lambda x: eq78(alpha1, x)
    alpha2 = fzero(fun, bracket=[1e-5, 1 - 1e-9]).root
    return alpha2


def getI_raw(alpha1, alpha2):
    # eq. (28)
    alpha3 = 1 / (alpha1 * alpha2)

    # eq. (18)
    deltaP = lambda lambd: np.sqrt((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * (alpha3 ** 2 + lambd))
    fung1 = lambda lambd: lambd / ((alpha2 ** 2 + lambd) * (alpha3 ** 2 + lambd) * deltaP(lambd))
    g1pp = integral(fung1, 0, np.Inf)

    fung2 = lambda lambd: lambd / ((alpha1 ** 2 + lambd) * (alpha3 ** 2 + lambd) * deltaP(lambd))
    g2pp = integral(fung2, 0, np.Inf)
    fung3 = lambda lambd: lambd / ((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * deltaP(lambd))
    g3pp = integral(fung3, 0, np.Inf)

    # eq. (39)
    I = 2 / 5 * (g1pp + g2pp) / (g2pp * g3pp + g3pp * g1pp + g1pp * g2pp)

    return I

def getK_raw(alpha1, alpha2):
    # eq. (28)
    alpha3 = 1 / (alpha1 * alpha2)

    # eq. (20)
    deltaP = lambda lambd: np.sqrt((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * (alpha3 ** 2 + lambd))
    fung3p = lambda lambd: 1 / ((alpha1 ** 2 + lambd) * (alpha2 ** 2 + lambd) * deltaP(lambd))
    g3p = integral(fung3p, 0, np.Inf)

    # eq. (43)
    K = 1 / (5 * g3p) * (alpha1 ** 2 + alpha2 ** 2) / (alpha1 ** 2 * alpha2 ** 2)

    return K

from scipy import interpolate

def getFormFactorFunctions():
    output = Path(__file__).parent / "form_factors.npy"
    if not output.exists():
        alpha1 = np.arange(1., 10, 0.01)
        alpha2 = np.array([getAlpha2(a1) for a1 in alpha1])
        I = np.array([getI_raw(a1, a2) for a1, a2 in zip(alpha1, alpha2)])
        K = np.array([getK_raw(a1, a2) for a1, a2 in zip(alpha1, alpha2)])
        alpha1_alpha2 = alpha1/alpha2

        np.save(output, np.array([alpha1_alpha2, alpha1, alpha2, K, I]))

    alpha1_alpha2, alpha1, alpha2, K, I = np.load(output)

    _getAlpha1 = interpolate.interp1d(alpha1_alpha2, alpha1)
    _getAlpha2 = interpolate.interp1d(alpha1_alpha2, alpha2)
    _getK = interpolate.interp1d(alpha1_alpha2, K)
    _getI = interpolate.interp1d(alpha1_alpha2, I)
    return _getAlpha1, _getAlpha2, _getK, _getI