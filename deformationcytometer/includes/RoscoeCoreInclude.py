import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar as fzero
import scipy.integrate
import scipy.optimize
from deformationcytometer.includes.form_factors import getFormFactorFunctions

def integral(*args):
    return quad(*args)[0]

getAlpha1, getAlpha2, getK, getI = getFormFactorFunctions()

def eq41(alpha1, alpha2, theta, kappa):
    t1 = -(alpha1 ** 2 + alpha2 ** 2) / (2 * alpha1 * alpha2)
    t2 = 1 - (alpha1 ** 2 - alpha2 ** 2) / (alpha1 ** 2 + alpha2 ** 2) * np.cos(2 * theta)

    nu = t1 * t2 * kappa / 2
    return nu


def eq79(alpha1, alpha2, theta, sigma):
    # eq. (39)
    I = getI(alpha1/alpha2)

    # eq. (79) with minus (!)
    kappa = (alpha1 ** 2 - alpha2 ** 2) / (2 * I) / (sigma * np.sin(2 * theta))

    return kappa


def eq80(alpha1, alpha2, sigma, tau):
    K = getK(alpha1/alpha2)

    t1 = (alpha1 ** 2 - alpha2 ** 2) / (alpha1 ** 2 + alpha2 ** 2)
    t2 = 1 + (tau - sigma) / (K * sigma) * ((alpha1 ** 2 + alpha2 ** 2) / (2 * alpha1 * alpha2)) ** 2
    n1 = 1 + (tau - sigma) / (K * sigma) * ((alpha1 ** 2 - alpha2 ** 2) / (2 * alpha1 * alpha2)) ** 2

    theta = np.arccos(t1 * t2 / n1) / 2

    return theta

def getEta1(alpha1, alpha2, theta, eta0):
    K = getK(alpha1/alpha2)
    A = (alpha1**2 + alpha2**2) / (alpha1**2 - alpha2**2)
    B = (alpha1**2 + alpha2**2) / (2 * alpha1 * alpha2)
    C = (alpha1**2 - alpha2**2) / (2 * alpha1 * alpha2)
    eta1 = eta0 * (5/2 * K * (1-np.cos(2*theta)*A) / (C**2 * np.cos(2*theta)*A - B**2) + 1)
    #print(np.cos(2*theta))
    #print(1/A * (1 + 2/(5*K)*(eta1-eta0)/eta0 * B**2)/ (1 + 2/(5*K)*(eta1-eta0)/eta0 * C**2))
    #np.testing.assert_almost_equal(np.array(np.cos(2*theta)), np.array(1/A * (1 + 2/(5*K)*(eta1-eta0)/eta0 * B**2)/ (1 + 2/(5*K)*(eta1-eta0)/eta0 * C**2)))
    return eta1

def getMu1(alpha1, alpha2, theta, stress):
    I = getI(alpha1/alpha2)
    mu1 = 5/2 * stress * (2*I) / (alpha1**2 - alpha2**2) * np.sin(2*theta)
    return mu1

def getRoscoeStrain(alpha1, alpha2):
    I = getI(alpha1 / alpha2)
    epsilon = (alpha1 ** 2 - alpha2 ** 2) / (2*I)
    return epsilon

def getShearRate(viscLiquid, NHmodulus, viscSolid, alpha1_alpha2):
    # physical parameters in Roscoe notation
    eta0 = viscLiquid
    mu1 = NHmodulus
    eta1 = viscSolid

    # transform parameters, eq. (62)
    sigma = 5 * eta0 / (2 * mu1)
    tau = (3 * eta0 + 2 * eta1) / (2 * mu1)

    # compute alpha2 from (78)
    alpha1 = getAlpha1(alpha1_alpha2)
    alpha2 = getAlpha2(alpha1_alpha2)

    # compute theta from (80)
    theta = eq80(alpha1, alpha2, sigma, tau)

    # compute shear rate from (79)
    kappa = eq79(alpha1, alpha2, theta, sigma)

    # transcribe
    shearRate = kappa

    return shearRate

def getThetaTTFreq(viscLiquid, NHmodulus, viscSolid, kappa, alpha1, alpha2):
    # physical parameters in Roscoe notation
    eta0 = viscLiquid
    mu1 = NHmodulus
    eta1 = viscSolid

    # transform parameters, eq. (62)
    sigma = 5 * eta0 / (2 * mu1)
    tau = (3 * eta0 + 2 * eta1) / (2 * mu1)

    # compute theta from (80)
    theta = eq80(alpha1, alpha2, sigma, tau)

    # compute TT frequency from (41)
    nu = - eq41(alpha1, alpha2, theta, kappa)

    # transcribe
    ttFreq = nu
    thetaDeg = theta / (2 * np.pi) * 360

    return thetaDeg, ttFreq

def RoscoeCore(viscLiquid, NHmodulus, viscSolid, alpha1, alpha2):
    # physical parameters in Roscoe notation
    eta0 = viscLiquid
    mu1 = NHmodulus
    eta1 = viscSolid

    # transform parameters, eq. (62)
    sigma = 5 * eta0 / (2 * mu1)
    tau = (3 * eta0 + 2 * eta1) / (2 * mu1)

    # compute theta from (80)
    theta = eq80(alpha1, alpha2, sigma, tau)

    # compute shear rate from (79)
    kappa = eq79(alpha1, alpha2, theta, sigma)

    # compute TT frequency from (41)
    nu = - eq41(alpha1, alpha2, theta, kappa)

    # transcribe
    shearRate = kappa
    ttFreq = nu
    thetaDeg = theta / (2 * np.pi) * 360

    return shearRate, alpha2, thetaDeg, ttFreq


def RoscoeShearSingle(viscLiquid, NHmodulus, viscSolid, shearRateWanted, alpha1min=1.001, alpha1step=0.01):
    # define function to find root
    # find alpha1=x which gives the desired shear rate
    fun = lambda x: shearRateWanted - RoscoeCore(viscLiquid, NHmodulus, viscSolid, x)[0]

    # determine suitable interval for alpha1
    shearRateMin = RoscoeCore(viscLiquid, NHmodulus, viscSolid, alpha1min)[0]
    if shearRateMin.imag != 0:
        raise ValueError('Minimum shear rate is imaginary!')

    diffMin = shearRateWanted - shearRateMin

    # take small steps and detect sign change
    alpha1max = alpha1min
    cont = 1
    while cont:
        alpha1max = alpha1max + alpha1step
        shearRateMax = RoscoeCore(viscLiquid, NHmodulus, viscSolid, alpha1max)[0]
        print(cont, shearRateMax)

        if shearRateMax.imag != 0:
            # imaginary shear rate means that step was too large
            # reduce and retry
            alpha1max = alpha1max - alpha1step
            alpha1step = alpha1step / 10
            alpha1max = alpha1max + alpha1step
            shearRateMax = RoscoeCore(viscLiquid, NHmodulus, viscSolid, alpha1max)[0]

        diffMax = shearRateWanted - shearRateMax
        if diffMax * diffMin < 0:
            cont = 0

    # now we have an interval, find root
    alpha1 = fzero(fun, bracket=[alpha1min, alpha1max]).root

    # compute actual values for return
    shearRateObtained, alpha2, thetaDeg, ttFreq = RoscoeCore(viscLiquid, NHmodulus, viscSolid, alpha1)

    return alpha1, alpha2, thetaDeg, ttFreq

def getRatio(eta0, alpha, tau, vdot, NHmodulus, viscSolid):
    eta = eta0 / (1 + tau ** alpha * vdot ** alpha)
    viscLiquid = eta
    ratio = np.zeros_like(eta)
    for i in range(len(ratio)):
        test_rations = np.geomspace(1, 10, 1000)
        try:
            j = np.nanargmax(getShearRate(viscLiquid[i], NHmodulus[i], viscSolid[i], test_rations))
        except ValueError:
            ratio[i] = np.nan
            continue
        max_ratio = test_rations[j]
        if j == len(test_rations)-1 and getShearRate(viscLiquid[i], NHmodulus[i], viscSolid[i], max_ratio) - vdot[i] < 0:
            break
        while getShearRate(viscLiquid[i], NHmodulus[i], viscSolid[i], max_ratio) - vdot[i] < 0:
            test_rations = np.geomspace(test_rations[j], test_rations[j+1], 1000)
            j = np.nanargmax(getShearRate(viscLiquid[i], NHmodulus[i], viscSolid[i], test_rations))
            max_ratio = test_rations[j]

        ratio[i] = scipy.optimize.root_scalar(lambda ratio: getShearRate(viscLiquid[i], NHmodulus[i], viscSolid[i], ratio) - vdot[i], bracket=[1, max_ratio]).root

    vdot = vdot[ratio > 0]
    eta = eta[ratio > 0]
    viscLiquid = viscLiquid[ratio > 0]
    ratio = ratio[ratio > 0]
    alpha1 = getAlpha1(ratio)
    alpha2 = getAlpha2(ratio)
    theta, ttfreq = getThetaTTFreq(viscLiquid, NHmodulus, viscSolid, vdot, alpha1, alpha2)
    strain = (alpha1 - alpha2) / np.sqrt(alpha1 * alpha2)
    stress = eta*vdot
    return ratio, alpha1, alpha2, strain, stress, theta, ttfreq, eta, vdot
