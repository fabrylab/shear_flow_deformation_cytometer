import numpy as np


def getQuadrature(N: int, xmin: float, xmax: float) -> (np.ndarray, np.ndarray):
    """
    Provides N quadrature points for an integration from xmin to xmax together with their weights.

    Parameters
    ----------
    N : int
        The number of quadrature points to use. Has to be 1 <= N <= 5.
    xmin : float
        The start of the integration range
    xmax : float
        The end of the integration range

    Returns
    -------
    points : np.ndarray
        The points of the quadrature
    w : np.ndarray
        The weights of the points
    """
    if N < 1:
        raise ValueError()

    if N == 1:
        points = [0]
        w = [2]

    if N == 2:
        points = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
        w = [1, 1]

    if N == 3:
        points = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        w = [5 / 9, 8 / 9, 5 / 9]

    if N == 4:
        points = [-np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), +np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                  -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), +np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))]
        w = [(18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36]

    if N == 5:
        points = [0,
                  -1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), +1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                  -1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), +1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))]
        w = [128 / 225, (322 + 13 * np.sqrt(70)) / 900, (322 + 13 * np.sqrt(70)) / 900, (322 - 13 * np.sqrt(70)) / 900,
             (322 - 13 * np.sqrt(70)) / 900]

    if N > 5:
        raise ValueError()

    points = np.array(points)
    w = np.array(w)
    factor = (xmax - xmin) / 2
    points = factor * points[:, None] + (xmax + xmin) / 2
    w = w[:, None] * factor
    return points, w

def flatten_input(f):
    return lambda x: f(x.ravel()).reshape(x.shape)

def quad(f, a, b, deg=5):
    points, w = getQuadrature(deg, a, b)
    integral = np.sum(flatten_input(f)(points) * w, axis=0)
    return integral


def newton(func, x0, args=[], maxiter=100, mtol=1e-5):
    for i in range(maxiter):
        y, ydot = func(x0, *args)
        dx = - y/ydot
        if np.all(np.abs(y) < mtol):
            break
        x0 += dx
    return x0

