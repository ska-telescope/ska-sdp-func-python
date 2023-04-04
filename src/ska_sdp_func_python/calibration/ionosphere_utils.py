# pylint: disable=invalid-name
"""
Utilities to support ionospheric calibration and the generation of ionospheric
phase screens.
"""

import logging
import math

import numpy

log = logging.getLogger("func-python-logger")


def decompose_phasescreen(x, y, r0, beta=5.0 / 3.0):
    """
    Generation and eigen-decomposition of pierce-point covariance matrix with
    Kolmogorov statistics. Used for generation of turbulent phase shifts and
    phase screens. Based on the the algorithm from "Fast simulation of a
    kolmogorov phase screen," Harding, Johnston & Lane (1999) Applied Optics,
    38 (11).

    :param x: coordinates of 1st pierce-point dimension
    :param y: coordinates of 2nd pierce-point dimension
    :param r0: diffractive scale [m]
    :param beta: exponent of the power spectrum (defaults to 5/3)
    :return: eigenvector matrix
    :return: sqrt(eigenvalues) vector

    """

    log.debug("generating phase screen eigenvectors and eigenvalues")

    # stack 2D array into a vector
    N = numpy.size(x)
    x = numpy.reshape(x, N)
    y = numpy.reshape(y, N)

    # calculate pierce-point offsets
    u = numpy.tile(x[:, numpy.newaxis], (1, N)) - numpy.tile(
        x[numpy.newaxis, :], (N, 1)
    )
    v = numpy.tile(y[:, numpy.newaxis], (1, N)) - numpy.tile(
        y[numpy.newaxis, :], (N, 1)
    )
    D = (numpy.sqrt(u * u + v * v) / r0) ** beta
    Dint = numpy.mean(D, axis=0)
    Dsum = numpy.tile(Dint[:, numpy.newaxis], (1, N)) + numpy.tile(
        Dint[numpy.newaxis, :], (N, 1)
    )
    # and the final covariance matrix
    C = -0.5 * D + 0.5 * Dsum

    # eigen decomposition
    S, U = numpy.linalg.eigh(C, UPLO="U")
    S[0] = 0.0

    return U, numpy.sqrt(S)


def zern(m, n, rho, phi):
    """
    Generate Zernike polynomial values

    :param m: standard Zernike m integer (azimuthal degree)
    :param n: standard Zernike n integer (radial degree)
    :param rho: array of polar radial coordinates
    :param phi: array of polar angular coordinates
    :return array of Zernike polynomial values:

    """
    if m >= 0:
        func = numpy.cos
    else:
        func = numpy.sin
    m = numpy.abs(m)
    R = numpy.zeros(rho.shape)
    for k in range((n - m) // 2 + 1):
        R += (
            (-1) ** k
            * math.factorial(n - k)
            // (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
            * rho ** (n - 2 * k)
        )

    return R * func(m * phi)


def zern_array(nm, x, y, noll_order=False):
    """
    Generate an array of all zernike upto a given degree

    :param nm: maximum degree. If noll_order is selected, this sets the maximum
        radial degree (n), otherwise it set the maximum values of n+|m|
    :param x: array of cartesian x coordinates
    :param y: array of cartesian y coordinates
    :param noll_order: whether to limit the Zernike polynomial order by
        n (True) or n+|m| (False)
    :return array of Zernike polynomial values:

    """
    # get normalised polar coords for all of the stations in cluster 0
    x -= numpy.mean(x)
    y -= numpy.mean(y)
    phi = numpy.arctan2(y, x)
    rho = numpy.sqrt(x * x + y * y)
    rho /= numpy.amax(rho)

    # coeff = [None] * len(x)
    # for stn in range(len(x)):
    #     coeff[stn] = []
    #     count = 0
    #     if noll_order is True:
    #         for n in range(nm + 1):
    #             for m in range(-n, n + 1, 2):
    #                 coeff[stn].append(zern(m, n, rho[stn], phi[stn]))
    #                 count += 1
    #     else:
    #         for n in range(nm + 1):
    #             for m in range(-n, n + 1, 2):
    #                 if n + numpy.abs(m) > nm:
    #                     continue
    #                 coeff[stn].append(zern(m, n, rho[stn], phi[stn]))
    #                 count += 1
    # return numpy.array(coeff)

    coeff = numpy.array([])
    count = 0
    if noll_order is True:
        for n in range(nm + 1):
            for m in range(-n, n + 1, 2):
                coeff = numpy.append(coeff, zern(m, n, rho, phi))
                count += 1

    else:
        for n in range(nm + 1):
            for m in range(-n, n + 1, 2):
                if n + numpy.abs(m) > nm:
                    continue
                coeff = numpy.append(coeff, zern(m, n, rho, phi))
                count += 1

    return numpy.reshape(coeff, (count, len(x))).T
