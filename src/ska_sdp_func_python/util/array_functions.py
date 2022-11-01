"""Useful array functions.

"""

__all__ = [
    "average_chunks",
    "average_chunks2",
    "tukey_filter",
    "insert_array",
    "insert_function_L",
    "insert_function_pswf",
    "insert_function_sinc",
]

import numpy


def average_chunks(arr, wts, chunksize):
    """Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize

    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data_models, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts

    # Original codes
    # places = range(0, len(arr), chunksize)
    # chunks = numpy.add.reduceat(wts * arr, places)
    # weights = numpy.add.reduceat(wts, places)
    # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    # Codes optimized

    mask = numpy.zeros(((len(arr) - 1) // chunksize + 1, arr.shape[0]), dtype=bool)
    for enumerate_id, i in enumerate(range(0, len(arr), chunksize)):
        mask[enumerate_id, i : i + chunksize] = 1
    chunks = mask.dot(wts * arr)
    weights = mask.dot(wts)
    # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    numpy.putmask(chunks, weights > 0.0, chunks / weights)

    return chunks, weights


def average_chunks2(arr, wts, chunksize):
    """Average the two dimensional array arr with weights by chunks

    Array len does not have to be multiple of chunksize.

    :param arr: 2D array of values
    :param wts: 2D array of weights
    :param chunksize: 2-tuple of averaging region e.g. (2,3)
    :return: 2D array of averaged data_models, 2d array of weights
    """
    # Do each axis to determine length
    #    assert arr.shape == wts.shape, "Shapes of arrays must be the same"
    # It is possible that there is a dangling null axis on wts
    wts = wts.reshape(arr.shape)

    l0 = len(average_chunks(arr[:, 0], wts[:, 0], chunksize[0])[0])
    l1 = len(average_chunks(arr[0, :], wts[0, :], chunksize[1])[0])

    tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
    tempwt = numpy.zeros([arr.shape[0], l1])

    tempchunks *= tempwt
    for i in range(arr.shape[0]):
        result = average_chunks(arr[i, :], wts[i, :], chunksize[1])
        tempchunks[i, :], tempwt[i, :] = result[0].flatten(), result[1].flatten()

    chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
    weights = numpy.zeros([l0, l1])

    for i in range(l1):
        result = average_chunks(tempchunks[:, i], tempwt[:, i], chunksize[0])
        chunks[:, i], weights[:, i] = result[0].flatten(), result[1].flatten()

    return chunks, weights


def tukey_filter(x, r):
    """Calculate the Tukey (tapered cosine) filter

    See e.g. https://uk.mathworks.com/help/signal/ref/tukeywin.html

    :param x: x coordinate (float)
    :param r: transition point of filter (float)
    :returns: Value of filter for x
    """
    if 0.0 <= x < r / 2.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - r / 2.0) / r))
    elif 1 - r / 2.0 <= x <= 1.0:
        return 0.5 * (1.0 + numpy.cos(2.0 * numpy.pi * (x - 1 + r / 2.0) / r))
    else:
        return 1.0


def insert_function_sinc(x):
    """Insertion with Sinc function

    :param x: 1D vector
    :return: 1d vector
    """
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi * x[x != 0.0]) / (numpy.pi * x[x != 0.0])
    return s


def insert_function_L(x, a=5):
    """Insertion with Lanczos function

    :param x: 1D vector
    :param a: width
    :return: 1d vector
    """
    L = insert_function_sinc(x) * insert_function_sinc(x / a)
    return L


def insert_function_pswf(x, a=5):
    """Insertion with PSWF

    :param x: 1D vector
    :param a: width
    :return: 1d vector
    """

    from rascil.processing_components.fourier_transforms.fft_coordinates import grdsf

    return grdsf(abs(x) / a)[1]


def insert_array(
    im, x, y, flux, bandwidth=1.0, support=7, insert_function=insert_function_L
):
    """Insert point into image using specified function

    :param im: Image
    :param x: x in float pixels
    :param y: y in float pixels
    :param flux: Flux[nchan, npol]
    :param bandwidth: Support of data in uv plane
    :param support: Support of function in image space
    :param insert_function: insert_function_L or insert_function_Sinc or insert_function_pswf
    :return:
    """
    nchan, npol, ny, nx = im.shape
    intx = int(numpy.round(x))
    inty = int(numpy.round(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-support, support)
    gridy = numpy.arange(-support, support)

    insert = numpy.outer(
        insert_function(bandwidth * (gridy - fracy)),
        insert_function(bandwidth * (gridx - fracx)),
    )

    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum

    for chan in range(nchan):
        for pol in range(npol):
            im[
                chan,
                pol,
                inty - support : inty + support,
                intx - support : intx + support,
            ] += (
                flux[chan, pol] * insert
            )

    return im
