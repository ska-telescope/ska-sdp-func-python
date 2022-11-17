"""
Functions to aid operations on imaging results.
"""

__all__ = [
    "remove_sumwt",
    "sum_invert_results",
    "sum_predict_results",
    "threshold_list",
]

import logging

import numpy
from ska_sdp_datamodels.image import Image

from ska_sdp_func_python.image.taylor_terms import (
    calculate_image_frequency_moments,
)
from ska_sdp_func_python.imaging.base import normalise_sumwt

log = logging.getLogger("func-python-logger")


def sum_invert_results(image_list):
    """Sum a set of invert results with appropriate weighting.

    :param image_list: List of [image, sum weights] pairs
    :return: Image, sum of weights
    """
    if len(image_list) == 1:
        im = image_list[0][0].copy(deep=True)
        sumwt = image_list[0][1]
        return im, sumwt

    im = Image.constructor(
        data=numpy.zeros_like(image_list[0][0]["pixels"].data),
        polarisation_frame=image_list[0][0].image_acc.polarisation_frame,
        wcs=image_list[0][0].image_acc.wcs,
    )

    sumwt = image_list[0][1]
    sumwt *= 0.0

    for _, arg in enumerate(image_list):
        if arg is not None:
            im["pixels"].data += (
                arg[1][..., numpy.newaxis, numpy.newaxis]
                * arg[0]["pixels"].data
            )
            sumwt += arg[1]

    im = normalise_sumwt(im, sumwt)

    return im, sumwt


def remove_sumwt(results):
    """Remove sumwt term in list of (image, sumwt) tuples.

    :param results: List of tuples (image, sumwt)
    :return: A list of just the dirty images
    """
    try:
        return [d[0] for d in results]
    except KeyError:
        return results


def sum_predict_results(results):
    """Sum a set of predict results of the same shape.

    :param results: List of visibilities to be summed
    :return: summed Visibility
    """
    # pylint: disable=unsubscriptable-object
    sum_results = None
    for result in results:
        if result is not None:
            if sum_results is None:
                sum_results = result
            else:
                assert (
                    sum_results["vis"].data.shape == result["vis"].data.shape
                )
                sum_results["vis"].data += result["vis"].data

    return sum_results


def threshold_list(
    imagelist, threshold, fractional_threshold, use_moment0=True, prefix=""
):
    """Find actual threshold for list of results, optionally using moment 0.

    :param prefix: Prefix in log messages
    :param imagelist: List of images
    :param threshold: Absolute threshold
    :param fractional_threshold: Fractional  threshold
    :param use_moment0: Use moment 0 for threshold
    :param prefix: Prefix for logging information
    :return: Actual threshold
    """
    peak = 0.0
    for i, result in enumerate(imagelist):
        if use_moment0:
            moments = calculate_image_frequency_moments(result)
            this_peak = numpy.max(
                numpy.abs(
                    moments["pixels"].data[0, ...] / result["pixels"].shape[0]
                )
            )
            peak = max(peak, this_peak)
            log.info(
                "threshold_list: using moment 0, sub_image %s, peak = %s",
                i,
                this_peak,
            )
        else:
            ref_chan = result["pixels"].data.shape[0] // 2
            this_peak = numpy.max(numpy.abs(result["pixels"].data[ref_chan]))
            peak = max(peak, this_peak)
            log.info(
                "threshold_list: using refchan %s, sub_image %s, peak = %s",
                ref_chan,
                i,
                this_peak,
            )

    actual = max(peak * fractional_threshold, threshold)

    if use_moment0:
        log.info(
            "threshold_list %s: Global peak in moment 0 = %s, "
            "sub-image clean threshold will be {actual}",
            prefix,
            peak,
        )
    else:
        log.info(
            "threshold_list {prefix}: Global peak = {peak}, "
            "sub-image clean threshold will be {actual}"
        )

    return actual
