""" SkyComponent functions using taylor terms in frequency

"""

__all__ = [
    "calculate_skycomponent_list_taylor_terms",
    "find_skycomponents_frequency_taylor_terms",
    "interpolate_skycomponents_frequency",
    "transpose_skycomponents_to_channels",
    "gather_skycomponents_from_channels",
]

import logging
from typing import List

import numpy
from numpy.polynomial import polynomial as polynomial
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

from src.ska_sdp_func_python import (
    copy_skycomponent,
    find_skycomponents,
    fit_skycomponent,
)
from src.ska_sdp_func_python.image.taylor_terms import (
    calculate_frequency_taylor_terms_from_image_list,
)
from src.ska_sdp_func_python.parameters import get_parameter

log = logging.getLogger("rascil-logger")


def calculate_skycomponent_list_taylor_terms(
    sc_list: List[SkyComponent], nmoment=1, reference_frequency=None
) -> List[List[SkyComponent]]:
    """Calculate frequency taylor terms for a list of skycomponents

    :param sc_list: List of SkyComponent
    :param nmoment: Number of moments/Taylor terms to use
    :param reference_frequency: Reference frequency (default None uses centre point)
    :return: SkyComponents as one component per Taylor term
    """
    if len(sc_list) == 0:
        return [nmoment * list()]
    nchan = len(sc_list[0].frequency)
    if reference_frequency is None:
        reference_frequency = sc_list[0].frequency[len(sc_list[0].frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    channel_moment_coupling = numpy.zeros([nchan, nmoment])
    for chan in range(nchan):
        for m in range(nmoment):
            channel_moment_coupling[chan, m] += numpy.power(
                (sc_list[0].frequency[chan] - reference_frequency)
                / reference_frequency,
                m,
            )

    pinv = numpy.linalg.pinv(channel_moment_coupling, rcond=1e-7)

    newsc_list = list()
    for sc in sc_list:
        taylor_term_sc_list = list()
        for moment in range(nmoment):
            taylor_term_data = numpy.zeros([1, sc.polarisation_frame.npol])
            for chan in range(nchan):
                taylor_term_data[0] += pinv[moment, chan] * sc.flux[chan, 0]
            taylor_term_sc = copy_skycomponent(sc)
            taylor_term_sc.flux = taylor_term_data
            taylor_term_sc.frequency = reference_frequency
            taylor_term_sc_list.append(taylor_term_sc)
        newsc_list.append(taylor_term_sc_list)

    return newsc_list


def find_skycomponents_frequency_taylor_terms(
    dirty_list: List[Image], nmoment=1, reference_frequency=None, **kwargs
) -> List[List[SkyComponent]]:
    """Find skycomponents by fitting to moment0, fit polynomial in frequency, return in frequency space

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param dirty_list: List of images to be searched. These should be different frequencies
    :param nmoment: Number of moments to be fitted
    :param reference_frequency: Reference frequency (default None uses centre frequency)
    :return: list of skycomponents
    """
    frequency = numpy.array([d.frequency[0] for d in dirty_list])

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "find_skycomponents_frequency_taylor_terms: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    moment0_list = calculate_frequency_taylor_terms_from_image_list(
        dirty_list, nmoment=1, reference_frequency=reference_frequency
    )
    threshold = get_parameter(kwargs, "component_threshold", numpy.inf)
    try:
        moment0_skycomponents = find_skycomponents(moment0_list[0], threshold=threshold)
    except ValueError:
        log.info(
            f"find_skycomponents_frequency_taylor_terms: No skycomponents found in moment 0"
        )
        return []

    ncomps = len(moment0_skycomponents)
    if ncomps > 0:
        log.info(
            f"find_skycomponents_frequency_taylor_terms: found {ncomps} skycomponents in moment 0"
        )
    else:
        return []

    found_component_list = []
    for isc, sc in enumerate(moment0_skycomponents):
        found_component = copy_skycomponent(sc)
        found_component.frequency = frequency
        found_component.flux = numpy.array(
            [list(fit_skycomponent(d, sc, **kwargs).flux[0, :]) for d in dirty_list]
        )
        found_component_list.append(found_component)
        log.info(f"Component {isc}: {found_component}")

    interpolated_sc_list = interpolate_skycomponents_frequency(
        found_component_list, nmoment=nmoment, reference_frequency=reference_frequency
    )
    return transpose_skycomponents_to_channels(interpolated_sc_list)


def interpolate_skycomponents_frequency(
    sc_list, nmoment=1, reference_frequency=None, **kwargs
) -> List[SkyComponent]:
    """Smooth skycomponent fluxes by fitting polynomial in frequency

    Each skycomponent in a list is interpolated in frequency using a Taylor series expansion.

    :param sc_list: List of skycomponents to be interpolated (in frequency_
    :param nmoment: Number of moments to be fitted
    :param reference_frequency: Reference frequency (default None uses central frequency)
    :return: list of interpolated skycomponents
    """
    frequency = sc_list[0].frequency

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "interpolate_skycomponents_frequency: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    # Now fit in frequency and keep the model
    newsc_list = list()
    for sc in sc_list:
        newsc = copy_skycomponent(sc)
        x = (frequency - reference_frequency) / reference_frequency
        y = sc.flux
        coeffs = polynomial.polyfit(x, y, nmoment - 1)
        newsc.flux = polynomial.polyval(x, coeffs).T
        newsc_list.append(newsc)

    return newsc_list


def transpose_skycomponents_to_channels(
    sc_list: List[SkyComponent],
) -> List[List[SkyComponent]]:
    """Tranpose a component list from [source,chan] to [chan,source]


    :param sc_list: List of SkyComponents
    :return: List[List[SkyComponent]]
    """
    newsc_list = list()
    nchan = len(sc_list[0].frequency)
    for chan in range(nchan):
        chan_sc_list = list()
        for comp in sc_list:
            newcomp = copy_skycomponent(comp)
            newcomp.frequency = numpy.array([comp.frequency[chan]])
            newcomp.flux = comp.flux[chan, :][numpy.newaxis, :]
            chan_sc_list.append(newcomp)
        newsc_list.append(chan_sc_list)
    return newsc_list


def gather_skycomponents_from_channels(
    sc_list: List[List[SkyComponent]],
) -> List[SkyComponent]:
    """Gather a component list from [chan][source] to [source]

    This function converts list of lists of single frequency skycomponents into
    a list of multi-frequency skycomponents

    :param sc_list:
    :return: List[List[SkyComponent]]
    """
    nsource = len(sc_list[0])
    nchan = len(sc_list)
    newsc_list = list()
    for source in range(nsource):
        newcomp = copy_skycomponent(sc_list[0][source])
        flux = numpy.array([sc_list[chan][source].flux[0, :] for chan in range(nchan)])
        frequency = numpy.array(
            [sc_list[chan][source].frequency[0] for chan in range(nchan)]
        )
        newcomp.frequency = frequency
        newcomp.flux = flux
        newsc_list.append(newcomp)
    return newsc_list
