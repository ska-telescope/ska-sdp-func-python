"""
Functions to invert and predict from skymodels
"""
__all__ = ["skymodel_calibrate_invert", "skymodel_predict_calibrate"]

import numpy

from ska_sdp_func_python.calibration import apply_gaintable
from ska_sdp_func_python.imaging.base import normalise_sumwt
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility
from ska_sdp_func_python.imaging.imaging import (
    invert_visibility,
    predict_visibility,
)
from ska_sdp_func_python.sky_component.operations import (
    apply_beam_to_skycomponent,
)
from ska_sdp_func_python.visibility import concatenate_visibility


def _dft_sky_component(vis_slice, skymodel, pb=None, **kwargs):
    """Run DFT of sky components"""
    if skymodel.mask is not None or pb is not None:
        comps = skymodel.components.copy()

        if skymodel.mask is not None:
            comps = apply_beam_to_skycomponent(comps, skymodel.mask)
        if pb is not None:
            comps = apply_beam_to_skycomponent(comps, pb)

        vis_slice = dft_skycomponent_visibility(vis_slice, comps, **kwargs)

    else:
        vis_slice = dft_skycomponent_visibility(
            vis_slice, skymodel.components, **kwargs
        )

    return vis_slice


def _fft_image(vis_slice, context, skymodel, pb=None, **kwargs):
    """Run FFT of image with non-zero pixel data"""
    imgv = vis_slice.copy(deep=True, zero=True)

    if skymodel.mask is not None or pb is not None:
        model = skymodel.image.copy(deep=True)

        if skymodel.mask is not None:
            model["pixels"].data *= skymodel.mask["pixels"].data
        if pb is not None:
            model["pixels"].data *= pb["pixels"].data

        imgv = predict_visibility(imgv, model, context=context, **kwargs)

    else:
        imgv = predict_visibility(
            imgv, skymodel.image, context=context, **kwargs
        )

    vis_slice["vis"].data += imgv["vis"].data


def skymodel_predict_calibrate(
    bvis,
    skymodel,
    context="ng",
    docal=False,
    inverse=True,
    get_pb=None,
    **kwargs
):
    """
    Predict visibility for a skymodel, optionally applying calibration

     A skymodel consists of an image and a list of components,
     optionally with a gaintable.

     The function get_pb should have the signature:

         get_pb(Visibility, Image)

     and should return the primary beam for the visibility.

    :param bvis: Input visibility
    :param skymodel: Skymodel
    :param context: Imaging context 2d or ng or awprojection
    :param get_pb: Function to get a primary beam
    :param docal: Apply calibration table in skymodel
    :param inverse: True means correction of calibration,
                    False means application of calibration
    :param kwargs: Parameters for functions in components
    :return: Visibility with dft of components, fft of image,
             gaintable applied (optional)
    """
    v = bvis.copy(deep=True, zero=True)

    vis_slices = []
    if get_pb is not None:
        # TODO: Expand control of the grouping, coord and step
        for _, vis_slice in v.groupby("time", squeeze=False):

            pb = get_pb(vis_slice, skymodel.image)

            # First do the DFT for the components
            if len(skymodel.components) > 0:
                vis_slice = _dft_sky_component(
                    vis_slice, skymodel, pb=pb, **kwargs
                )

            # Now do the FFT of the image, after multiplying
            # by the mask and primary beam
            if skymodel.image is not None:
                if numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0:
                    _fft_image(vis_slice, context, skymodel, pb=pb, **kwargs)

            vis_slices.append(vis_slice)

        v = concatenate_visibility(vis_slices, "time")

        if docal and skymodel.gaintable is not None:
            v = apply_gaintable(v, skymodel.gaintable, inverse=inverse)

        return v

    # First do the DFT or the components
    v = _dft_sky_component(v, skymodel, pb=None, **kwargs)

    # Now do the FFT of the image, after multiplying by the mask
    if skymodel.image is not None:
        if numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0:
            _fft_image(v, context, skymodel, pb=None, **kwargs)

    if docal and skymodel.gaintable is not None:
        v = apply_gaintable(v, skymodel.gaintable, inverse=inverse)

    return v


def skymodel_calibrate_invert(
    bvis,
    skymodel,
    context="ng",
    docal=False,
    get_pb=None,
    normalise=True,
    flat_sky=False,
    **kwargs
):
    """Inverse Fourier sum of visibility to image and components

     If the get_pb function is defined, the sum of weights will be
     an image

    :param bvis: Visibility to be transformed
    :param skymodel: Skymodel
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param docal: Apply calibration table in skymodel
    :param get_pb: Function to get the primary beam for a given image and vis
    :param normalise: Normalise the dirty image by sum of weights
    :param flat_sky: Make the flux values correct (instead of noise)
    :return: Skymodel containing transforms
    """

    if skymodel.image is None:
        raise ValueError("skymodel image is None")

    bvis_cal = bvis.copy(deep=True)

    if docal and skymodel.gaintable is not None:
        bvis_cal = apply_gaintable(bvis_cal, skymodel.gaintable)

    sum_flats = skymodel.image.copy(deep=True)
    sum_flats["pixels"][...] = 0.0
    sum_dirtys = skymodel.image.copy(deep=True)
    sum_dirtys["pixels"][...] = 0.0

    if get_pb is not None:
        # TODO: Expand control of the grouping, coord and step
        for _, vis_slice in bvis_cal.groupby("time", squeeze=False):

            pb = get_pb(vis_slice, skymodel.image)

            # Just do a straightforward invert for just this vis
            # and then apply the mask and primary beam if present
            # The return value result contains the weighted image and
            # the weights as an image (including mask and primary beam)
            result = invert_visibility(
                vis_slice,
                skymodel.image,
                context=context,
                normalise=False,
                **kwargs
            )
            flat = numpy.ones_like(result[0]["pixels"].data)
            if skymodel.mask is not None:
                flat *= skymodel.mask["pixels"].data
            if pb is not None:
                flat *= pb["pixels"].data

            # We need to apply the flat to the dirty image
            sum_dirtys["pixels"].data += flat * result[0]["pixels"].data
            # The sum_flats should contain the weights and the square of the PB
            sum_flats["pixels"].data += (
                flat * flat * result[1][:, :, numpy.newaxis, numpy.newaxis]
            )
        if normalise:
            sum_dirtys = normalise_sumwt(
                sum_dirtys, sum_flats, flat_sky=flat_sky
            )
            sum_flats["pixels"].data = numpy.sqrt(sum_flats["pixels"].data)

        return sum_dirtys, sum_flats

    result = invert_visibility(
        bvis_cal, skymodel.image, context=context, **kwargs
    )
    if skymodel.mask is not None:
        result[0]["pixels"].data *= skymodel.mask["pixels"].data

    return result
