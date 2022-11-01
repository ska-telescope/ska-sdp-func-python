"""
Functions that aid weighting the visibility data prior to imaging.

There are two classes of functions:
    - Changing the weight dependent on noise level or sample density or a combination
    - Tapering the weihght spatially to avoid effects of sharp edges or to emphasize a given scale size in the image

"""

__all__ = ["weight_visibility", "taper_visibility_gaussian", "taper_visibility_tukey"]

import logging

import numpy
from ska_sdp_datamodels.image.image_model import Image

from src import phyconst
from src.ska_sdp_func_python.griddata.gridding import (
    grid_visibility_weight_to_griddata,
    griddata_visibility_reweight,
)
from src.ska_sdp_func_python.griddata.operations import create_griddata_from_image
from src.ska_sdp_func_python.util.array_functions import tukey_filter

log = logging.getLogger("rascil-logger")


def weight_visibility(vis, model, weighting="uniform", robustness=0.0, **kwargs):
    """Weight the visibility data

    This is done collectively so the weights are summed over all vis_lists and then
    corrected

    :param vis_list:
    :param model_imagelist: Model required to determine weighting parameters
    :param weighting: Type of weighting (uniform or robust or natural)
    :param robustness: Robustness parameter
    :param kwargs: Parameters for functions in graphs
    :return: Reweighted vis
    """

    assert isinstance(model, Image), model
    assert model.image_acc.is_canonical()

    griddata = create_griddata_from_image(
        model, polarisation_frame=vis.visibility_acc.polarisation_frame
    )
    griddata, sumwt = grid_visibility_weight_to_griddata(vis, griddata)
    vis = griddata_visibility_reweight(
        vis, griddata, weighting=weighting, robustness=robustness
    )

    return vis


def taper_visibility_gaussian(vis, beam=None):
    """Taper the visibility weights

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_components.imaging.weighting.weight_visibility`

    :param vis: visibility with imaging_weight's to be tapered
    :param beam: desired resolution (Full width half maximum, radians)
    :return: visibility with imaging_weight column modified
    """
    # assert isinstance(vis, Visibility), vis

    if beam is None:
        raise ValueError("Beam size not specified for Gaussian taper")

    # assert isinstance(vis, Visibility), vis
    # See http://mathworld.wolfram.com/FourierTransformGaussian.html
    scale_factor = numpy.pi**2 * beam**2 / (4.0 * numpy.log(2.0))

    for chan, freq in enumerate(vis.frequency.data):
        wave = phyconst.c_m_s / freq
        uvdistsq = (
            vis.visibility_acc.u.data**2 + vis.visibility_acc.v.data**2
        ) / wave**2
        wt = numpy.exp(-scale_factor * uvdistsq)
        vis.imaging_weight.data[..., chan, :] = (
            vis.visibility_acc.flagged_imaging_weight[..., chan, :]
            * wt[..., numpy.newaxis]
        )

    return vis


def taper_visibility_tukey(vis, tukey=0.1):
    """Taper the visibility weights

    This algorithm is present in WSClean.

    See https://sourceforge.net/p/wsclean/wiki/Tapering

    tukey, a circular taper that smooths the outer edge set by -maxuv-l
    inner-tukey, a circular taper that smooths the inner edge set by -minuv-l
    edge-tukey, a square-shaped taper that smooths the edge set by the uv grid and -taper-edge.

    These are cumulative. If You can reset the imaging_weights
    using :py:mod:`processing_components.imaging.weighting.weight_visibility`

    :param vis: visibility with imaging_weight's to be tapered
    :return: visibility with imaging_weight column modified
    """

    # assert isinstance(vis, Visibility), vis

    oshape = vis.imaging_weight.data[..., 0, 0].shape
    for chan, freq in enumerate(vis.frequency.data):
        wave = phyconst.c_m_s / freq
        uvdist = numpy.sqrt(
            vis.visibility_acc.u.data**2 + vis.visibility_acc.v.data**2
        )
        uvdist = uvdist.flatten() / wave
        uvdistmax = numpy.max(uvdist)
        uvdist /= uvdistmax
        wt = numpy.array([tukey_filter(uv, tukey) for uv in uvdist]).reshape(oshape)
        vis.imaging_weight.data[..., chan, :] = (
            vis.visibility_acc.flagged_imaging_weight[..., chan, :]
            * wt[..., numpy.newaxis]
        )

    return vis
