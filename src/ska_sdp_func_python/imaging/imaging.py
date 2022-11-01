"""
Functions for predicting visibility from a model image, and invert a visibility to
make an (image, sumweights) tuple. These redirect to specific versions.
"""

__all__ = [
    "predict_visibility",
    "invert_visibility",
]

import logging

import numpy
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.visibility.vis_model import Visibility

from src.ska_sdp_func_python.imaging.base import (
    predict_awprojection,
    invert_awprojection,
)
from src.ska_sdp_func_python.imaging.ng import predict_ng, invert_ng
from src.ska_sdp_func_python.imaging.wg import predict_wg, invert_wg

log = logging.getLogger("rascil-logger")


def predict_visibility(
    vis: Visibility, model: Image, context="ng", gcfcf=None, **kwargs
) -> Visibility:
    """Predict visibility from an image

    For awprojection, the gridding details must be supplied via a tuple of
    (gridding correction function, convolution function) or a partial
    to calculate it.

    :param vis: visibility to be predicted
    :param model: model image
    :param context: Type: 2d or awprojection, ng or wg (nifty-gridder or WAGG GPU-based gridder/degridder), default: ng
    :param gcfcf: Tuple of (grid correction function, convolution function) or partial function
    :return: resulting visibility (in place works)
    """
    if context == "awprojection":
        return predict_awprojection(vis, model, gcfcf=gcfcf, **kwargs)
    elif context == "2d":
        return predict_ng(vis, model, do_wstacking=False, **kwargs)
    elif context == "ng":
        return predict_ng(vis, model, **kwargs)
    elif context == "wg":
        return predict_wg(vis, model, **kwargs)
    else:
        raise ValueError(f"Unknown imaging context {context}")


def invert_visibility(
    vis: Visibility,
    im: Image,
    dopsf: bool = False,
    normalise: bool = True,
    context="ng",
    gcfcf=None,
    **kwargs,
) -> (Image, numpy.ndarray):
    """Invert visibility to make an (image, sum weights) tuple

    Use the image im as a template. Do PSF in a separate call.

    For awprojection, the gridding details must be supplied via a tuple of
    (gridding correction function, convolution function) or a partial
    to calculate it.

    :param vis: visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image (default: False)
    :param normalise: normalise by the sum of weights (default: True)
    :param context: Type: 2d or awprojection, ng or wg (nifty-gridder or WAGG GPU-based gridder/degridder), default: ng
    :param gcfcf: Tuple of (grid correction function, convolution function) or partial function
    :return: (resulting image, sum of weights)
    """

    if context == "awprojection":
        return invert_awprojection(
            vis, im, dopsf=dopsf, normalise=normalise, gcfcf=gcfcf, **kwargs
        )
    elif context == "2d":
        return invert_ng(
            vis, im, dopsf=dopsf, normalise=normalise, do_wstacking=False, **kwargs
        )
    elif context == "ng":
        return invert_ng(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)
    elif context == "wg":
        return invert_wg(vis, im, dopsf=dopsf, normalise=normalise, **kwargs)

    else:
        raise ValueError(f"Unknown imaging context {context}")
