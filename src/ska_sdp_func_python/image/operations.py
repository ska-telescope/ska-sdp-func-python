"""
Image operations visible to the Execution Framework as Components.
"""

__all__ = [
    "convert_clean_beam_to_degrees",
    "convert_clean_beam_to_pixels",
    "convert_polimage_to_stokes",
    "convert_stokes_to_polimage",
]

import logging
import warnings

import numpy
from astropy.wcs import FITSFixedWarning
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_circular_to_stokes,
    convert_linear_to_stokes,
    convert_stokes_to_circular,
    convert_stokes_to_linear,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

warnings.simplefilter("ignore", FITSFixedWarning)
log = logging.getLogger("func-python-logger")


def convert_clean_beam_to_degrees(im, beam_pixels):
    """Convert clean beam in pixels to deg, deg, deg.

    :param im: Image
    :param beam_pixels: Beam size in pixels
    :return: dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}.
             Units are deg, deg, deg
    """
    # cellsize in radians
    cellsize = numpy.deg2rad(im.image_acc.wcs.wcs.cdelt[1])
    to_mm = numpy.sqrt(8.0 * numpy.log(2.0))
    if beam_pixels[1] > beam_pixels[0]:
        clean_beam = {
            "bmaj": numpy.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bmin": numpy.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bpa": numpy.rad2deg(beam_pixels[2]),
        }
    else:
        clean_beam = {
            "bmaj": numpy.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bmin": numpy.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bpa": numpy.rad2deg(beam_pixels[2]) + 90.0,
        }
    return clean_beam


def convert_clean_beam_to_pixels(model, clean_beam):
    """Convert clean beam to pixels.

    :param model: Model image containing beam information
    :param clean_beam: e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}.
                Units are deg, deg, deg
    :return: Beam size in pixels
    """
    to_mm = numpy.sqrt(8.0 * numpy.log(2.0))
    # Cellsize in radians
    cellsize = numpy.deg2rad(model.image_acc.wcs.wcs.cdelt[1])
    # Beam in pixels
    beam_pixels = (
        numpy.deg2rad(clean_beam["bmin"]) / (cellsize * to_mm),
        numpy.deg2rad(clean_beam["bmaj"]) / (cellsize * to_mm),
        numpy.deg2rad(clean_beam["bpa"]),
    )
    return beam_pixels


def convert_stokes_to_polimage(
    im: Image, polarisation_frame: PolarisationFrame
):
    """Convert a stokes image in IQUV to polarisation_frame.

    For example::
        impol = convert_stokes_to_polimage(imIQUV, PolarisationFrame('linear'))

    :param im: Image to be converted
    :param polarisation_frame: desired polarisation frame
    :returns: Complex image

    See also
        :py:func:`ska_sdp_func_python.image.operations.convert_polimage_to_stokes`
        :py:func:`ska_sdp_datamodels.polarisation.convert_circular_to_stokes`
        :py:func:`ska_sdp_datamodels.polarisation.convert_linear_to_stokes`
    """

    if polarisation_frame == PolarisationFrame("linear"):
        cimarr = convert_stokes_to_linear(im["pixels"].data)
        # Need to make sure image data is copied (cimarr in this case)
        return Image.constructor(
            data=cimarr,
            polarisation_frame=polarisation_frame,
            wcs=im.image_acc.wcs,
        )
    if polarisation_frame == PolarisationFrame("linearnp"):
        cimarr = convert_stokes_to_linear(im["pixels"].data)
        return Image.constructor(
            data=cimarr,
            polarisation_frame=polarisation_frame,
            wcs=im.image_acc.wcs,
        )
    if polarisation_frame == PolarisationFrame("circular"):
        cimarr = convert_stokes_to_circular(im["pixels"].data)
        return Image.constructor(
            data=cimarr,
            polarisation_frame=polarisation_frame,
            wcs=im.image_acc.wcs,
        )
    if polarisation_frame == PolarisationFrame("circularnp"):
        cimarr = convert_stokes_to_circular(im["pixels"].data)
        return Image.constructor(
            data=cimarr,
            polarisation_frame=polarisation_frame,
            wcs=im.image_acc.wcs,
        )
    if polarisation_frame == PolarisationFrame("stokesI"):
        return Image.constructor(
            data=im["pixels"].data.astype("complex"),
            polarisation_frame=PolarisationFrame("stokesI"),
            wcs=im.image_acc.wcs,
        )

    raise ValueError(f"Cannot convert stokes to {polarisation_frame.type}")


def convert_polimage_to_stokes(im: Image, complex_image=False):
    """Convert a polarisation image to stokes IQUV (complex).

    For example:
        imIQUV = convert_polimage_to_stokes(impol)

    :param im: Complex Image in linear or circular
    :param complex_image: boolean for complex image, (default False)
    :returns: Complex or Real image

    See also
        :py:func:`ska_sdp_func_python.image.operations.convert_stokes_to_polimage`
        :py:func:`ska_sdp_datamodels.polarisation.convert_stokes_to_circular`
        :py:func:`ska_sdp_datamodels.polarisation.convert_stokes_to_linear`

    """
    assert im["pixels"].data.dtype == "complex", im["pixels"].data.dtype

    def _to_required(data):
        if complex_image:
            return data

        return numpy.real(data)

    if im.image_acc.polarisation_frame == PolarisationFrame("linear"):
        cimarr = convert_linear_to_stokes(im["pixels"].data)
        return Image.constructor(
            data=_to_required(cimarr),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            wcs=im.image_acc.wcs,
        )
    if im.image_acc.polarisation_frame == PolarisationFrame("linearnp"):
        cimarr = convert_linear_to_stokes(im["pixels"].data)
        return Image.constructor(
            data=_to_required(cimarr),
            polarisation_frame=PolarisationFrame("stokesIQ"),
            wcs=im.image_acc.wcs,
        )
    if im.image_acc.polarisation_frame == PolarisationFrame("circular"):
        cimarr = convert_circular_to_stokes(im["pixels"].data)
        return Image.constructor(
            data=_to_required(cimarr),
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            wcs=im.image_acc.wcs,
        )
    if im.image_acc.polarisation_frame == PolarisationFrame("circularnp"):
        cimarr = convert_circular_to_stokes(im["pixels"].data)
        return Image.constructor(
            data=_to_required(cimarr),
            polarisation_frame=PolarisationFrame("stokesIV"),
            wcs=im.image_acc.wcs,
        )
    if im.image_acc.polarisation_frame == PolarisationFrame("stokesI"):
        return Image.constructor(
            data=_to_required(im["pixels"].data),
            polarisation_frame=PolarisationFrame("stokesI"),
            wcs=im.image_acc.wcs,
        )

    raise ValueError(
        f"Cannot convert {im.image_acc.polarisation_frame.type} to stokes"
    )
