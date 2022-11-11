# pylint: disable=invalid-name, too-many-arguments, unused-argument
# pylint: disable=import-error, no-name-in-module
"""
Functions that perform gather/scatter operations on Images.
"""

__all__ = [
    "image_gather_channels",
    "image_scatter_channels",
    "image_gather_facets",
    "image_scatter_facets",
]

import logging
import numpy
from typing import List

import xarray
from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.image.image_model import Image

from ska_sdp_func_python.image.iterators import image_raster_iter


log = logging.getLogger("func-python-logger")


def image_scatter_facets(
    im: Image, facets=1, overlap=0, taper=None
) -> List[Image]:
    """Scatter an image into a list of subimages using the  image_raster_iterator

     If the overlap is greater than zero, we choose to keep all images the same size so the
     other ring of facets are ignored. So if facets=4 and overlap > 0 then the scatter returns
     (facets-2)**2 = 4 images.

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: Number of pixels overlap
    :param taper: Taper at edges None or 'linear'
    :return: list of subimages

     See also:
        :py:func:`processing_components.image.iterators.image_raster_iter`
    """
    if im is None:
        return None

    return list(
        image_raster_iter(im, facets=facets, overlap=overlap, taper=taper)
    )


def image_gather_facets(
    image_list: List[Image],
    im: Image,
    facets=1,
    overlap=0,
    taper=None,
    return_flat=False,
):
    """Gather a list of subimages back into an image using the  image_raster_iterator

     If the overlap is greater than zero, we choose to keep all images the same size so the
     other ring of facets are ignored. So if facets=4 and overlap > 0 then the gather expects
     (facets-2)**2 = 4 images.

     To normalise the overlap we make a set of flats, gather that and divide.
     The flat may be optionally returned instead of the result

    :param image_list: List of subimages
    :param im: Output image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: Overlap between neighbours in pixels
    :param taper: Taper at edges None or 'linear' or 'Tukey'
    :param return_flat: Return the flat
    :return: list of subimages

     See also
        :py:func:`ska_sdp_func_python.image.iterators.image_raster_iter`
    """
    out = create_image(
        im["pixels"].data.shape[3],
        cellsize=numpy.deg2rad(numpy.abs(im.image_acc.wcs.wcs.cdelt[1])),
        phasecentre=im.image_acc.phasecentre)
    if overlap > 0:
        flat = create_image(
            im["pixels"].data.shape[3],
            cellsize=numpy.deg2rad(numpy.abs(im.image_acc.wcs.wcs.cdelt[1])),
            phasecentre=im.image_acc.phasecentre,
        )
        flat["pixels"].data[...] = 1.0
        flats = list(
            image_raster_iter(
                flat,
                facets=facets,
                overlap=overlap,
                taper=taper,
                make_flat=True,
            )
        )

        sum_flats = create_image(
            im["pixels"].data.shape[3],
            cellsize=numpy.deg2rad(numpy.abs(im.image_acc.wcs.wcs.cdelt[1])),
            phasecentre=im.image_acc.phasecentre)

        if return_flat:
            i = 0
            for sum_flat_facet in image_raster_iter(
                sum_flats, facets=facets, overlap=overlap, taper=taper
            ):
                sum_flat_facet["pixels"].data[...] += flats[i]["pixels"].data[
                    ...
                ]
                i += 1

            return sum_flats

        i = 0
        for out_facet, sum_flat_facet in zip(
            image_raster_iter(
                out, facets=facets, overlap=overlap, taper=taper
            ),
            image_raster_iter(
                sum_flats, facets=facets, overlap=overlap, taper=taper
            ),
        ):
            out_facet["pixels"].data[...] += (
                flats[i]["pixels"].data * image_list[i]["pixels"].data[...]
            )
            sum_flat_facet["pixels"].data[...] += flats[i]["pixels"].data[...]
            i += 1

        out["pixels"].data[sum_flats["pixels"].data > 0.0] /= sum_flats[
            "pixels"
        ].data[sum_flats["pixels"].data > 0.0]
        out["pixels"].data[sum_flats["pixels"].data <= 0.0] = 0.0

        return out

    # if no overlap
    flat = create_image(im["pixels"].data.shape[3],
                        cellsize=numpy.deg2rad(numpy.abs(im.image_acc.wcs.wcs.cdelt[1])),
                        phasecentre=im.image_acc.phasecentre
                        )
    flat["pixels"].data[...] = 1.0

    if return_flat:
        return flat

    for i, facet in enumerate(
        image_raster_iter(out, facets=facets, overlap=overlap, taper=taper)
    ):
        facet["pixels"].data[...] += image_list[i]["pixels"].data[...]

    return out


def image_scatter_channels(im: Image, subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the channels

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

     See also
        :py:func:`ska_sdp_func_python.image.iterators.image_channel_iter`
    """
    if im is None:
        return None

    return [
        r[1]
        for r in im.groupby_bins("frequency", bins=subimages, squeeze=False)
    ]


def image_gather_channels(
    image_list: List[Image], im: Image = None, subimages=0
) -> Image:
    """Gather a list of subimages back into an image

    :param image_list: List of subimages
    :param im: Output image
    :param subimages: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    return xarray.concat(image_list, dim="frequency")
