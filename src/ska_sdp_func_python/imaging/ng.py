"""
Functions that implement prediction of and imaging
from visibilities using the nifty gridder (DUCC version).

https://gitlab.mpcdf.mpg.de/mtr/ducc.git

This performs all necessary w term corrections, to high precision.

Note that nifty gridder doesn't like some null
data such as all w = 0 and do_wstacking=True.
Also true of the visibilities.
"""

__all__ = [
    "invert_ng",
    "predict_ng",
]

import copy
import logging

import ducc0.wgridder as ng  # pylint: disable=import-error,no-name-in-module
import numpy
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_pol_frame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.imaging.base import (
    normalise_sumwt,
    shift_vis_to_image,
)

log = logging.getLogger("func-python-logger")


def predict_ng(bvis: Visibility, model: Image, **kwargs) -> Visibility:
    """
    Predict using convolutional degridding.

    Nifty-gridder version. https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

    In the imaging and pipeline workflows,
    this may be invoked using context='ng'.

    :param bvis: Visibility to be predicted
    :param model: Model Image
    :return: Resulting Visibility (in place works)
    """

    if model is None:
        return bvis

    assert isinstance(model, Image), model
    assert model.image_acc.is_canonical()

    nthreads = kwargs.get("threads", 4)
    epsilon = kwargs.get("epsilon", 1e-12)
    do_wstacking = kwargs.get("do_wstacking", True)
    verbosity = kwargs.get("verbosity", 0)

    newbvis = bvis.copy(deep=True, zero=True)

    # Extracting data from Visibility
    freq = bvis.frequency.data  # frequency, Hz
    nrows, nbaselines, vnchan, vnpol = bvis.vis.shape

    uvw = newbvis.uvw.data
    uvw = uvw.reshape([nrows * nbaselines, 3])
    uvw = numpy.nan_to_num(uvw)
    vist = numpy.zeros([vnpol, vnchan, nbaselines * nrows], dtype="complex")

    # Get the image properties
    m_nchan = model["pixels"].data.shape[0]
    m_npol = model["pixels"].data.shape[1]

    assert m_npol == vnpol

    fuvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes. The flip in w
    # is equivalent to the conjugation of the
    # convolution function grid_visibility to griddata
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0

    # Find out the image size/resolution
    pixsize = numpy.abs(numpy.radians(model.image_acc.wcs.wcs.cdelt[0]))

    # Make de-gridding over a frequency range and pol fields
    vis_to_im = numpy.round(
        model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
    ).astype("int")

    mfs = m_nchan == 1

    if mfs:
        for vpol in range(vnpol):
            vist[vpol, :, :] = ng.dirty2ms(
                fuvw.astype(float),
                bvis.frequency.data.astype(float),
                model["pixels"].data[0, vpol, :, :].T.astype(float),
                None,
                pixsize,
                pixsize,
                0,
                0,
                epsilon,
                do_wstacking,
                nthreads,
                verbosity,
            ).T
    else:
        for vpol in range(vnpol):
            for vchan in range(vnchan):
                imchan = vis_to_im[vchan]
                vist[vpol, vchan, :] = ng.dirty2ms(
                    fuvw.astype(float),
                    numpy.array(freq[vchan : vchan + 1]).astype(float),
                    model["pixels"].data[imchan, vpol, :, :].T.astype(float),
                    None,
                    pixsize,
                    pixsize,
                    0,
                    0,
                    epsilon,
                    do_wstacking,
                    nthreads,
                )[:, 0]

    vis = convert_pol_frame(
        vist.T,
        model.image_acc.polarisation_frame,
        bvis.visibility_acc.polarisation_frame,
        polaxis=2,
    )

    vis = vis.reshape([nrows, nbaselines, vnchan, vnpol])
    newbvis["vis"].data = vis

    # Now we can shift the visibility from the image
    # frame to the original visibility frame
    return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)


def invert_ng(
    bvis: Visibility,
    model: Image,
    dopsf: bool = False,
    normalise: bool = True,
    **kwargs,
) -> (Image, numpy.ndarray):
    """Invert using nifty-gridder module.

     https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

    Use the Image im as a template. Do PSF in a separate call.

    In the imaging and pipeline workflows,
    this may be invoked using context='ng'. It is the default.

    :param dopsf: Make the PSF instead of the dirty image
    :param bvis: Visibility to be inverted
    :param im: Image template (not changed)
    :param normalise: Normalise by the sum of weights (True)
    :return: (resulting Image, sum of the weights for
                each frequency and polarization)

    """

    # Make sure we are dealing with an Image
    assert isinstance(model, Image), model
    assert model.image_acc.is_canonical()

    im = model.copy(deep=True)

    nthreads = kwargs.get("threads", 4)
    epsilon = kwargs.get("epsilon", 1e-12)
    do_wstacking = kwargs.get("do_wstacking", True)
    verbosity = kwargs.get("verbosity", 0)

    sbvis = bvis.copy(deep=True)
    sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)

    freq = sbvis.frequency.data  # frequency, Hz

    nrows, nbaselines, vnchan, vnpol = sbvis.vis.shape
    # if dopsf:
    #     sbvis = fill_vis_for_psf(sbvis)

    ms = sbvis.visibility_acc.flagged_vis
    ms = ms.reshape([nrows * nbaselines, vnchan, vnpol])
    ms = convert_pol_frame(
        ms,
        bvis.visibility_acc.polarisation_frame,
        im.image_acc.polarisation_frame,
        polaxis=2,
    ).astype("c16")

    uvw = copy.deepcopy(sbvis.uvw.data)
    uvw = uvw.reshape([nrows * nbaselines, 3])

    wgt = sbvis.visibility_acc.flagged_imaging_weight.astype("f8")
    wgt = wgt.reshape([nrows * nbaselines, vnchan, vnpol])

    # Find out the image size/resolution
    npixdirty = im["pixels"].data.shape[-1]
    pixsize = numpy.abs(numpy.radians(im.image_acc.wcs.wcs.cdelt[0]))

    fuvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes.
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0

    nchan = im["pixels"].data.shape[0]
    npol = im["pixels"].data.shape[1]
    im["pixels"].data[...] = 0.0
    sumwt = numpy.zeros([nchan, npol])

    # Set up the conversion from visibility channels to image channels
    vis_to_im = numpy.round(
        model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
    ).astype("int")

    # Nifty gridder likes to receive contiguous arrays
    # so we transpose at the beginning

    mfs = nchan == 1 and vnchan > 1
    mst = ms.T
    wgtt = wgt.T
    if dopsf:
        mst[...] = 0.0
        mst[0, ...] = 1.0

    if mfs:
        for pol in range(npol):
            lms = numpy.ascontiguousarray(mst[pol, :, :].T)
            if numpy.max(numpy.abs(lms)) > 0.0:
                lwt = numpy.ascontiguousarray(wgtt[pol, :, :].T)
                dirty = ng.ms2dirty(
                    fuvw,
                    bvis.frequency.data,
                    lms,
                    lwt,
                    npixdirty,
                    npixdirty,
                    pixsize,
                    pixsize,
                    0,
                    0,
                    epsilon,
                    do_wstacking,  # =do_wstacking,
                    nthreads=nthreads,
                    double_precision_accumulation=True,
                    verbosity=verbosity,
                )
                im["pixels"].data[0, pol] += dirty.T
            sumwt[0, pol] += numpy.sum(wgtt[pol, :, :].T)
    else:
        for pol in range(npol):
            for vchan in range(vnchan):
                ichan = vis_to_im[vchan]
                frequency = numpy.array(freq[vchan : vchan + 1]).astype(float)
                lms = numpy.ascontiguousarray(
                    mst[pol, vchan, :, numpy.newaxis]
                )
                if numpy.max(numpy.abs(lms)) > 0.0:
                    lwt = numpy.ascontiguousarray(
                        wgtt[pol, vchan, :, numpy.newaxis]
                    )
                    dirty = ng.ms2dirty(
                        fuvw,
                        frequency,
                        lms,
                        lwt,
                        npixdirty,
                        npixdirty,
                        pixsize,
                        pixsize,
                        0,
                        0,
                        epsilon,
                        do_wstacking,
                        nthreads=nthreads,
                        double_precision_accumulation=True,
                        verbosity=verbosity,
                    )
                    im["pixels"].data[ichan, pol] += dirty.T
                sumwt[ichan, pol] += numpy.sum(wgtt[pol, vchan, :].T, axis=0)

    if normalise:
        im = normalise_sumwt(im, sumwt)

    return im, sumwt
