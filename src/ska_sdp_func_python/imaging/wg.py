"""
Functions that implement prediction of and imaging from visibilities
using the GPU-based w-stacking gridder from ska-sdp-func
(which is compatible with the DUCC/nifty gridder).
"""

__all__ = [
    "invert_wg",
    "predict_wg",
]

import copy
import logging

import numpy

try:
    import cupy
except ImportError:
    cupy = None

from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_pol_frame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.imaging.base import (
    normalise_sumwt,
    shift_vis_to_image,
)

try:
    from ska_sdp_func.grid_data import GridderUvwEsFft
except ImportError:
    GridderUvwEsFft = None

log = logging.getLogger("func-python-logger")


def predict_wg(bvis: Visibility, model: Image, **kwargs) -> Visibility:
    """Predict using GPU-based w-stacking degridder module.

    In the imaging and pipeline workflows, this may
    be invoked using context='wg'.

    :param bvis: Visibility to be predicted
    :param model: Model Image
    :return: Resulting Visibility (in place works)
    """
    if not GridderUvwEsFft:
        log.warning("ska-sdp-func is not installed. Cannot run predict_wg")
        return None

    if not cupy:
        log.warning("cupy is not installed. Cannot run predict_wg")
        return None

    if not model.image_acc.is_canonical():
        log.error(
            "The image model is not canonical. See the logfile. Exiting..."
        )
        raise ValueError("WG:The image model is not canonical")

    # If the model is None than just return the input visibility.
    if model is None:
        return bvis

    epsilon = kwargs.get("epsilon", 1e-12)
    do_wstacking = kwargs.get("do_wstacking", True)

    newbvis = bvis.copy(deep=True, zero=True)

    # Extracting data from Visibility
    freq = bvis.frequency.data  # frequency, Hz
    nrows, nbaselines, vnchan, vnpol = bvis.vis.shape

    uvw = newbvis.uvw.data
    uvw = uvw.reshape([nrows * nbaselines, 3])
    uvw = numpy.nan_to_num(uvw)
    vis_temp = numpy.zeros(
        [vnpol, vnchan, nbaselines * nrows], dtype="complex"
    )

    # Get the image properties
    m_nchan = model["pixels"].data.shape[0]
    m_npol = model["pixels"].data.shape[1]

    if m_npol != vnpol:
        log.error(
            "The number of polarisations in bvis "
            "and a model does not match, exiting..."
        )
        raise ValueError(
            "WG: The number of polarisations in "
            "bvis and a model does not match"
        )

    flipped_uvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes. The flip in w is
    # equivalent to the conjugation of the
    # convolution function grid_visibility to griddata
    flipped_uvw[:, 0] *= -1.0
    flipped_uvw[:, 2] *= -1.0

    # Find out the image size/resolution
    pixsize = numpy.abs(numpy.radians(model.image_acc.wcs.wcs.cdelt[0]))

    # Make de-gridding over a frequency range and pol fields
    vis_to_im = numpy.round(
        model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
    ).astype("int")

    # Copy uvw-coordinates to GPU and allocate scratch GPU memory.
    uvw_gpu = cupy.asarray(flipped_uvw)
    vis_gpu = cupy.zeros([nbaselines * nrows], dtype=vis_temp.dtype)
    weight_gpu = cupy.ones(
        vis_gpu.shape,
        dtype=cupy.float32
        if vis_gpu.dtype == cupy.complex64
        else cupy.float64,
    )  # FIXME Not sure if this should also be obtained from input bvis?

    if m_nchan == 1:
        freq_gpu = cupy.asarray(freq)
        for vpol in range(vnpol):
            image_gpu = cupy.asarray(model["pixels"].data[0, vpol, :, :].T)
            vis_gpu.fill(0)
            gridder = GridderUvwEsFft(
                uvw_gpu,
                freq_gpu,
                vis_gpu,
                weight_gpu,
                image_gpu,
                pixsize,
                pixsize,
                epsilon,
                do_wstacking,
            )
            gridder.ifft_grid_uvw_es(
                uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu
            )
            vis_temp[vpol, 0, :] = cupy.asnumpy(vis_gpu)
    else:
        for vpol in range(vnpol):
            for vchan in range(vnchan):
                imchan = vis_to_im[vchan]
                freq_gpu = cupy.array(freq[vchan : vchan + 1])
                image_gpu = cupy.asarray(
                    model["pixels"].data[imchan, vpol, :, :].T
                )
                vis_gpu.fill(0)
                gridder = GridderUvwEsFft(
                    uvw_gpu,
                    freq_gpu,
                    vis_gpu,
                    weight_gpu,
                    image_gpu,
                    pixsize,
                    pixsize,
                    epsilon,
                    do_wstacking,
                )
                gridder.ifft_grid_uvw_es(
                    uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu
                )
                vis_temp[vpol, vchan, :] = cupy.asnumpy(vis_gpu)
    vis = convert_pol_frame(
        vis_temp.T,
        model.image_acc.polarisation_frame,
        bvis.visibility_acc.polarisation_frame,
        polaxis=2,
    )

    vis = vis.reshape([nrows, nbaselines, vnchan, vnpol])
    newbvis["vis"].data = vis

    # Now we can shift the visibility from the image frame
    # to the original visibility frame
    return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)


def invert_wg(
    bvis: Visibility,
    model: Image,
    dopsf: bool = False,
    normalise: bool = True,
    **kwargs
) -> (Image, numpy.ndarray):
    """
    Invert using GPU-based w-stacking gridder module.

    Use the Image im as a template. Do PSF in a separate call.

    In the imaging and pipeline workflows, this may be
    invoked using context='wg'.

    :param dopsf: Make the PSF instead of the dirty image
    :param bvis: Visibility to be inverted
    :param model: Image template (not changed)
    :param normalise: Normalise by the sum of weights (True)
    :return: (resulting Image, sum of the weights for
              each frequency and polarization)
    """
    if not GridderUvwEsFft:
        log.warning("ska-sdp-func is not installed. Cannot run invert_wg")
        return None

    if not cupy:
        log.warning("cupy is not installed. Cannot run invert_wg")
        return None

    if not model.image_acc.is_canonical():
        log.error(
            "The image model is not canonical. See the logfile. Exiting..."
        )
        raise ValueError("WG:The image model is not canonical")

    im = model.copy(deep=True)

    epsilon = kwargs.get("epsilon", 1e-12)
    do_wstacking = kwargs.get("do_wstacking", True)

    bvis_shifted = bvis.copy(deep=True)
    bvis_shifted = shift_vis_to_image(
        bvis_shifted, im, tangent=True, inverse=False
    )

    freq = bvis_shifted.frequency.data  # frequency, Hz

    nrows, nbaselines, vnchan, vnpol = bvis_shifted.vis.shape

    ms = bvis_shifted.visibility_acc.flagged_vis
    ms = ms.reshape([nrows * nbaselines, vnchan, vnpol])
    ms = convert_pol_frame(
        ms,
        bvis.visibility_acc.polarisation_frame,
        im.image_acc.polarisation_frame,
        polaxis=2,
    ).astype("c16")

    uvw = copy.deepcopy(bvis_shifted.uvw.data)
    uvw = uvw.reshape([nrows * nbaselines, 3])

    weight = bvis_shifted.visibility_acc.flagged_imaging_weight.astype("f8")
    weight = weight.reshape([nrows * nbaselines, vnchan, vnpol])

    # Find out the image size/resolution
    npixdirty = im["pixels"].data.shape[-1]
    pixsize = numpy.abs(numpy.radians(im.image_acc.wcs.wcs.cdelt[0]))

    flipped_uvw = copy.deepcopy(uvw)
    # We need to flip the u and w axes.
    flipped_uvw[:, 0] *= -1.0
    flipped_uvw[:, 2] *= -1.0

    nchan = im["pixels"].data.shape[0]
    npol = im["pixels"].data.shape[1]
    im["pixels"].data[...] = 0.0
    sum_weight = numpy.zeros([nchan, npol])

    # Set up the conversion from visibility channels to image channels
    vis_to_im = numpy.round(
        model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
    ).astype("int")

    mfs = nchan == 1 and vnchan > 1
    ms_temp = ms.T
    weight_temp = weight.T
    if dopsf:
        ms_temp[...] = 0.0
        ms_temp[0, ...] = 1.0

    # Copy uvw-coordinates to GPU and allocate scratch GPU memory.
    uvw_gpu = cupy.asarray(flipped_uvw)
    image_gpu = cupy.zeros(
        [npixdirty, npixdirty],
        dtype=cupy.float32 if ms.dtype == numpy.complex64 else cupy.float64,
    )

    # lms and lwt - the contiguous versions of ms_temp and weight_temp
    if mfs:
        freq_gpu = cupy.asarray(freq)
        for pol in range(npol):
            lms = numpy.ascontiguousarray(ms_temp[pol, :, :].T)
            if numpy.max(numpy.abs(lms)) > 0.0:
                lwt = numpy.ascontiguousarray(weight_temp[pol, :, :].T)
                vis_gpu = cupy.asarray(lms)
                weight_gpu = cupy.asarray(lwt)
                image_gpu.fill(0)
                gridder = GridderUvwEsFft(
                    uvw_gpu,
                    freq_gpu,
                    vis_gpu,
                    weight_gpu,
                    image_gpu,
                    pixsize,
                    pixsize,
                    epsilon,
                    do_wstacking,
                )
                gridder.grid_uvw_es_fft(
                    uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu
                )
                im["pixels"].data[0, pol] += cupy.asnumpy(image_gpu).T
            sum_weight[0, pol] += numpy.sum(weight_temp[pol, :, :].T)
    else:
        for pol in range(npol):
            for vchan in range(vnchan):
                ichan = vis_to_im[vchan]
                lms = numpy.ascontiguousarray(
                    ms_temp[pol, vchan, :, numpy.newaxis]
                )
                if numpy.max(numpy.abs(lms)) > 0.0:
                    lwt = numpy.ascontiguousarray(
                        weight_temp[pol, vchan, :, numpy.newaxis]
                    )
                    freq_gpu = cupy.array(freq[vchan : vchan + 1])
                    vis_gpu = cupy.asarray(lms)
                    weight_gpu = cupy.asarray(lwt)
                    image_gpu.fill(0)
                    gridder = GridderUvwEsFft(
                        uvw_gpu,
                        freq_gpu,
                        vis_gpu,
                        weight_gpu,
                        image_gpu,
                        pixsize,
                        pixsize,
                        epsilon,
                        do_wstacking,
                    )
                    gridder.grid_uvw_es_fft(
                        uvw_gpu, freq_gpu, vis_gpu, weight_gpu, image_gpu
                    )
                    im["pixels"].data[ichan, pol] += cupy.asnumpy(image_gpu).T
                sum_weight[ichan, pol] += numpy.sum(
                    weight_temp[pol, vchan, :].T, axis=0
                )

    if normalise:
        im = normalise_sumwt(im, sum_weight)

    return im, sum_weight
