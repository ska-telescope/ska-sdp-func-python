""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This module contains functions for performing the griddata process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.

GridData, ConvolutionFunction and Vis/BlockVis always have the same PolarisationFrame. Conversion to
stokesIQUV is only done in the image plane.
"""

__all__ = [
    "convolution_mapping_visibility",
    "grid_visibility_to_griddata",
    "degrid_visibility_from_griddata",
    "fft_griddata_to_image",
    "fft_image_to_griddata",
    "griddata_merge_weights",
    "grid_visibility_weight_to_griddata",
    "griddata_visibility_reweight",
]


import copy
import logging

import numpy
import numpy.testing
from ska_sdp_datamodels.gridded_visibility.grid_vis_model import GridData

from src.ska_sdp_func_python.fourier_transforms import ifft, fft
from src.ska_sdp_func_python.image.operations import create_image_from_array

log = logging.getLogger("rascil-logger")


def convolution_mapping_visibility(
    vis, griddata, chan, cf=None, channel_tolerance=1e-8
):
    """Find the mappings between visibility, griddata, and convolution function

    :param vis:
    :param griddata:
    :param cf:
    :param channel_tolerance:
    :return:
    """
    assert (
        vis.visibility_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )

    u = vis.visibility_acc.uvw_lambda[..., chan, 0].flat
    v = vis.visibility_acc.uvw_lambda[..., chan, 1].flat
    w = vis.visibility_acc.uvw_lambda[..., chan, 2].flat

    u = numpy.nan_to_num(u)
    v = numpy.nan_to_num(v)
    w = numpy.nan_to_num(w)

    return spatial_mapping(griddata, u, v, w, cf)


def spatial_mapping(griddata, u, v, w, cf=None):
    """Map u,v,w per row into coordinates in the grid

    :param cf:
    :param griddata:
    :return:
    """
    if cf is not None:
        assert (
            cf.convolutionfunction_acc.polarisation_frame
            == griddata.griddata_acc.polarisation_frame
        )

        nchan, npol, nw, ndv, ndu, nv, nu = cf.convolutionfunction_acc.shape

        grid_wcs = griddata.griddata_acc.griddata_wcs
        cf_wcs = cf.convolutionfunction_acc.cf_wcs
        numpy.testing.assert_almost_equal(grid_wcs.wcs.cdelt[0], cf_wcs.wcs.cdelt[0], 7)
        numpy.testing.assert_almost_equal(grid_wcs.wcs.cdelt[1], cf_wcs.wcs.cdelt[1], 7)
        # UV mapping:
        # We use the grid_wcs's to do the coordinate conversion
        # Find the nearest grid points

        pu_grid, pv_grid = numpy.round(
            grid_wcs.sub([1, 2]).wcs_world2pix(u, v, 0)
        ).astype("int")

        if ndu > 1 and ndv > 1:
            # We now have the location of grid points, convert back to uv space and
            # find the remainder (in wavelengths). We then use this to calculate the
            # subsampling indices (DUU, DVV)
            wu_grid, wv_grid = grid_wcs.sub([1, 2]).wcs_pix2world(pu_grid, pv_grid, 0)
            wu_subsample, wv_subsample = u - wu_grid, v - wv_grid
            pu_offset, pv_offset = numpy.round(
                cf_wcs.sub([3, 4]).wcs_world2pix(wu_subsample, wv_subsample, 0)
            ).astype("int")
            assert (
                numpy.min(pu_offset) >= 0
            ), "image sampling wrong: DU axis underflows: %f" % numpy.min(pu_offset)
            assert (
                numpy.max(pu_offset) < cf["pixels"].data.shape[3]
            ), "DU axis overflows: %f" % numpy.max(pu_offset)
            assert (
                numpy.min(pv_offset) >= 0
            ), "image sampling wrong: DV axis underflows: %f" % numpy.min(pv_offset)
            assert (
                numpy.max(pv_offset) < cf["pixels"].data.shape[4]
            ), "DV axis overflows: %f" % numpy.max(pv_offset)
        else:
            pu_offset = numpy.zeros_like(pu_grid)
            pv_offset = numpy.zeros_like(pv_grid)
        # W mapping for CF:
        if nw > 1:
            # nchan, npol, w, dv, du, v, u
            pwc_pixel = cf_wcs.sub([5]).wcs_world2pix(w, 0)[0]
            pwc_grid = numpy.round(pwc_pixel).astype("int")
            if numpy.min(pwc_grid) < 0:
                print(w[0:10])
                print(cf.convolutionfunction_acc.cf_wcs.sub([5]).__repr__())
            assert numpy.min(pwc_grid) >= 0, "W axis underflows: %f" % numpy.min(
                pwc_grid
            )
            assert (
                numpy.max(pwc_grid) < cf["pixels"].data.shape[2]
            ), "W axis overflows: %f" % numpy.max(pwc_grid)
            pwc_fraction = pwc_pixel - pwc_grid
        else:
            pwc_fraction = numpy.zeros_like(pu_grid)
            pwc_grid = numpy.zeros_like(pu_grid)

        return pu_grid, pu_offset, pv_grid, pv_offset, pwc_grid, pwc_fraction
    else:
        nchan, npol, nv, nu = griddata.griddata_acc.shape

        grid_wcs = griddata.griddata_acc.griddata_wcs
        # UV mapping:
        # We use the grid_wcs's to do the coordinate conversion
        # Find the nearest grid points
        pu_grid, pv_grid = numpy.round(
            grid_wcs.sub([1, 2]).wcs_world2pix(u, v, 0)
        ).astype("int")

        return pu_grid, pv_grid


def grid_visibility_to_griddata(vis, griddata, cf):
    """Grid Visibility onto a GridData

    :param vis: visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """
    assert (
        vis.visibility_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )

    griddata["pixels"].data[...] = 0.0

    vis_to_im = numpy.round(
        griddata.griddata_acc.griddata_wcs.sub([4]).wcs_world2pix(
            vis.frequency.data, 0
        )[0]
    ).astype("int")

    nrows, nbaselines, nvchan, nvpol = vis["vis"].data.shape
    nichan, nipol, _, _ = griddata["pixels"].data.shape

    fvist = numpy.nan_to_num(
        vis.visibility_acc.flagged_vis.reshape([nrows * nbaselines, nvchan, nvpol]).T
    )
    fwtt = numpy.nan_to_num(
        vis.visibility_acc.flagged_imaging_weight.reshape(
            [nrows * nbaselines, nvchan, nvpol]
        ).T
    )
    # Do this in place to avoid creating a new copy. Doing the conjugation outside the loop
    # reduces run time immensely
    ccf = numpy.conjugate(cf["pixels"].data)
    ccf = numpy.nan_to_num(ccf)
    _, _, _, _, _, gv, gu = ccf.shape
    du = gu // 2
    dv = gv // 2

    sumwt = numpy.zeros([nichan, nipol])

    gd = griddata["pixels"].data

    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        (
            pu_grid,
            pu_offset,
            pv_grid,
            pv_offset,
            pwc_grid,
            pwc_fraction,
        ) = convolution_mapping_visibility(vis, griddata, vchan, cf)
        for pol in range(nvpol):
            num_skipped = 0
            for row in range(nrows * nbaselines):
                subcf = ccf[
                    imchan, pol, pwc_grid[row], pv_offset[row], pu_offset[row], :, :
                ]
                # skipped over underflows
                if (
                    pv_grid[row] - dv < 0
                    or pv_grid[row] + dv >= gd.shape[2]
                    or pu_grid[row] - du < 0
                    or pu_grid[row] + du >= gd.shape[3]
                ):
                    num_skipped += 1
                    continue
                gd[
                    imchan,
                    pol,
                    (pv_grid[row] - dv) : (pv_grid[row] + dv),
                    (pu_grid[row] - du) : (pu_grid[row] + du),
                ] += (
                    subcf * fvist[pol, vchan, row] * fwtt[pol, vchan, row]
                )
                sumwt[imchan, pol] += fwtt[pol, vchan, row]
            if num_skipped > 0:
                log.warning(
                    "warning visibility_to_griddata gridding: skipped %d visbility",
                    num_skipped,
                )

    griddata["pixels"].data = numpy.nan_to_num(gd)
    return griddata, numpy.nan_to_num(sumwt)


def grid_visibility_weight_to_griddata(vis, griddata: GridData):
    """Grid Visibility weight onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :return: GridData
    """
    assert (
        vis.visibility_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )

    nchan, npol, ny, nx = griddata.griddata_acc.shape
    sumwt = numpy.zeros([nchan, npol])

    _, _, gv, gu = griddata["pixels"].data.shape
    vis_to_im = numpy.round(
        griddata.griddata_acc.griddata_wcs.sub([4]).wcs_world2pix(
            vis.frequency.data, 0
        )[0]
    ).astype("int")

    griddata["pixels"].data[...] = 0.0
    real_gd = numpy.real(griddata["pixels"].data)

    nrows, nbaselines, nvchan, nvpol = vis.vis.shape

    # Note that we are gridding with the imaging_weight, not the weight
    # Transpose to get row varying fastest
    fwtt = vis.visibility_acc.flagged_imaging_weight.reshape(
        [nrows * nbaselines, nvchan, nvpol]
    ).T

    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pv_grid = convolution_mapping_visibility(vis, griddata, vchan)
        num_skipped = 0
        for pol in range(nvpol):
            for row in range(nrows * nbaselines):
                # skipped over underflows
                if (
                    pv_grid[row] < 0
                    or pv_grid[row] >= real_gd.shape[2]
                    or pu_grid[row] < 0
                    or pu_grid[row] >= real_gd.shape[3]
                ):
                    num_skipped += 1
                    continue

                real_gd[imchan, pol, pv_grid[row], pu_grid[row]] += fwtt[
                    pol, vchan, row
                ]
                sumwt[imchan, pol] += fwtt[pol, vchan, row]
            if num_skipped > 0:
                log.warning(
                    "warning visibility_weight_to_griddata gridding: skipped %d visbility",
                    num_skipped,
                )

    griddata["pixels"].data = real_gd.astype("complex")

    return griddata, sumwt


def griddata_merge_weights(gd_list):
    """Merge weights into one grid

    :param gd_list:
    :return:
    """
    centre = len(gd_list) // 2
    gd = copy.deepcopy(gd_list[centre][0])
    sumwt = gd_list[centre][1]

    frequency = 0.0
    bandwidth = 0.0

    for i, g in enumerate(gd_list):
        if i != centre:
            gd["pixels"].data += g[0]["pixels"].data
            sumwt += g[1]
        frequency += g[0].griddata_acc.griddata_wcs.wcs.crval[3]
        bandwidth += g[0].griddata_acc.griddata_wcs.wcs.cdelt[3]

    gd.griddata_acc.griddata_wcs.wcs.cdelt[3] = bandwidth
    gd.griddata_acc.griddata_wcs.wcs.crval[3] = frequency / len(gd_list)
    return (gd, sumwt)


def griddata_visibility_reweight(vis, griddata, weighting="uniform", robustness=0.0):
    """Reweight visibility weight using the weights in griddata
       The fundamental equations are from
       https://casadocs.readthedocs.io/en/latest/notebooks/synthesis_imaging.html

    :param weighting:
    :param vis: visibility to be reweighted
    :param griddata: GridData holding gridded weights
    :return: Visibility with imaging_weights corrected
    """
    assert (
        vis.visibility_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )

    assert weighting in [
        "natural",
        "uniform",
        "robust",
    ], "Weighting {} not supported".format(weighting)

    real_gd = numpy.real(griddata["pixels"].data)

    vis_to_im = numpy.round(
        griddata.griddata_acc.griddata_wcs.sub([4]).wcs_world2pix(vis.frequency, 0)[0]
    ).astype("int")

    nrows, nbaselines, nvchan, nvpol = vis.vis.shape
    fimwtt = vis.visibility_acc.flagged_imaging_weight.reshape(
        [nrows * nbaselines, nvchan, nvpol]
    ).T
    fwtt = vis.visibility_acc.flagged_weight.reshape(
        [nrows * nbaselines, nvchan, nvpol]
    ).T

    if weighting == "natural":
        vis.imaging_weight.data[...] = vis.weight.data[...]
        return vis

    # All cases preserve the scaling such that a signal point in a grid cell
    # is unaffected. This means that the sensitivity may be calculated from
    # the sum of gridded weights

    sumlocwt = numpy.sum(real_gd**2)
    sumwt = numpy.sum(vis.visibility_acc.flagged_weight)

    if weighting == "robust":
        # Larger +ve robustness tends to natural weighting
        # Larger -ve robustness tends to uniform weighting
        f2 = (5.0 * numpy.power(10.0, -robustness)) ** 2 * sumwt / sumlocwt

    for pol in range(nvpol):
        for vchan in range(nvchan):
            imchan = vis_to_im[vchan]
            (
                pu_grid,
                pv_grid,
            ) = convolution_mapping_visibility(vis, griddata, vchan)

            # drop underflows
            v_overflows_mask = numpy.logical_or(
                pv_grid >= real_gd.shape[2], pv_grid < 0
            )
            u_overflows_mask = numpy.logical_or(
                pu_grid >= real_gd.shape[3], pu_grid < 0
            )
            uv_ingrid_mask = ~numpy.logical_or(u_overflows_mask, v_overflows_mask)

            if len(uv_ingrid_mask[uv_ingrid_mask == False]) > 0:
                log.warning(
                    "warning weighting gridding: skipped %d visbility",
                    len(uv_ingrid_mask[uv_ingrid_mask == False]),
                )

            gdwt_all = numpy.zeros((nrows * nbaselines,), dtype=real_gd.dtype)
            # mask in grid point and feed in gdwt
            gdwt_all[uv_ingrid_mask] = real_gd[
                imchan, pol, pv_grid[uv_ingrid_mask], pu_grid[uv_ingrid_mask]
            ]

            # visbility overflow grid
            fimwtt[pol, vchan, :][~uv_ingrid_mask] = 0.0

            # visbility in grid
            if weighting == "uniform":
                # This is the asymptotic version of the robust equation for infinite robustness
                fimwtt[pol, vchan, :][
                    numpy.logical_and(uv_ingrid_mask, gdwt_all > 0.0)
                ] *= (
                    fimwtt[pol, vchan, :][
                        numpy.logical_and(uv_ingrid_mask, gdwt_all > 0.0)
                    ]
                    / gdwt_all[gdwt_all > 0.0]
                )
            elif weighting == "robust":
                # Equation 3.15, 3.16 in Briggs thesis
                # https://casa.nrao.edu/Documents/Briggs-PhD.pdf
                # http://www.aoc.nrao.edu/dissertations/dbriggs/
                fimwtt[pol, vchan, :][
                    numpy.logical_and(uv_ingrid_mask, gdwt_all > 0.0)
                ] *= (
                    1
                    + f2
                    * fimwtt[pol, vchan, :][
                        numpy.logical_and(uv_ingrid_mask, gdwt_all > 0.0)
                    ]
                ) / (
                    1 + f2 * gdwt_all[gdwt_all > 0.0]
                )

            fimwtt[pol, vchan, :][
                numpy.logical_and(uv_ingrid_mask, gdwt_all <= 0.0)
            ] = 0.0

    vis.imaging_weight.data[...] = fimwtt.T.reshape([nrows, nbaselines, nvchan, nvpol])

    return vis


def degrid_visibility_from_griddata(vis, griddata, cf, **kwargs):
    """Degrid blockVisibility from a GridData
       Note: if parameter oversampling_synthesised_beam in advise_wide_field()
             has been set less than 2, some visibilities would be discarded.

    :param vis: visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    assert (
        vis.visibility_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )
    assert (
        cf.convolutionfunction_acc.polarisation_frame
        == griddata.griddata_acc.polarisation_frame
    )

    newvis = vis.copy(deep=True, zero=True)

    nchan, npol, nz, oversampling, _, support, _ = cf["pixels"].data.shape
    vis_to_im = numpy.round(
        griddata.griddata_acc.griddata_wcs.sub([4]).wcs_world2pix(
            vis.frequency.data, 0
        )[0]
    ).astype("int")

    nrows, nbaselines, nvchan, nvpol = vis.vis.shape
    fvist = numpy.zeros([nvpol, nvchan, nrows * nbaselines], dtype="complex")

    _, _, _, _, _, gv, gu = cf["pixels"].data.shape

    du = gu // 2
    dv = gv // 2

    gd = griddata["pixels"].data
    scf = cf["pixels"].data
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        (
            pu_grid,
            pu_offset,
            pv_grid,
            pv_offset,
            pwc_grid,
            pwc_fraction,
        ) = convolution_mapping_visibility(vis, griddata, vchan, cf)
        num_skipped = 0
        for pol in range(nvpol):
            for row in range(nrows * nbaselines):
                # skipped over underflows
                if (
                    pv_grid[row] - dv < 0
                    or pv_grid[row] + dv >= gd.shape[2]
                    or pu_grid[row] - du < 0
                    or pu_grid[row] + du >= gd.shape[3]
                ):
                    num_skipped += 1
                    continue

                subgrid = gd[
                    imchan,
                    pol,
                    (pv_grid[row] - dv) : (pv_grid[row] + dv),
                    (pu_grid[row] - du) : (pu_grid[row] + du),
                ]

                subcf = scf[
                    imchan, pol, pwc_grid[row], pv_offset[row], pu_offset[row], :, :
                ]
                fvist[pol, vchan, row] = numpy.einsum("ij,ij", subgrid, subcf)
            if num_skipped > 0:
                log.warning("warning gridding: skipped %d visbility", num_skipped)

    newvis["vis"].data[...] = fvist.T.reshape([nrows, nbaselines, nvchan, nvpol])

    return newvis


def fft_griddata_to_image(griddata, template, gcf=None):
    """FFT griddata after applying gcf

    If imaginary is true the data array is complex

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    # assert isinstance(griddata, GridData)

    ny, nx = griddata["pixels"].data.shape[-2], griddata["pixels"].data.shape[-1]

    if gcf is None:
        im_data = ifft(griddata["pixels"].data) * float(nx) * float(ny)
    else:
        im_data = (
            ifft(griddata["pixels"].data) * gcf["pixels"].data * float(nx) * float(ny)
        )

    return create_image_from_array(
        im_data, template.image_acc.wcs, griddata.griddata_acc.polarisation_frame
    )


def fft_image_to_griddata(im, griddata, gcf=None):
    """Fill griddata with transform of im

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    # chan, pol, z, u, v, w
    assert im.image_acc.polarisation_frame == griddata.griddata_acc.polarisation_frame

    if gcf is None:
        griddata["pixels"].data[...] = fft(im["pixels"].data)[...]
    else:
        griddata["pixels"].data[...] = fft(im["pixels"].data * gcf["pixels"].data)[...]

    return griddata
