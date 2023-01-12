"""
Unit tests for visibility weighting
"""
import numpy

from ska_sdp_func_python.image.deconvolution import fit_psf
from ska_sdp_func_python.imaging.imaging import invert_visibility
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    taper_visibility_tukey,
    weight_visibility,
)


def test_tapering_gaussian(visibility, model):
    """
    Apply a Gaussian taper to the visibility and check to see if
    the PSF size is close
    """
    size_required = 0.020
    vis = visibility.copy(deep=True)
    vis = weight_visibility(
        vis,
        model,
        weighting="uniform",
    )
    vis = taper_visibility_gaussian(vis, beam=size_required)
    psf, _ = invert_visibility(
        vis,
        model,
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)

    assert (
        numpy.abs(fit["bmaj"] - 1.279952050682638) < 1
    ), f"Fit should be {1.279952050682638}, actually is {fit['bmaj']}"


def test_tapering_tukey(visibility, model):
    """
    Apply a Tukey window taper and output the psf and FT of the PSF.
    No quantitative check.
    """
    vis = visibility.copy(deep=True)
    vis = weight_visibility(
        vis,
        model,
        weighting="uniform",
    )
    vis = taper_visibility_tukey(vis, tukey=0.1)
    psf, _ = invert_visibility(
        vis,
        model,
        dopsf=True,
        context="2d",
    )
    fit = fit_psf(psf)
    assert (
        numpy.abs(fit["bmaj"] - 0.14492670913355402) < 1.0
    ), f"Fit should be {0.14492670913355402}, actually is {fit['bmaj']}"
