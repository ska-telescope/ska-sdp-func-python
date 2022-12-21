"""
Unit tests for visibility base
"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from ska_sdp_datamodels import physical_constants
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.base import (
    calculate_visibility_phasor,
    calculate_visibility_uvw_lambda,
    phaserotate_visibility,
)


@pytest.mark.parametrize(
    "new_phase_centre",
    [
        SkyCoord(182, -35, unit=u.deg),
        SkyCoord(182, -30, unit=u.deg),
        SkyCoord(177, -30, unit=u.deg),
        SkyCoord(176, -35, unit=u.deg),
        SkyCoord(216, -35, unit=u.deg),
        SkyCoord(180, -70, unit=u.deg),
    ],
)
def test_phase_rotation_identity(new_phase_centre, visibility):
    """
    Check phaserotate_visibility consistently gives good results.
    If rotated twice, doesn't make a difference
    """
    # Phase rotating back should not make a difference
    original_vis = visibility.vis
    original_uvw = visibility.uvw
    rotated_vis = phaserotate_visibility(
        phaserotate_visibility(
            visibility, new_phase_centre, tangent=False, inverse=False
        ),
        visibility.phasecentre,
        tangent=False,
        inverse=False,
    )
    assert_allclose(rotated_vis.uvw, original_uvw, atol=1e-7)
    assert_allclose(rotated_vis.vis, original_vis, atol=1e-7)


def test_phase_rotation(visibility, comp_direction):
    """
    Check that phaserotate_visibility gives the same answer
    as offsetting phase centre "manually"
    """
    # Predict visibilities with new phase centre independently
    ha_diff = -(comp_direction.ra - visibility.phasecentre.ra).to(u.rad).value
    # same as for dft_vis time; we need this in HA
    times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 300.0, 2)
    vis_manual_offset = create_visibility(
        visibility.configuration,
        times + ha_diff,
        visibility.frequency.data,
        channel_bandwidth=visibility.channel_bandwidth.data,
        phasecentre=comp_direction,
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
        times_are_ha=True,
    )

    # Should yield the same results as rotation
    rotated_vis = phaserotate_visibility(
        visibility,
        newphasecentre=comp_direction,
        tangent=False,
    )
    assert_allclose(rotated_vis.vis, vis_manual_offset.vis, rtol=3e-6)
    assert_allclose(
        rotated_vis.visibility_acc.uvw_lambda,
        vis_manual_offset.visibility_acc.uvw_lambda,
        rtol=3e-6,
    )


def test_calculate_visibility_phasor(visibility):
    """Check calculate_visibility_phasor gives the correct phasor"""
    phasor = calculate_visibility_phasor(visibility.phasecentre, visibility)

    assert (phasor == 1).all()


def test_calculate_visibility_uvw_lambda(visibility):
    """Check calculate_visibility_uvw_lambda updates the uvw values"""
    updated_vis = calculate_visibility_uvw_lambda(visibility)
    expected_uvw = numpy.einsum(
        "tbs,k->tbks",
        visibility.uvw.data,
        visibility.frequency.data / physical_constants.C_M_S,
    )

    assert visibility != updated_vis
    assert expected_uvw == pytest.approx(updated_vis.visibility_acc.uvw_lambda)
