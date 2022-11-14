# pylint: skip-file
# flake8: noqa
""" Unit tests for visibility base


"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from ska_sdp_datamodels import physical_constants
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.base import (
    calculate_visibility_phasor,
    calculate_visibility_uvw_lambda,
    phaserotate_visibility,
)


@pytest.fixture(scope="module", name="result_base")
def visibility_operations_fixture():
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
    # Define the component and give it some spectral behaviour
    f = numpy.array([100.0, 20.0, -10.0, 1.0])
    flux = numpy.array([f, 0.8 * f, 0.6 * f])

    # The phase centre is absolute and component is specified relative
    # This means that the component should end up at the position
    # phasecentre+compredirection
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    pcof = phasecentre.skyoffset_frame()
    compreldirection = compabsdirection.transform_to(pcof)
    comp = SkyComponent(
        direction=compreldirection,
        frequency=frequency,
        flux=flux,
    )
    vis = create_visibility(
        lowcore,
        times,
        frequency,
        phasecentre,
        channel_bandwidth,
        1.0,
        PolarisationFrame("stokesIQUV"),
    )
    parameters = {
        "lowcore": lowcore,
        "times": times,
        "frequency": frequency,
        "channel_bandwidth": channel_bandwidth,
        "phasecentre": phasecentre,
        "compabsdirection": compabsdirection,
        "comp": comp,
        "visibility": vis,
    }
    return parameters


def test_phase_rotation_identity(result_base):
    """Check phaserotate_visibility consisitently gives good results"""
    vis = result_base["visibility"]
    newphasecenters = [
        SkyCoord(182, -35, unit=u.deg),
        SkyCoord(182, -30, unit=u.deg),
        SkyCoord(177, -30, unit=u.deg),
        SkyCoord(176, -35, unit=u.deg),
        SkyCoord(216, -35, unit=u.deg),
        SkyCoord(180, -70, unit=u.deg),
    ]
    for newphasecentre in newphasecenters:
        # Phase rotating back should not make a difference
        original_vis = vis.vis
        original_uvw = vis.uvw
        rotatedvis = phaserotate_visibility(
            phaserotate_visibility(vis, newphasecentre, tangent=False),
            result_base["phasecentre"],
            tangent=False,
        )
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)


def test_phase_rotation(result_base):
    """Check that phaserotate_visibility gives the same answer as offsetting phase centre "manually" """
    vis = result_base["visibility"]
    # Predict visibilities with new phase centre independently
    ha_diff = (
        -(result_base["compabsdirection"].ra - result_base["phasecentre"].ra)
        .to(u.rad)
        .value
    )
    vispred = create_visibility(
        result_base["lowcore"],
        result_base["times"] + ha_diff,
        result_base["frequency"],
        channel_bandwidth=result_base["channel_bandwidth"],
        phasecentre=result_base["compabsdirection"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
        times_are_ha=True,
    )

    # Should yield the same results as rotation
    rotatedvis = phaserotate_visibility(
        vis,
        newphasecentre=result_base["compabsdirection"],
        tangent=False,
    )
    assert_allclose(rotatedvis.vis, vispred.vis, rtol=3e-6)
    assert_allclose(
        rotatedvis.visibility_acc.uvw_lambda,
        vispred.visibility_acc.uvw_lambda,
        rtol=3e-6,
    )


def test_phase_rotation_inverse(result_base):
    """Check that using phase_rotate twice makes no difference"""
    vis = result_base["visibility"]
    there = SkyCoord(
        ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    original_vis = vis.vis
    original_uvw = vis.uvw
    rotatedvis = phaserotate_visibility(
        phaserotate_visibility(vis, there, tangent=False, inverse=False),
        result_base["phasecentre"],
        tangent=False,
        inverse=False,
    )
    assert_allclose(rotatedvis.uvw.data, original_uvw.data, rtol=1e-7)
    assert_allclose(rotatedvis["vis"].data, original_vis.data, rtol=1e-7)


def test_calculate_visibility_phasor(result_base):
    """Check calculate_visibility_phasor gives the correct phasor"""
    direction = result_base["phasecentre"]
    vis = result_base["visibility"]

    phasor = calculate_visibility_phasor(direction, vis)

    assert (phasor == 1).all()


def test_calculate_visibility_uvw_lambda(result_base):
    """Check calculate_visibility_uvw_lambda updates the uvw values"""
    vis = result_base["visibility"]

    updated_vis = calculate_visibility_uvw_lambda(vis)
    expected_uvw = numpy.einsum(
        "tbs,k->tbks",
        vis.uvw.data,
        vis.frequency.data / physical_constants.C_M_S,
    )

    assert vis != updated_vis
    assert expected_uvw == pytest.approx(updated_vis.visibility_acc.uvw_lambda)
