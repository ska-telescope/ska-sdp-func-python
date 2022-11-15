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


@pytest.fixture(scope="module", name="base_params")
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


def test_phase_rotation_identity(base_params):
    """Check phaserotate_visibility consisitently gives good results"""
    vis = base_params["visibility"]
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
            base_params["phasecentre"],
            tangent=False,
        )
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)


def test_phase_rotation(base_params):
    """
    Check that phaserotate_visibility gives the same answer
    as offsetting phase centre "manually"
    """
    vis = base_params["visibility"]
    # Predict visibilities with new phase centre independently
    ha_diff = (
        -(base_params["compabsdirection"].ra - base_params["phasecentre"].ra)
        .to(u.rad)
        .value
    )
    vispred = create_visibility(
        base_params["lowcore"],
        base_params["times"] + ha_diff,
        base_params["frequency"],
        channel_bandwidth=base_params["channel_bandwidth"],
        phasecentre=base_params["compabsdirection"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
        times_are_ha=True,
    )

    # Should yield the same results as rotation
    rotatedvis = phaserotate_visibility(
        vis,
        newphasecentre=base_params["compabsdirection"],
        tangent=False,
    )
    assert_allclose(rotatedvis.vis, vispred.vis, rtol=3e-6)
    assert_allclose(
        rotatedvis.visibility_acc.uvw_lambda,
        vispred.visibility_acc.uvw_lambda,
        rtol=3e-6,
    )


def test_phase_rotation_inverse(base_params):
    """Check that using phase_rotate twice makes no difference"""
    vis = base_params["visibility"]
    there = SkyCoord(
        ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    original_vis = vis.vis
    original_uvw = vis.uvw
    rotatedvis = phaserotate_visibility(
        phaserotate_visibility(vis, there, tangent=False, inverse=False),
        base_params["phasecentre"],
        tangent=False,
        inverse=False,
    )
    assert_allclose(rotatedvis.uvw.data, original_uvw.data, rtol=1e-7)
    assert_allclose(rotatedvis["vis"].data, original_vis.data, rtol=1e-7)


def test_calculate_visibility_phasor(base_params):
    """Check calculate_visibility_phasor gives the correct phasor"""
    direction = base_params["phasecentre"]
    vis = base_params["visibility"]

    phasor = calculate_visibility_phasor(direction, vis)

    assert (phasor == 1).all()


def test_calculate_visibility_uvw_lambda(base_params):
    """Check calculate_visibility_uvw_lambda updates the uvw values"""
    vis = base_params["visibility"]

    updated_vis = calculate_visibility_uvw_lambda(vis)
    expected_uvw = numpy.einsum(
        "tbs,k->tbks",
        vis.uvw.data,
        vis.frequency.data / physical_constants.C_M_S,
    )

    assert vis != updated_vis
    assert expected_uvw == pytest.approx(updated_vis.visibility_acc.uvw_lambda)
