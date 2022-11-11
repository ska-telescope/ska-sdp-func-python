# pylint: skip-file
""" Unit tests for visibility base


"""
import astropy.units as u
import numpy
import pytest
pytest.skip(
    allow_module_level=True,
    reason="not able importing ska-sdp-func in dft_skycomponent_visibility",
)
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_create import create_visibility
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility
from ska_sdp_func_python.visibility.base import phaserotate_visibility



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
    parameters = {
        "lowcore": lowcore,
        "times": times,
        "frequency": frequency,
        "channel_bandwidth": channel_bandwidth,
        "phasecentre": phasecentre,
        "compabsdirection": compabsdirection,
        "comp": comp,
    }
    return parameters


@pytest.mark.skip(reason="import issues with dft_skycomponent")
def test_phase_rotation_identity(result_base):
    vis = create_visibility(
        result_base["lowcore"],
        result_base["times"],
        result_base["frequency"],
        channel_bandwidth=result_base["channel_bandwidth"],
        phasecentre=result_base["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    vismodel = dft_skycomponent_visibility(vis, result_base["comp"])
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
        original_vis = vismodel.vis
        original_uvw = vismodel.uvw
        rotatedvis = phaserotate_visibility(
            phaserotate_visibility(vismodel, newphasecentre, tangent=False),
            result_base["phasecentre"],
            tangent=False,
        )
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)


@pytest.mark.skip(reason="import issues with dft_skycomponent")
def test_phase_rotation(result_base):
    vis = create_visibility(
        result_base["lowcore"],
        result_base["times"],
        result_base["frequency"],
        channel_bandwidth=result_base["channel_bandwidth"],
        phasecentre=result_base["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
        times_are_ha=True,
    )
    vismodel = dft_skycomponent_visibility(vis, result_base["comp"])
    # Predict visibilities with new phase centre independently
    ha_diff = (
        -(
            result_base["compabsdirection"].ra
            - result_base["phasecentre"].ra
        )
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
    vismodel2 = dft_skycomponent_visibility(vispred, result_base["comp"])

    # Should yield the same results as rotation
    rotatedvis = phaserotate_visibility(
        vismodel,
        newphasecentre=result_base["compabsdirection"],
        tangent=False,
    )
    assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=3e-6)
    assert_allclose(
        rotatedvis.visibility_acc.uvw_lambda,
        vismodel2.visibility_acc.uvw_lambda,
        rtol=3e-6,
    )


@pytest.mark.skip(reason="import issues with dft_skycomponent")
def test_phase_rotation_inverse(result_base):
    vis = create_visibility(
        result_base["lowcore"],
        result_base["times"],
        result_base["frequency"],
        channel_bandwidth=result_base["channel_bandwidth"],
        phasecentre=result_base["phasecentre"],
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    vismodel = dft_skycomponent_visibility(vis, result_base["comp"])
    there = SkyCoord(
        ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    # Phase rotating back should not make a difference
    original_vis = vismodel.vis
    original_uvw = vismodel.uvw
    rotatedvis = phaserotate_visibility(
        phaserotate_visibility(vismodel, there, tangent=False, inverse=False),
        result_base["phasecentre"],
        tangent=False,
        inverse=False,
    )
    assert_allclose(rotatedvis.uvw.data, original_uvw.data, rtol=1e-7)
    assert_allclose(rotatedvis["vis"].data, original_vis.data, rtol=1e-7)