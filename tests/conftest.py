"""
Pytest fixtures
"""
import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

N_CHAN = 6


@pytest.fixture(scope="package", name="phase_centre")
def phase_center():
    """Phase Centre fixture"""
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return phase_centre


@pytest.fixture(scope="package", name="comp_direction")
def com_direction():
    """
    Component Absolute direction fixture
    """
    # The phase centre (given by phase_centre fixture)
    # is absolute and the component
    # is specified relative (for now).
    # This means that the component should end up at
    # the position phase_centre+comp_redirection
    comp_abs_direction = SkyCoord(
        ra=+181.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return comp_abs_direction


@pytest.fixture(scope="package", name="comp_dft")
def component_dft_fixture(phase_centre, comp_direction):
    """
    SkyComponent list fixture to be used with DFT tests
    """
    n_comp = 20
    phase_centre_offset = phase_centre.skyoffset_frame()
    comp_rel_direction = comp_direction.transform_to(phase_centre_offset)

    flux = numpy.array(N_CHAN * [100.0, 20.0, -10.0, 1.0]).reshape([N_CHAN, 4])
    frequency = numpy.linspace(1.0e8, 1.1e8, N_CHAN)

    comp = n_comp * [
        SkyComponent(
            direction=comp_rel_direction,
            frequency=frequency,
            flux=flux,
        )
    ]
    return comp


@pytest.fixture(scope="package", name="visibility")
def vis_fixture(phase_centre):
    """
    Visibility fixture
    """
    n_times = 2
    low_core = create_named_configuration("LOW")
    times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 300.0, n_times)
    frequency = numpy.linspace(1.0e8, 1.1e8, N_CHAN)
    channel_bandwidth = numpy.array(N_CHAN * [1e7 / N_CHAN])

    vis = create_visibility(
        low_core,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phase_centre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("linear"),
    )
    return vis


# ----------------------- Deconv ------------------------
# TODO: review
# these are needed for deconvolution test only, for now;
# move there if that's the only place that needs them
@pytest.fixture(scope="package", name="phase_centre_30")
def phase_center_fixture_2():
    """
    Phase Centre fixture for deconvolution test
    """
    phase_centre = SkyCoord(
        ra=+30.0 * units.deg,
        dec=-60.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return phase_centre


@pytest.fixture(scope="package", name="comp_direction_30")
def com_direction_fixture_2():
    """
    Component direction fixture for deconvolution test
    """
    comp_abs_direction = SkyCoord(
        ra=+30.0 * units.deg,
        dec=-61.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    return comp_abs_direction


@pytest.fixture(scope="package", name="visibility_deconv")
def vis_deconv_fixture(phase_centre_30):
    """
    Visibility fixture for deconvolution tests
    """
    ntimes = 3

    # Choose the interval so that the maximum change in w is smallish
    integration_time = numpy.pi * (24 / (12 * 60))
    times = numpy.linspace(
        -integration_time * (ntimes // 2),
        integration_time * (ntimes // 2),
        ntimes,
    )

    frequency = numpy.array([1.0e8])
    channelwidth = numpy.array([4e7])

    low = create_named_configuration("LOW", rmax=300.0)
    vis = create_visibility(
        low,
        times,
        frequency,
        phase_centre_30,
        channel_bandwidth=channelwidth,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True,
        times_are_ha=True,
    )
    return vis
