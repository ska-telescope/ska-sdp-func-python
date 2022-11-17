"""
Pytest fixtures
"""
import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility


@pytest.fixture(scope="package", name="visibility")
def vis_fixture():
    """
    Visibility fixture
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
    vis_pol = PolarisationFrame("stokesI")

    low = create_named_configuration("LOW", rmax=300.0)
    phase_centre = SkyCoord(
        ra=+30.0 * units.deg, dec=-60.0 * units.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        low,
        times,
        frequency,
        phase_centre,
        channel_bandwidth=channelwidth,
        weight=1.0,
        polarisation_frame=vis_pol,
        zerow=False,
        times_are_ha=True,
    )
    return vis
