""" Unit tests for coordinate calculations

"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_func_python.visibility.visibility_geometry import (
    calculate_visibility_azel,
    calculate_visibility_hourangles,
    calculate_visibility_parallactic_angles,
    calculate_visibility_transit_time,
)


@pytest.fixture(scope="module", name="vis_geo_params")
def visibility_geometry_fixture():
    """Fixture for visbility_geometry.py unit tests"""
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(-21600, +21600, 3600.0)
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-65.0 * u.deg, frame="icrs", equinox="J2000"
    )
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
    bvis = create_visibility(
        lowcore,
        times,
        frequency,
        utc_time=Time(["2020-01-01T00:00:00"], format="isot", scale="utc"),
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
    )
    return bvis


def test_azel(vis_geo_params):
    """Unit test for the calculate_visibility_azel function"""
    azel = calculate_visibility_azel(vis_geo_params)
    numpy.testing.assert_array_almost_equal(azel[0][0].deg, 152.546993)
    numpy.testing.assert_array_almost_equal(azel[1][0].deg, 24.061762)


def test_hourangle(vis_geo_params):
    """Unit test for the calculate_visibility_hourangles function"""
    hr_angle = calculate_visibility_hourangles(vis_geo_params)
    numpy.testing.assert_array_almost_equal(hr_angle[0].deg, -89.989667)


def test_parallactic_angle(vis_geo_params):
    """Unit test for the calculate_visibility_parallactic_angles function"""
    p_angle = calculate_visibility_parallactic_angles(vis_geo_params)
    numpy.testing.assert_array_almost_equal(p_angle[0].deg, -102.050543)


def test_transit_time(vis_geo_params):
    """Unit test for the calculate_visibility_transit_time function"""
    transit_time = calculate_visibility_transit_time(vis_geo_params)
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895812)
