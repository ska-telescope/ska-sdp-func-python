""" Unit tests for coordinate calculations

"""
import astropy.units as u
import numpy
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ska_sdp_func_python.util.geometry import (
    calculate_azel,
    calculate_hourangles,
    calculate_parallactic_angles,
    calculate_transit_time,
    utc_to_ms_epoch,
)


@pytest.fixture(scope="module", name="geo_params")
def geometry_fixture():
    """Fixture for the geometry.py unit tests"""
    location = EarthLocation(
        lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
    )
    times = (numpy.pi / 43200.0) * numpy.arange(-43200, +43200, 3600.0)
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    utc_time = Time(["2020-01-01T00:00:00"], format="isot", scale="utc")

    parameters = {
        "location": location,
        "times": times,
        "phasecentre": phasecentre,
        "utc_time": utc_time,
    }
    return parameters


def test_azel(geo_params):
    """Check calculate_azel returns the correct values"""
    utc_times = Time(
        numpy.arange(0.0, 1.0, 0.1) + geo_params["utc_time"].mjd,
        format="mjd",
        scale="utc",
    )
    azel = calculate_azel(
        geo_params["location"], utc_times, geo_params["phasecentre"]
    )
    numpy.testing.assert_array_almost_equal(azel[0][0].deg, -113.964241)
    numpy.testing.assert_array_almost_equal(azel[1][0].deg, 57.715754)
    numpy.testing.assert_array_almost_equal(azel[0][-1].deg, -171.470433)
    numpy.testing.assert_array_almost_equal(azel[1][-1].deg, 81.617363)


def test_hourangles(geo_params):
    """Check calculate_hourangles returns the correct values"""
    h_angles = calculate_hourangles(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(h_angles[0].deg, 36.881315)


def test_parallacticangles(geo_params):
    """Check calculate_parallactic_angles returns the correct values"""
    p_angles = calculate_parallactic_angles(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(p_angles[0].deg, 85.756057)


def test_transit_time(geo_params):
    """Check calculate_transit_time returns the correct values"""
    transit_time = calculate_transit_time(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895866)


def test_transit_time_below_horizon(geo_params):
    """Check calculate_transit_times returns correct values
        below the horizon"""
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=+80.0 * u.deg, frame="icrs", equinox="J2000"
    )
    transit_time = calculate_transit_time(
        geo_params["location"], geo_params["utc_time"], phasecentre
    )
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895804)


def test_utc_to_ms_epoch(geo_params):
    """Check calculate_utc_to_ms_epoch returns the correct values"""
    ms_epoch = utc_to_ms_epoch(geo_params["utc_time"])
    numpy.testing.assert_array_almost_equal(ms_epoch, 5084553600.0)
