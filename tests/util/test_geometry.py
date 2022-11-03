# pylint: disable=invalid-name, too-many-arguments
# pylint: disable= missing-class-docstring, missing-function-docstring
# pylint: disable=import-error, no-name-in-module
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
    ha = calculate_hourangles(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(ha[0].deg, 36.881315)


def test_parallacticangles(geo_params):
    pa = calculate_parallactic_angles(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(pa[0].deg, 85.756057)


def test_transit_time(geo_params):
    transit_time = calculate_transit_time(
        geo_params["location"],
        geo_params["utc_time"],
        geo_params["phasecentre"],
    )
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895866)


def test_transit_time_below_horizon(geo_params):
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=+80.0 * u.deg, frame="icrs", equinox="J2000"
    )
    transit_time = calculate_transit_time(
        geo_params["location"], geo_params["utc_time"], phasecentre
    )
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895804)


def test_utc_to_ms_epoch(geo_params):
    ms_epoch = utc_to_ms_epoch(geo_params["utc_time"])
    numpy.testing.assert_array_almost_equal(ms_epoch, 5084553600.0)
