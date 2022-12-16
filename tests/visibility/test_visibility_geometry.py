"""
Unit tests for coordinate calculations
"""
import numpy

from ska_sdp_func_python.visibility.visibility_geometry import (
    calculate_visibility_azel,
    calculate_visibility_hourangles,
    calculate_visibility_parallactic_angles,
    calculate_visibility_transit_time,
)


def test_azel(visibility):
    """Unit test for the calculate_visibility_azel function"""
    azel = calculate_visibility_azel(visibility)
    numpy.testing.assert_array_almost_equal(azel[0][0].deg, -179.8739982)
    numpy.testing.assert_array_almost_equal(azel[1][0].deg, 81.82924412)


def test_hourangle(visibility):
    """Unit test for the calculate_visibility_hourangles function"""
    hr_angle = calculate_visibility_hourangles(visibility)
    numpy.testing.assert_array_almost_equal(hr_angle[0].deg, 0.0207416)


def test_parallactic_angle(visibility):
    """Unit test for the calculate_visibility_parallactic_angles function"""
    p_angle = calculate_visibility_parallactic_angles(visibility)
    numpy.testing.assert_array_almost_equal(p_angle[0].deg, 0.15243407)


def test_transit_time(visibility):
    """Unit test for the calculate_visibility_transit_time function"""
    transit_time = calculate_visibility_transit_time(visibility)
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 51545.890737)
