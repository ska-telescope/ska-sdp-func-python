# pylint: disable=missing-function-docstring
# pylint: disable=import-error
""" Functions for calculating geometry of a Visibility

"""

__all__ = [
    "calculate_visibility_transit_time",
    "calculate_visibility_hourangles",
    "calculate_visibility_parallactic_angles",
    "calculate_visibility_azel",
]

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity

from ska_sdp_func_python.util.geometry import (
    calculate_azel,
    calculate_hourangles,
    calculate_parallactic_angles,
    calculate_transit_time,
)


def get_direction_time_location(bvis):
    location = bvis.configuration.location
    if location is None:
        location = EarthLocation(
            x=Quantity(bvis.configuration.antxyz[0]),
            y=Quantity(bvis.configuration.antxyz[1]),
            z=Quantity(bvis.configuration.antxyz[2]),
        )

    utc_time = Time(bvis.time / 86400.0, format="mjd", scale="utc")
    direction = bvis.phasecentre
    # assert isinstance(bvis, Visibility), bvis
    assert isinstance(location, EarthLocation), location
    assert isinstance(utc_time, Time), utc_time
    assert isinstance(direction, SkyCoord), direction
    return location, utc_time, direction


def calculate_visibility_hourangles(bvis):
    """Return hour angles for location, utc_time, and direction

    :param bvis:
    :return:
    """

    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_hourangles(location, utc_time, direction)


def calculate_visibility_parallactic_angles(bvis):
    """Return parallactic angles for location, utc_time, and direction

    :param bvis:
    :return:
    """

    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_parallactic_angles(location, utc_time, direction)


def calculate_visibility_transit_time(bvis, fraction_day=1e-10):
    """Find the UTC time of the nearest transit

    :param fraction_day:
    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_transit_time(
        location, utc_time[0], direction, fraction_day=fraction_day
    )


def calculate_visibility_azel(bvis):
    """Return az el for a location, utc_time, and direction

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_azel(location, utc_time, direction)
