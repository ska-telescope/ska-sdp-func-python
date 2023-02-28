"""
Functions for calculating the geometry of a Visibility.
"""

__all__ = [
    "calculate_visibility_azel",
    "calculate_visibility_hourangles",
    "calculate_visibility_parallactic_angles",
    "calculate_visibility_transit_time",
]

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity

from ska_sdp_func_python.util.geometry import (
    calculate_azel,
    calculate_hourangles,
    calculate_parallactic_angles,
    calculate_transit_time,
)


def get_direction_time_location(bvis):
    """
    Get the direction, time and location from Visibility data.
    This function is used by the other calculation functions.

    :param bvis: Visibility
    return: Location, UTC time, direction (SkyCoord)
    """
    location = bvis.configuration.location
    if location is None:
        location = EarthLocation(
            x=Quantity(bvis.configuration.antxyz[0]),
            y=Quantity(bvis.configuration.antxyz[1]),
            z=Quantity(bvis.configuration.antxyz[2]),
        )

    utc_time = Time(bvis.time / 86400.0, format="mjd", scale="utc")
    direction = bvis.phasecentre

    return location, utc_time, direction


def calculate_visibility_hourangles(bvis):
    """Return hour angles for location, utc_time, and direction.

    :param bvis: Visibility
    :return: Hour angles
    """

    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_hourangles(location, utc_time, direction)


def calculate_visibility_parallactic_angles(bvis):
    """Return parallactic angles for location, utc_time, and direction.

    :param bvis: Visibility
    :return: Angle
    """

    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_parallactic_angles(location, utc_time, direction)


def calculate_visibility_transit_time(bvis):
    """Find the UTC time of the nearest transit.

    :param bvis: Visibility
    :return: Transit time
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_transit_time(location, utc_time[0], direction)


def calculate_visibility_azel(bvis):
    """Return az el for a location, utc_time, and direction.

    :param bvis: Visibility
    :return: Az, El coordinates
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_azel(location, utc_time, direction)
