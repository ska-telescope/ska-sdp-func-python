"""
Unit tests for coordinate support
"""
import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from ska_sdp_func_python.util.coordinate_support import (
    azel_to_hadec,
    baselines,
    hadec_to_azel,
    simulate_point,
    skycoord_to_lmn,
    uvw_to_xyz,
    xyz_at_latitude,
    xyz_to_uvw,
)


@pytest.mark.parametrize(
    "x, y, z, lat, expected_xyz",
    [
        # At the north pole the zenith is the celestial north
        (1, 0, 0, 90, [1, 0, 0]),
        (0, 1, 0, 90, [0, 1, 0]),
        (0, 0, 1, 90, [0, 0, 1]),
        # At the equator the zenith is negative Y
        (1, 0, 0, 0, [1, 0, 0]),
        (0, 1, 0, 0, [0, 0, 1]),
        (0, 0, 1, 0, [0, -1, 0]),
        # At the south pole we have flipped Y and Z
        (1, 0, 0, -90, [1, 0, 0]),
        (0, 1, 0, -90, [0, -1, 0]),
        (0, 0, 1, -90, [0, 0, -1]),
    ],
)
def test_xyz_at_latitude(x, y, z, lat, expected_xyz):
    """
    xyz_at_latitude works correctly for different use-cases
    (see comments in the parametrize section of the test)
    """
    res = xyz_at_latitude(numpy.array([x, y, z]), numpy.radians(lat))
    assert_allclose(numpy.linalg.norm(res), numpy.linalg.norm([x, y, z]))
    assert_allclose(res, expected_xyz, atol=1e-15)


@pytest.mark.parametrize(
    "x, y, z, ha, dec, expected_uvw",
    [
        # For ha=0,dec=90, we should have UVW=XYZ
        (0, 0, 1, 0, 90, [0, 0, 1]),
        (0, 1, 0, 0, 90, [0, 1, 0]),
        (1, 0, 0, 0, 90, [1, 0, 0]),
        # For dec=90, we always have Z=W
        (0, 0, 1, -90, 90, [0, 0, 1]),
        (0, 0, 1, 90, 90, [0, 0, 1]),
        # When W is on the local meridian (hour angle 0),
        # U points east (positive X)
        (1, 0, 0, 0, 0, [1, 0, 0]),
        (1, 0, 0, 0, 30, [1, 0, 0]),
        (1, 0, 0, 0, -20, [1, 0, 0]),
        (1, 0, 0, 0, -90, [1, 0, 0]),
        # When the direction of observation is at zero declination,
        # an hour-angle of -6 hours (-90 degrees) makes W point to
        # the east (positive X).
        (1, 0, 0, -90, 0, [0, 0, 1]),
        (1, 0, 0, 90, 0, [0, 0, -1]),
    ],
)
def test_xyz_to_uvw(x, y, z, ha, dec, expected_uvw):
    """
    xyz_to_uvw determines uvw correctly for various use cases.
    (see comments in the parametrize section of the test)

    Derived from http://casa.nrao.edu/Memos/CoordConvention.pdf
    """
    res = xyz_to_uvw(
        numpy.array([x, y, z]), numpy.radians(ha), numpy.radians(dec)
    )
    assert_allclose(numpy.linalg.norm(res), numpy.linalg.norm([x, y, z]))
    assert_allclose(
        uvw_to_xyz(res, numpy.radians(ha), numpy.radians(dec)),
        [x, y, z],
    )
    assert_allclose(res, expected_uvw, atol=1e-15)


def test_baselines():
    """
    baselines correctly retuns the baselines
    in uvw from antenna locations in uvw

    There should be exactly npixel*(npixel+1)/2 baselines
    """
    for i in range(10):
        ants_uvw = numpy.repeat(numpy.array(range(10 + i)), 3)
        num_ants = len(ants_uvw)
        result = baselines(ants_uvw)

        assert len(result) == num_ants * (num_ants - 1) // 2


def test_simulate_point():
    """
    simulate_point generates unit visibilities at
    right l,m coordinates for different use-cases
    """
    # Prepare a synthetic layout
    uvw = numpy.concatenate(
        numpy.concatenate(numpy.transpose(numpy.mgrid[-3:4, -3:4, 0:1]))
    )
    bls = baselines(uvw)

    # Should have positive amplitude for the middle of the picture
    vis = simulate_point(bls, 0, 0)
    assert_allclose(vis, numpy.ones(len(vis)))

    # For the half-way point the result is either -1 or 1
    # depending on whether the baseline length is even
    bl_even = 1 - 2 * (numpy.sum(bls, axis=1) % 2)
    vis = simulate_point(bls, 0.5, 0.5)
    assert_allclose(vis, bl_even)
    vis = simulate_point(bls, -0.5, 0.5)
    assert_allclose(vis, bl_even)
    vis = simulate_point(bls, 0.5, -0.5)
    assert_allclose(vis, bl_even)
    vis = simulate_point(bls, -0.5, -0.5)
    assert_allclose(vis, bl_even)


def test_skycoord_to_lmn():
    """
    l, m, n coordinates correctly determined from
    astropy.SkyCoord object
    """
    center = SkyCoord(ra=0, dec=0, unit=u.deg)
    north = SkyCoord(ra=0, dec=90, unit=u.deg)
    south = SkyCoord(ra=0, dec=-90, unit=u.deg)
    east = SkyCoord(ra=90, dec=0, unit=u.deg)
    west = SkyCoord(ra=-90, dec=0, unit=u.deg)

    assert_allclose(skycoord_to_lmn(center, center), (0, 0, 0))
    assert_allclose(skycoord_to_lmn(north, center), (0, 1, -1))
    assert_allclose(skycoord_to_lmn(south, center), (0, -1, -1))
    assert_allclose(skycoord_to_lmn(south, north), (0, 0, -2), atol=1e-14)
    assert_allclose(skycoord_to_lmn(east, center), (1, 0, -1))
    assert_allclose(skycoord_to_lmn(west, center), (-1, 0, -1))
    assert_allclose(skycoord_to_lmn(center, west), (1, 0, -1))
    assert_allclose(skycoord_to_lmn(north, west), (0, 1, -1), atol=1e-14)
    assert_allclose(skycoord_to_lmn(south, west), (0, -1, -1), atol=1e-14)
    assert_allclose(skycoord_to_lmn(north, east), (0, 1, -1), atol=1e-14)
    assert_allclose(skycoord_to_lmn(south, east), (0, -1, -1), atol=1e-14)


SKY_COORD_1 = SkyCoord(17, 35, unit=u.deg)


@pytest.mark.parametrize("source_position, phase_center, new_phase_center", [
    (SKY_COORD_1, SKY_COORD_1, SKY_COORD_1),
    (SKY_COORD_1, SkyCoord(12, 30, unit=u.deg), SkyCoord(51, 35, unit=u.deg)),
    (SkyCoord(11, 35, unit=u.deg), SKY_COORD_1, SKY_COORD_1)
])
def test_phase_rotate(source_position, phase_center, new_phase_center):
    """
    This test combines various functions previously tested to see that they
    correctly work together to rotate phases.

    They don't test an individual function.
    I'm not sure this test is needed, but keeping it for refrence.
    """

    uvw = numpy.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # Rotate UVW
    xyz = uvw_to_xyz(uvw, -phase_center.ra.rad, phase_center.dec.rad)
    uvw_rotated = xyz_to_uvw(
        xyz, -new_phase_center.ra.rad, new_phase_center.dec.rad
    )

    # Determine phasor
    l_p, m_p, _ = skycoord_to_lmn(phase_center, new_phase_center)
    phasor = simulate_point(uvw_rotated, l_p, m_p)

    # Simulate visibility at old and new phase centre
    l, m, _ = skycoord_to_lmn(source_position, phase_center)
    vis = simulate_point(uvw, l, m)
    l_r, m_r, _ = skycoord_to_lmn(source_position, new_phase_center)
    vis_rotated = simulate_point(uvw_rotated, l_r, m_r)

    # Difference should be given by phasor
    assert_allclose(vis * phasor, vis_rotated, atol=1e-10)


@pytest.mark.parametrize(
    "position",
    [
        SkyCoord(17, -35, unit=u.deg),
        SkyCoord(17, -30, unit=u.deg),
        SkyCoord(12, -89.909, unit=u.deg),
        SkyCoord(11, -35, unit=u.deg),
        SkyCoord(51, -35, unit=u.deg),
        SkyCoord(15, -70, unit=u.deg),
    ],
)
def test_azel_hadec(position):
    """
    First we convert given HA, DEC to azimuth, elevation,
    then back using azel_to_hadec.

    Original and converted HA, DEC match.
    """
    ha, dec = position.ra.rad, position.dec.rad

    az, el = hadec_to_azel(ha, dec, latitude=-numpy.pi / 4.0)
    har, decr = azel_to_hadec(az, el, latitude=-numpy.pi / 4.0)

    assert_allclose(ha, har)
    assert_allclose(dec, decr)
