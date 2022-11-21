"""
Base simple visibility operations, placed here to avoid circular dependencies.
"""

__all__ = [
    "calculate_visibility_phasor",
    "calculate_visibility_uvw_lambda",
    "phaserotate_visibility",
]

import logging

import numpy
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels import physical_constants
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.util.coordinate_support import (
    skycoord_to_lmn,
    uvw_to_xyz,
    xyz_to_uvw,
)

log = logging.getLogger("func-python-logger")


def calculate_visibility_phasor(direction, vis):
    """Calculate the phasor for a component for a Visibility.

    :param direction: Direction (SkyCoords)
    :param vis: Visibility
    :return: Phasor (numpy.array)
    """
    # assert isinstance(vis, Visibility)
    ntimes, nbaseline, nchan, npol = vis["vis"].data.shape
    l, m, _ = skycoord_to_lmn(direction, vis.phasecentre)
    s = numpy.array([l, m, numpy.sqrt(1 - l**2 - m**2) - 1.0])

    phasor = numpy.ones([ntimes, nbaseline, nchan, npol], dtype="complex")
    phasor[...] = numpy.exp(
        -2j
        * numpy.pi
        * numpy.einsum("tbfs,s->tbf", vis.visibility_acc.uvw_lambda.data, s)
    )[..., numpy.newaxis]
    return phasor


def calculate_visibility_uvw_lambda(vis):
    """Recalculate the uvw_lambda values.

    :param vis: Visibility
    :return: Visibility with updated uvw_lambda
    """
    k = vis.frequency.data / physical_constants.C_M_S
    uvw_lambda = numpy.einsum("tbs,k->tbks", vis.uvw.data, k)
    vis.visibility_acc.uvw_lambda = uvw_lambda
    return vis


def phaserotate_visibility(
    vis: Visibility, newphasecentre: SkyCoord, tangent=True, inverse=False
) -> Visibility:
    """Phase rotate from the current phase centre to a new phase centre.

    If tangent is False the uvw are recomputed and the
    visibility phasecentre is updated. Otherwise, only the
    visibility phases are adjusted.

    :param vis: Visibility to be rotated
    :param newphasecentre: SkyCoord of new phasecentre
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :return: Visibility
    """
    _, _, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)

    # No significant change?
    if numpy.abs(n) < 1e-15:
        return vis

    # Make a new copy
    newvis = vis.copy(deep=True)

    phasor = calculate_visibility_phasor(newphasecentre, newvis)
    assert vis["vis"].data.shape == phasor.shape
    if inverse:
        newvis["vis"].data *= phasor
    else:
        newvis["vis"].data *= numpy.conj(phasor)
    # To rotate UVW, rotate into the global XYZ coordinate system and back.
    # We have the option of staying on the tangent plane or not.
    # If we stay on the tangent then the raster will join smoothly at the
    # edges. If we change the tangent then we will have to reproject to get
    # the results on the same image, in which case overlaps or gaps are
    # difficult to deal with.
    if not tangent:
        # The rotation can be done on the uvw (metres)
        # values but we also have to update
        # The wavelength dependent values
        nrows, nbl, _ = vis.uvw.shape
        if inverse:
            uvw_linear = vis.uvw.data.reshape([nrows * nbl, 3])
            xyz = uvw_to_xyz(
                uvw_linear,
                ha=-newvis.phasecentre.ra.rad,
                dec=newvis.phasecentre.dec.rad,
            )
            uvw_linear = xyz_to_uvw(
                xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad
            )[...]
        else:
            # This is the original (non-inverse) code
            uvw_linear = newvis.uvw.data.reshape([nrows * nbl, 3])
            xyz = uvw_to_xyz(
                uvw_linear,
                ha=-newvis.phasecentre.ra.rad,
                dec=newvis.phasecentre.dec.rad,
            )
            uvw_linear = xyz_to_uvw(
                xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad
            )[...]
        newvis.attrs["phasecentre"] = newphasecentre
        newvis["uvw"].data[...] = uvw_linear.reshape([nrows, nbl, 3])
        newvis = calculate_visibility_uvw_lambda(newvis)
    return newvis
