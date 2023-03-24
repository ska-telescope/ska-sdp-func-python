"""
Functions to solve for delta-TEC variations across the array
"""

__all__ = ["solve_ionosphere"]

import logging

import numpy
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_func_python.visibility.operations import divide_visibility

log = logging.getLogger("func-python-logger")


def solve_ionosphere(
    vis: Visibility,
    modelvis: Visibility,
    gain_table=None,
) -> GainTable:
    """
    Solve a gain table by fitting for delta-TEC variations across the array
    The resulting delta-TEC variations will be converted to antenna-dependent
    phase shifts and the gain_table updated.

    The form of modelvis will likely need to change as direction-dependence is
    established.

    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the visibility predicted by a model
    :param gain_table: Existing gaintable
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :return: GainTable containing solution

    """
    if modelvis is not None:
        # pylint: disable=unneeded-not
        if not numpy.max(numpy.abs(modelvis.vis)) > 0.0:
            raise ValueError("solve_gaintable: Model visibility is zero")

    point_vis = (
        divide_visibility(vis, modelvis) if modelvis is not None else vis
    )

    if gain_table is None:
        log.debug("solve_ionosphere: creating new gaintable")
        gain_table = create_gaintable_from_visibility(vis, jones_type="B")
    else:
        log.debug("solve_ionosphere: starting from existing gaintable")

    # nants = gain_table.gaintable_acc.nants
    # nchan = gain_table.gaintable_acc.nchan
    # npol = point_vis.visibility_acc.npol

    for row, time in enumerate(gain_table.time):
        time_slice = {
            "time": slice(
                time - gain_table.interval[row] / 2,
                time + gain_table.interval[row] / 2,
            )
        }
        pointvis_sel = point_vis.sel(time_slice)
        # pylint: disable=unneeded-not
        if not pointvis_sel.visibility_acc.ntimes > 0:
            log.warning(
                "Gaintable %s, vis time mismatch %s", gain_table.time, vis.time
            )
            continue

        gain_table["gain"].data[row, ...] = 1.0 + 0.0j
        gain_table["weight"].data[row, ...] = 0.0
        gain_table["residual"].data[row, ...] = 0.0

    return gain_table
