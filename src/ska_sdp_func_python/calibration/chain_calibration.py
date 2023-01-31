"""
Functions to solve for and apply chains of antenna/station gain tables.
See documentation for further information.
"""

__all__ = [
    "apply_calibration_chain",
    "calibrate_chain",
    "create_calibration_controls",
    "solve_calibrate_chain",
]

import logging

import numpy
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_func_python.calibration.solvers import solve_gaintable

log = logging.getLogger("func-python-logger")


def create_calibration_controls():
    """
    Create a dictionary containing default chain calibration controls.

    The fields are

        T: Atmospheric phase
        G: Electronic gains
        P: Polarisation
        B: Bandpass
        I: Ionosphere

    Therefore, first get this default dictionary and then
    adjust parameters as desired.
    The calibrate function takes a context string e.g. TGB.
    It then calibrates each of these Jones matrices in turn.

     Note that P and I calibration require off diagonal terms producing n
     on-commutation of the Jones matrices. This is not handled yet.

    :return: dictionary
    """
    controls = {
        "T": {
            "shape": "scalar",
            "timeslice": "auto",
            "phase_only": True,
            "first_selfcal": 0,
        },
        "G": {
            "shape": "vector",
            "timeslice": 60.0,
            "phase_only": False,
            "first_selfcal": 0,
        },
        "B": {
            "shape": "vector",
            "timeslice": 1e5,
            "phase_only": False,
            "first_selfcal": 0,
        },
    }

    return controls


def apply_calibration_chain(
    vis,
    gaintables,
    calibration_context="T",
    controls=None,
    iteration=0,
):
    """
    Calibrate using algorithm specified by calibration_context
    and the calibration controls.

    The context string can denote a sequence of calibrations
    e.g. TGB with different timescales.

    :param vis: Visibility
    :param gaintables: GainTables to perform calibration
    :param calibration_context: calibration contexts in order
                    of correction e.g. 'TGB'
    :param controls: Controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared
                    to the 'first_selfcal' field.
    :return: Calibrated data_models, dict(gaintables)
    """

    if controls is None:
        controls = create_calibration_controls()

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if (iteration >= controls[c]["first_selfcal"]) and (
            c in gaintables.keys()
        ):
            changes = True

    if changes:

        for c in calibration_context:
            if iteration >= controls[c]["first_selfcal"]:
                avis = apply_gaintable(vis, gaintables[c])

        return avis

    return vis


def calibrate_chain(
    vis,
    model_vis,
    gaintables=None,
    calibration_context="T",
    controls=None,
    iteration=0,
    tol=1e-6,
):
    """
    Calibrate using algorithm specified by calibration_context.

    The context string can denote a sequence of calibrations
    e.g. TGB with different timescales.

    :param vis: Visibility containing the observed data_models
    :param model_vis: Visibility containing the visibility predicted by a model
    :param gaintables: Existing GainTables
    :param calibration_context: Calibration contexts in order
                of correction e.g. 'TGB'
    :param controls: Controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to
                the 'first_selfcal' field.
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :return: Calibrated data_models, dict(GainTables)
    """
    if controls is None:
        controls = create_calibration_controls()

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if iteration >= controls[c]["first_selfcal"]:
            changes = True

    if changes:

        avis = vis
        amvis = model_vis

        if gaintables is None:
            gaintables = {}

        for c in calibration_context:
            if iteration >= controls[c]["first_selfcal"]:
                if c not in gaintables.keys():
                    log.info("Creating new %s gaintable", c)
                    gaintables[c] = create_gaintable_from_visibility(
                        avis, timeslice=controls[c]["timeslice"], jones_type=c
                    )
                gaintables[c] = solve_gaintable(
                    avis,
                    amvis,
                    gain_table=gaintables[c],
                    phase_only=controls[c]["phase_only"],
                    crosspol=controls[c]["shape"] == "matrix",
                    timeslice=controls[c]["timeslice"],
                    tol=tol,
                )
                log.debug(
                    "calibrate_chain: Jones matrix %s, iteration %s",
                    c,
                    iteration,
                )
                log.debug(
                    gaintables[c].gaintable_acc.qa_gain_table(
                        context=f"Jones matrix {c}, iteration {iteration}"
                    )
                )
                avis = apply_gaintable(
                    avis,
                    gaintables[c],
                    inverse=True,
                )
            else:
                log.debug(
                    "calibrate_chain: Jones matrix %s "
                    "not solved, iteration %s",
                    c,
                    iteration,
                )

        return avis, gaintables

    return vis, gaintables


def solve_calibrate_chain(
    vis,
    model_vis,
    gaintables=None,
    calibration_context="T",
    controls=None,
    iteration=0,
    tol=1e-6,
):
    """
    Calibrate using algorithm specified by calibration_context.

    The context string can denote a sequence of calibrations
    e.g. TGB with different timescales.

    :param vis: Visibility containing the observed data_models
    :param model_vis: Visibility containing the visibility predicted by a model
    :param gaintables: Existing GainTables
    :param calibration_context: calibration contexts in order
                    of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to
                    the 'first_selfcal' field.
    :param tol: Iteration stops when the fractional change
                 in the gain solution is below this tolerance
    :return: Calibrated data_models, dict(gaintables)
    """
    if controls is None:
        controls = create_calibration_controls()

    avis = vis
    amvis = model_vis

    # Always return a gain table, even if null
    if gaintables is None:
        gaintables = {}

    for c in calibration_context:
        if c not in gaintables.keys():
            gaintables[c] = create_gaintable_from_visibility(
                avis, timeslice=controls[c]["timeslice"], jones_type=c
            )
        fmin = gaintables[c].frequency.data[0]
        fmax = gaintables[c].frequency.data[-1]
        if iteration >= controls[c]["first_selfcal"]:
            if numpy.max(
                numpy.abs(vis.visibility_acc.flagged_weight)
            ) > 0.0 and (
                amvis is None or numpy.max(numpy.abs(amvis.vis)) > 0.0
            ):
                gaintables[c] = solve_gaintable(
                    avis,
                    amvis,
                    gain_table=gaintables[c],
                    phase_only=controls[c]["phase_only"],
                    crosspol=controls[c]["shape"] == "matrix",
                    timeslice=controls[c]["timeslice"],
                    tol=tol,
                )
                context_message = (
                    f"Model is non-zero: solving for Jones matrix {c}, "
                    f"iteration {iteration}, frequency "
                    f"{fmin:4g} - {fmax:4g} Hz"
                )
                qa = gaintables[c].gaintable_acc.qa_gain_table(
                    context=context_message
                )
                log.info("calibrate_chain: %s", qa)
            else:
                log.info(
                    "No model data: cannot solve for Jones matrix %s, "
                    "iteration %s, frequency %4g - %4g Hz",
                    c,
                    iteration,
                    fmin,
                    fmax,
                )
        else:
            log.info(
                "Not solving for Jones matrix %s this iteration: "
                "iteration %s, frequency %4g - %4g Hz",
                c,
                iteration,
                fmin,
                fmax,
            )

    return gaintables
