"""
Functions to use DP3 for calibration purposes.
"""

__all__ = [
    "dp3_gaincal",
]

import logging

import numpy

from ska_sdp_func_python.calibration.chain_calibration import (
    create_calibration_controls,
)
from ska_sdp_func_python.visibility.operations import expand_polarizations

log = logging.getLogger("func-python-logger")


def create_parset_from_context(
    vis,
    calibration_context,
    global_solution,
    skymodel_filename,
):
    """Defines input parset for DP3 based on calibration context.

    :param vis: Visibility object
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Find a single solution over all frequency channels
    :param skymodel_filename: Filename of the skymodel used by DP3
    :return: list of parsets for the different calibrations to run
    """

    from dp3.parameterset import (  # noqa: E501 # pylint: disable=import-error,import-outside-toplevel
        ParameterSet,
    )

    parset_list = []
    controls = create_calibration_controls()
    for calibration_control in calibration_context:
        parset = ParameterSet()

        parset.add("gaincal.parmdb", "gaincal_solutions")
        parset.add("gaincal.sourcedb", skymodel_filename)
        timeslice = controls[calibration_control]["timeslice"]
        if timeslice == "auto" or timeslice is None or timeslice <= 0.0:
            parset.add("gaincal.solint", "1")
        else:
            nbins = max(
                1,
                numpy.ceil(
                    (numpy.max(vis.time.data) - numpy.min(vis.time.data))
                    / timeslice
                ).astype("int"),
            )
            parset.add("gaincal.solint", str(nbins))
        if global_solution:
            parset.add("gaincal.nchan", "0")
        else:
            parset.add("gaincal.nchan", "1")
        parset.add("gaincal.applysolution", "true")

        if controls[calibration_control]["phase_only"]:
            if controls[calibration_control]["shape"] == "matrix":
                parset.add("gaincal.caltype", "diagonalphase")
            else:
                parset.add("gaincal.caltype", "scalarphase")
        else:
            if controls[calibration_control]["shape"] == "matrix":
                parset.add("gaincal.caltype", "diagonal")
            else:
                parset.add("gaincal.caltype", "scalar")
        parset_list.append(parset)

    return parset_list


def dp3_gaincal(
    vis,
    calibration_context,
    global_solution,
    skymodel_filename="test.skymodel",
):
    """Calibrates visibilities using the DP3 package.

    :param vis: Visibility object (or graph)
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param skymodel_filename: Filename of the skymodel used by DP3
    :return: calibrated visibilities
    """

    from dp3 import (  # noqa: E501 # pylint:disable=no-name-in-module,import-error,import-outside-toplevel
        DPBuffer,
        DPInfo,
        MsType,
        make_step,
        steps,
    )

    log.info("Started computing dp3_gaincal")
    calibrated_vis = vis.copy(deep=True)

    parset_list = create_parset_from_context(
        calibrated_vis, calibration_context, global_solution, skymodel_filename
    )

    for parset in parset_list:
        gaincal_step = make_step(
            "gaincal",
            parset,
            "gaincal.",
            MsType.regular,
        )
        queue_step = steps.QueueOutput(parset, "")
        gaincal_step.set_next_step(queue_step)

        # DP3 GainCal step assumes 4 polarization are present in the visibility
        nr_correlations = 4
        dpinfo = DPInfo(nr_correlations)
        dpinfo.set_channels(vis.frequency.data, vis.channel_bandwidth.data)

        antenna1 = vis.antenna1.data
        antenna2 = vis.antenna2.data
        antenna_names = vis.configuration.names.data
        antenna_positions = vis.configuration.xyz.data
        antenna_diameters = vis.configuration.diameter.data
        dpinfo.set_antennas(
            antenna_names,
            antenna_diameters,
            antenna_positions,
            antenna1,
            antenna2,
        )
        first_time = vis.time.data[0]
        last_time = vis.time.data[-1]
        time_interval = vis.integration_time.data[0]
        dpinfo.set_times(first_time, last_time, time_interval)
        dpinfo.phase_center = [vis.phasecentre.ra.rad, vis.phasecentre.dec.rad]
        gaincal_step.set_info(dpinfo)
        queue_step.set_info(dpinfo)
        for time, vis_per_timeslot in calibrated_vis.groupby("time"):
            # Run DP3 GainCal step over each time step
            dpbuffer = DPBuffer()
            dpbuffer.set_time(time)
            dpbuffer.set_data(
                expand_polarizations(
                    vis_per_timeslot.vis.data, numpy.complex64
                )
            )
            dpbuffer.set_uvw(-vis_per_timeslot.uvw.data)
            dpbuffer.set_flags(
                expand_polarizations(vis_per_timeslot.flags.data, bool)
            )
            dpbuffer.set_weights(
                expand_polarizations(
                    vis_per_timeslot.weight.data, numpy.float32
                )
            )
            gaincal_step.process(dpbuffer)

        gaincal_step.finish()

        for time, vis_per_timeslot in calibrated_vis.groupby("time"):
            # Get data out of queue in QueueOutput step

            dpbuffer_from_queue = queue_step.queue.get()
            visibilities_out = numpy.array(
                dpbuffer_from_queue.get_data(), copy=False
            )
            nr_polarizations = vis_per_timeslot.vis.data.shape[-1]
            if nr_polarizations == 4:
                vis_per_timeslot.vis.data[:] = visibilities_out
            elif nr_polarizations == 2:
                vis_per_timeslot.vis.data[:, :, 0] = (
                    visibilities_out[:, :, 0] + visibilities_out[:, :, 1]
                ) / 2
            else:
                vis_per_timeslot.vis.data[:, :, 0] = (
                    visibilities_out[:, :, 0] + visibilities_out[:, :, 3]
                ) / 2

        log.info("Finished computing dp3_gaincal")

    return calibrated_vis
