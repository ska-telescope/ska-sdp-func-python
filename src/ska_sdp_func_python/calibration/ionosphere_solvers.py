# pylint: disable=invalid-name,too-many-arguments,no-member

"""
Functions to solve for delta-TEC variations across the array
"""

__all__ = ["solve_ionosphere"]

import logging

import numpy
from astropy import constants as const
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.visibility.vis_model import Visibility
from ska_sdp_func_python.calibration.ionosphere_utils import zern_array

log = logging.getLogger("func-python-logger")


def solve_ionosphere(
    vis: Visibility,
    modelvis: Visibility,
    xyz,
    cluster_id=None,
    niter=15,
) -> GainTable:
    """
    Solve a gain table by fitting for delta-TEC variations across the array
    The resulting delta-TEC variations will be converted to antenna-dependent
    phase shifts and the gain_table updated.

    Fits are performed within user-defined station clusters

    TODO: user (and/or auto) control of num param per cluster
    TODO: tolerance stopping
    TODO: phase referencing to reference antenna

    :param vis: Visibility containing the observed data_model
    :param modelvis: Visibility containing the predicted data_model
    :param xyz: [n_antenna,3] array containing the antenna locations in the
        local horizontal frame
    :param cluster_id: [n_antenna] array containing the cluster ID of each
        antenna. Defaults to a single cluster comprising all stations
    :param niter: Number of iterations (default 15)
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance. [Not yet implemented]
    :param ref_ant: integer index of reference antenna. No phase referencing if
        unset. [Not yet implemented]
    :return: GainTable containing solutions

    """
    if numpy.all(modelvis.vis == 0.0):
        raise ValueError("solve_ionosphere: Model visibilities are zero")

    # Create a new gaintable based on the visibilities
    # In general it will be filled with antenna-based phase shifts per channel
    gain_table = create_gaintable_from_visibility(vis, jones_type="B")

    # Ensure that the gain table and the input cluster indices are consistent
    if cluster_id is None:
        cluster_id = numpy.zeros(len(gain_table.antenna), "int")

    n_cluster = numpy.amax(cluster_id) + 1

    # Could be less strict & require max(gain_table.antenna) < len(cluster_id)
    if len(gain_table.antenna) != len(cluster_id):
        raise ValueError(f"cluster_id has wrong size {len(cluster_id)}")

    # Calculate coefficients for each cluster and initialise parameter values
    [param, coeff] = set_coeffs_and_params(xyz, cluster_id)

    n_param = get_param_count(param)[0]

    log.info(
        "Setting up iono solver for %d stations in %d clusters",
        len(gain_table.antenna),
        n_cluster,
    )
    if n_cluster == 0:
        log.info("There are %d total parameters in a single cluster", n_param)
    else:
        log.info(
            "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
            n_param,
            len(param[0]),
            len(param[1]),
            len(param)-1,
        )

    for it in range(niter):
        [AA, Ab] = build_normal_equation(
            vis, modelvis, param, coeff, cluster_id
        )

        # Solve the normal equations and update parameters
        param_update = solve_normal_equation(AA, Ab, param, it)

        # Update the model
        apply_phase_distortions(modelvis, param_update, coeff, cluster_id)

        # test something against tol here
        #  - max or rms of the relative update of gain or vis

    # Update and return the gain table
    update_gain_table(gain_table, param, coeff, cluster_id)

    return gain_table


def set_cluster_maps(cluster_id):
    """
    ...

    :param: cluster_id
    :return n_cluster: total number of clusters
    :return mask: station index mask for each cluster
    :return cid2stn: mapping from station index to cluster index
    :return stn2cid: mapping from cluster index to a list of station indices

    """
    n_station = len(cluster_id)
    n_cluster = numpy.amax(cluster_id) + 1

    # Mask for each cluster to flag which stations are present
    mask = numpy.empty((n_cluster, n_station), "bool")

    # Mapping from station index to cluster index
    stn2cid = numpy.empty(n_station, "int")

    # Mapping from cluster index to a list of station indices
    cid2stn = []

    stations = numpy.arange(n_station).astype("int")
    for cid in range(n_cluster):
        mask[cid, :] = cluster_id == cid
        cid2stn.append(stations[mask[cid, :]])
        stn2cid[mask[cid, :]] = cid

    return n_cluster, mask, cid2stn, stn2cid


def get_param_count(param):
    """
    ...

    :param param: [n_cluster] list of solution vectors, one for each cluster
    :return n_param: int, total number of parameters
    :return pidx0: [n_cluster], starting index of each cluster in param vectors

    """
    n_cluster = len(param)

    # Total number of parameters across all clusters
    n_param = 0

    # Starting parameter for each cluster
    pidx0 = numpy.zeros(n_cluster, "int")

    for cid in range(n_cluster):
        pidx0[cid] = n_param
        n_param += len(param[cid])

    return n_param, pidx0


def set_coeffs_and_params(
    xyz,
    cluster_id,
):
    """
    Calculate coefficients (a basis function value vector for each cluster) and
    initialise parameter values (a solution vector for each station)

    :param vis: Visibility containing the observed data_model
    :param xyz: [n_antenna,3] array containing the antenna locations in the
        local horizontal frame
    :param cluster_id:
    :return param: [n_cluster] list of solution vectors
    :return coeff: [n_station] list of basis-func value vectors
        Stored as a numpy dtype=object array of variable-length coeff vectors

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, _, cid2stn, _] = set_cluster_maps(cluster_id)

    n_station = len(cluster_id)
    coeff = [None] * n_station
    param = [None] * n_cluster

    # Coefficients and parameter values
    # treat cluster zero differently; it is assumed to be a larger central core
    cid = 0

    # Get Zernike parameters for the stations in this cluster
    zern_params = zern_array(
        6, xyz[cid2stn[cid], 0], xyz[cid2stn[cid], 1], noll_order=False
    )

    # note: This isn't the most efficient approach, but get it working first
    print(zern_params.shape)
    for idx, stn in enumerate(cid2stn[cid]):
        coeff[stn] = zern_params[idx]

    if len(cid2stn[cid]) > 0:
        param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    # now do the rest of the clusters
    for cid in range(1, n_cluster):
        # Remove the average position of the cluster
        xave = numpy.mean(xyz[cid2stn[cid], 0])
        yave = numpy.mean(xyz[cid2stn[cid], 1])
        for stn in cid2stn[cid]:
            # coeff[stn] = numpy.array([1, x[stn], y[stn]])
            coeff[stn] = numpy.array(
                [
                    1,
                    xyz[stn, 0] - xave,
                    xyz[stn, 1] - yave,
                ]
            )
        if len(cid2stn[cid]) > 0:
            param[cid] = numpy.zeros(len(coeff[cid2stn[cid][0]]))

    return param, numpy.array(coeff, dtype=object)


def apply_phase_distortions(
    vis: Visibility,
    param,
    coeff,
    cluster_id,
):
    """
    ...

    :param vis: Visibility containing the data_models to be distorted
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id:

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, _, _, stn2cid] = set_cluster_maps(cluster_id)

    # set up a few references and constants
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data
    vis_data = vis.vis.data

    T = numpy.dtype(coeff[0][0])

    for chan in range(len(vis.frequency)):
        pconst = 1j * 2.0 * numpy.pi * const.c.value / vis.frequency.data[chan]
        # Loop over pairs of clusters and update the associated baselines
        for cid1 in range(0, n_cluster):
            for cid2 in range(0, n_cluster):
                # A mask for all baselines in this cluster pair
                # DAM could/should remove autos at this point as well
                vismask = (stn2cid[ant1] == cid1) * (stn2cid[ant2] == cid2)
                if numpy.sum(vismask) == 0:
                    continue

                vis_data[0, vismask, chan, 0] *= numpy.exp(
                    pconst
                    * (
                        numpy.einsum(
                            "j,ij->i",
                            param[cid1],
                            numpy.vstack(coeff[ant1[vismask]]).astype(T),
                        )
                        - numpy.einsum(
                            "j,ij->i",
                            param[cid2],
                            numpy.vstack(coeff[ant2[vismask]]).astype(T),
                        )
                    )
                )


def build_normal_equation(
    vis: Visibility,
    modelvis: Visibility,
    param,
    coeff,
    cluster_id,
):
    # pylint: disable=too-many-locals
    """
    ...

    :param vis: Visibility containing the observed data_models
    :param modelvis: Visibility containing the predicted data_models
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id:

    """
    # Get common mapping vectors between stations and clusters
    [n_cluster, _, _, stn2cid] = set_cluster_maps(cluster_id)
    [n_param, pidx0] = get_param_count(param)

    # set up a few references and constants
    frequency = vis.frequency.data
    wl = const.c.value / frequency
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data
    vis_data = vis.vis.data
    mdl_data = modelvis.vis.data

    n_baselines = len(vis.baselines)

    T = numpy.dtype(coeff[0][0])

    # Loop over frequency and accumulate normal equations
    #     note: Can probably handly frequency within an einsum
    #           Get it working in a loop first

    AA = numpy.zeros((n_param, n_param))
    Ab = numpy.zeros(n_param)

    for chan in range(len(frequency)):
        # Could accumulate AA and Ab directly, but go via a
        # design matrix for clarity. Update later if need be.

        A = numpy.zeros((n_param, n_baselines))

        # Just use simple point-source phasing for now
        phase_model = mdl_data[0, :, chan, 0] / numpy.abs(
            mdl_data[0, :, chan, 0]
        )

        # Precalculate some constants
        A0 = (
            2.0
            * numpy.pi
            * wl[chan]
            * numpy.real(mdl_data[0, :, chan, 0] * numpy.conj(phase_model))
        )

        # Loop over pairs of clusters and update the design matrix for the
        # associated baselines
        for cid1 in range(0, n_cluster):
            pidx1 = numpy.arange(pidx0[cid1], pidx0[cid1] + len(param[cid1]))

            for cid2 in range(0, n_cluster):
                pidx2 = numpy.arange(
                    pidx0[cid2], pidx0[cid2] + len(param[cid2])
                )

                # A mask for all baselines in this cluster pair
                # DAM could/should remove autos at this point as well
                vismask = (stn2cid[ant1] == cid1) * (stn2cid[ant2] == cid2)

                if numpy.sum(vismask) == 0:
                    continue

                # indices of baselines in this cluster pair
                blidx = numpy.arange(n_baselines)[vismask]

                # [nvis] A0 terms x [nvis,nparam] coeffs (1st antenna)
                # need to replicate 1D index vectors across the 2D output
                ii = numpy.tile(pidx1[:, numpy.newaxis], (1, len(blidx)))
                jj = numpy.tile(blidx[numpy.newaxis, :], (len(pidx1), 1))
                # all masked antennas have the same number of coeffs so can
                # form a coeff matrix and multiply
                A[ii, jj] += numpy.einsum(
                    "i,ij->ji",
                    A0[vismask],
                    numpy.vstack(coeff[ant1[vismask]]).astype(T),
                )

                # [nvis] A0 terms x [nvis,nparam] coeffs 2nd antenna)
                ii = numpy.tile(pidx2[:, numpy.newaxis], (1, len(blidx)))
                jj = numpy.tile(blidx[numpy.newaxis, :], (len(pidx2), 1))
                A[ii, jj] -= numpy.einsum(
                    "i,ij->ji",
                    A0[vismask],
                    numpy.vstack(coeff[ant2[vismask]]).astype(T),
                )

        AA += numpy.einsum("ij,kj->ik", A, A)
        Ab += numpy.einsum(
            "ij,j->i",
            A,
            numpy.imag(
                (vis_data[0, :, chan, 0] - mdl_data[0, :, chan, 0])
                * numpy.conj(phase_model)
            ),
        )

    return AA, Ab


def solve_normal_equation(
    AA,
    Ab,
    param,
    it=0,
):
    """
    Solve the normal equations and update parameters

    Using the SVD-based DGELSD solver via numpy.linalg.lstsq.
    Could use the LU-decomposition-based DGESV solver in numpy.linalg.solve,
    but the normal matrix may not be full rank.

    If n_param gets large (~ 100) it may be better to use a numerical solver
    like lsmr or lsqr.

    :param AA: [n_param, n_param] normal equation
    :param Ab: [n_param] data vector
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param it: int, current iteration
    :return param_update: the current incremental param update

    """
    n_cluster = len(param)
    [_, pidx0] = get_param_count(param)

    # Solve the current incremental normal equations
    soln_vec = numpy.linalg.lstsq(AA, Ab, rcond=None)[0]

    # Make a copy of coeff for just the current incremental update
    param_update = []
    for cid in range(n_cluster):
        param_update.append(numpy.zeros(len(param[cid])))

    nu = 1.0 - 0.5 * (it % 2)
    for cid in range(n_cluster):
        param_update[cid] = (
            nu
            * soln_vec[pidx0[cid] : pidx0[cid] + len(param[cid])]  # noqa: E203
        )
        param[cid] += param_update[cid]

    return param_update


def update_gain_table(
    gain_table: GainTable,
    param,
    coeff,
    cluster_id,
):
    """
    ...

    :param gain_table: GainTable to be updated
    :param param: [n_cluster] list of solution vectors, one for each cluster
    :param coeff: [n_station] list of basis-func value vectors, one per station
        Stored as a numpy dtype=object array of variable-length coeff vectors
    :param cluster_id:

    """
    # set up a few references and constants
    n_cluster = numpy.amax(cluster_id) + 1
    n_station = len(cluster_id)

    wl = const.c.value / gain_table.frequency.data

    table_data = gain_table.gain.data

    T = numpy.dtype(coeff[0][0])

    # Total number of parameters across all clusters
    n_param = 0
    # Starting parameter for each cluster
    pidx0 = numpy.zeros(n_cluster, "int")
    # Mask for each cluster to flag which stations are present
    mask = numpy.empty((n_cluster, n_station), "bool")
    for cid in range(n_cluster):
        mask[cid, :] = cluster_id == cid
        pidx0[cid] = n_param
        n_param += len(param[cid])

    for cid in range(0, n_cluster):
        phase_term = (
            2.0
            * numpy.pi
            * numpy.einsum(
                "j,ij->i",
                param[cid],
                numpy.vstack(coeff[mask[cid, :]]).astype(T),
            )
        )
        for chan in range(len(gain_table.frequency)):
            table_data[0, mask[cid, :], chan, 0, 0] = numpy.exp(
                1j * wl[chan] * phase_term
            )
