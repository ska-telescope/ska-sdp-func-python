# pylint: skip-file
# flake8: noqa
"""
Unit tests for dp3 calibration
"""
import logging

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_functions import export_skymodel_to_text
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent, SkyModel

from ska_sdp_func_python.calibration.dp3_calibration import (
    create_parset_from_context,
    dp3_gaincal,
)

log = logging.getLogger("func-python-logger")


@pytest.fixture
def skycomponent():
    """Create a skycomponent to use for testing"""
    sky_pol_frame = "stokesIQUV"
    frequency = numpy.array([1.0e8])

    f = [100.0, 50.0, -10.0, 40.0]

    flux = numpy.outer(
        numpy.array([numpy.power(freq / 1e8, -0.7) for freq in frequency]),
        f,
    )

    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    comp = SkyComponent(
        direction=compabsdirection,
        frequency=frequency,
        flux=flux,
        polarisation_frame=PolarisationFrame(sky_pol_frame),
    )

    return comp


def test_dp3_gaincal(skycomponent, visibility):
    """
    Test that DP3 calibration runs without throwing exception.
    Only run this test if DP3 is available.
    """

    is_dp3_available = True
    try:
        import dp3  # pylint: disable=import-error
    except ImportError:
        log.info("DP3 module not available. Test is skipped.")
        is_dp3_available = False

    if is_dp3_available:

        export_skymodel_to_text(
            SkyModel(components=skycomponent), "test.skymodel"
        )

        # Check that the call is successful
        dp3_gaincal(visibility, ["T"], True)


def test_create_parset_from_context(visibility):
    """
    Test that the correct parset is created based on the calibration context.
    Only run this test if DP3 is available.
    """

    is_dp3_available = True
    try:
        import dp3  # pylint: disable=import-error
    except ImportError:
        log.info("DP3 module not available. Test is skipped.")
        is_dp3_available = False

    if is_dp3_available:

        calibration_context_list = []
        calibration_context_list.append("T")
        calibration_context_list.append("G")
        calibration_context_list.append("B")

        global_solution = True

        parset_list = create_parset_from_context(
            visibility,
            calibration_context_list,
            global_solution,
            "test.skymodel",
        )

        assert len(parset_list) == len(calibration_context_list)

        for i in numpy.arange(len(calibration_context_list)):

            assert parset_list[i].get_string("gaincal.nchan") == "0"
            if calibration_context_list[i] == "T":
                assert (
                    parset_list[i].get_string("gaincal.caltype")
                    == "scalarphase"
                )
                assert parset_list[i].get_string("gaincal.solint") == "1"
            elif calibration_context_list[i] == "G":
                assert parset_list[i].get_string("gaincal.caltype") == "scalar"
                nbins = max(
                    1,
                    numpy.ceil(
                        (
                            numpy.max(visibility.time.data)
                            - numpy.min(visibility.time.data)
                        )
                        / 60.0
                    ).astype("int"),
                )
                assert parset_list[i].get_string("gaincal.solint") == str(
                    nbins
                )
            elif calibration_context_list[i] == "B":
                assert parset_list[i].get_string("gaincal.caltype") == "scalar"
                nbins = max(
                    1,
                    numpy.ceil(
                        (
                            numpy.max(visibility.time.data)
                            - numpy.min(visibility.time.data)
                        )
                        / 1e5
                    ).astype("int"),
                )
                assert parset_list[i].get_string("gaincal.solint") == str(
                    nbins
                )
