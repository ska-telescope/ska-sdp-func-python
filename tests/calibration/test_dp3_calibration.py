# pylint: skip-file
# flake8: noqa
"""
Unit tests for calibration solution
"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_functions import export_skymodel_to_text
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent, SkyModel
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.calibration.dp3_calibration import (
    create_parset_from_context,
    dp3_gaincal,
)

log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


class TestCalibrationDP3(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(1805550721)

    def actualSetup(
        self,
        sky_pol_frame="stokesIQUV",
        data_pol_frame="linear",
        f=None,
        vnchan=1,
    ):
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 30.0, 3)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        if vnchan > 1:
            self.channel_bandwidth = numpy.array(
                vnchan * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.channel_bandwidth = numpy.array([2e7])

        if f is None:
            f = [100.0, 50.0, -10.0, 40.0]

        if sky_pol_frame == "stokesI":
            f = [100.0]

        self.flux = numpy.outer(
            numpy.array(
                [numpy.power(freq / 1e8, -0.7) for freq in self.frequency]
            ),
            f,
        )

        # The phase centre is absolute and the component is specified relative
        # This means that the component should end up at the position
        # phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.comp = SkyComponent(
            direction=self.compabsdirection,
            frequency=self.frequency,
            flux=self.flux,
            polarisation_frame=PolarisationFrame(sky_pol_frame),
        )
        self.vis = create_visibility(
            self.lowcore,
            self.times,
            self.frequency,
            phasecentre=self.phasecentre,
            channel_bandwidth=self.channel_bandwidth,
            weight=1.0,
            polarisation_frame=PolarisationFrame(data_pol_frame),
        )

    def test_dp3_gaincal(self):
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

            self.actualSetup()

            export_skymodel_to_text(
                SkyModel(components=self.comp), "test.skymodel"
            )

            # Check that the call is successful
            dp3_gaincal(self.vis, ["T"], True)

    def test_create_parset_from_context(self):
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

            self.actualSetup()

            calibration_context_list = []
            calibration_context_list.append("T")
            calibration_context_list.append("G")
            calibration_context_list.append("B")

            global_solution = True

            parset_list = create_parset_from_context(
                self.vis, calibration_context_list, global_solution
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
                    assert (
                        parset_list[i].get_string("gaincal.caltype")
                        == "scalar"
                    )
                    nbins = max(
                        1,
                        numpy.ceil(
                            (
                                numpy.max(self.vis.time.data)
                                - numpy.min(self.vis.time.data)
                            )
                            / 60.0
                        ).astype("int"),
                    )
                    assert parset_list[i].get_string("gaincal.solint") == str(
                        nbins
                    )
                elif calibration_context_list[i] == "B":
                    assert (
                        parset_list[i].get_string("gaincal.caltype")
                        == "scalar"
                    )
                    nbins = max(
                        1,
                        numpy.ceil(
                            (
                                numpy.max(self.vis.time.data)
                                - numpy.min(self.vis.time.data)
                            )
                            / 1e5
                        ).astype("int"),
                    )
                    assert parset_list[i].get_string("gaincal.solint") == str(
                        nbins
                    )


if __name__ == "__main__":
    unittest.main()
