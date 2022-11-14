# pylint: skip-file
# flake8: noqa
""" Unit tests for calibration solution


"""
import logging
import unittest

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from numpy.random import default_rng
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.visibility import create_visibility

from ska_sdp_func_python.calibration.chain_calibration import (
    calibrate_chain,
    create_calibration_controls,
)
from ska_sdp_func_python.calibration.operations import apply_gaintable
# from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

pytestmark = pytest.skip(
    allow_module_level=True,
    reason="not able importing ska-sdp-func in dft_skycomponent_visibility",
)
log = logging.getLogger("func-python-logger")

log.setLevel(logging.WARNING)


# Quick mockup of the simulate_gaintable() function in rascil_main
def simulate_gaintable(gt, phase_error, seed):
    rng = default_rng(seed)
    phases = rng.normal(0, phase_error, gt["gain"].data.shape)
    gt["gain"].data = 1 * numpy.exp(0 + 1j * phases)
    gt["gain"].data[..., 0, 1] = 0.0
    gt["gain"].data[..., 1, 0] = 0.0
    return gt


class TestCalibrationChain(unittest.TestCase):
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
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)

    def test_calibrate_T_function(self):
        self.actualSetup("stokesI", "stokesI", f=[100.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_visibility(self.vis)
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(gt, 10.0, 180550721)
        original = self.vis.copy(deep=True)
        self.vis = apply_gaintable(self.vis, gt)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["phase_only"] = False
        calibrated_vis, gaintables = calibrate_chain(
            self.vis, original, calibration_context="T", controls=controls
        )
        residual = numpy.max(gaintables["T"].residual)
        assert residual < 1.3e-6, "Max T residual = %s" % (residual)

    def test_calibrate_T_function_phase_only(self):
        self.actualSetup("stokesI", "stokesI", f=[100.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_visibility(self.vis)
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(gt, 10.0, 180550721)
        original = self.vis.copy(deep=True)
        self.vis = apply_gaintable(self.vis, gt)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["phase_only"] = True
        calibrated_vis, gaintables = calibrate_chain(
            self.vis, original, calibration_context="T", controls=controls
        )
        residual = numpy.max(gaintables["T"].residual)
        assert residual < 1e-6, "Max T residual = %s" % (residual)

    def test_calibrate_G_function(self):
        self.actualSetup("stokesIQUV", "linear", f=[100.0, 50.0, 0.0, 0.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_visibility(self.vis)
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(gt, 1.0, 180550721)
        corrupted = self.vis.copy(deep=True)
        corrupted = apply_gaintable(corrupted, gt)
        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls["G"]["first_selfcal"] = 0
        controls["G"]["timeslice"] = 0.0
        controls["G"]["phase_only"] = False
        calibrated_vis, gaintables = calibrate_chain(
            corrupted, self.vis, calibration_context="G", controls=controls
        )
        residual = numpy.max(gaintables["G"].residual)
        assert residual < 1e-6, "Max G residual = %s" % (residual)

    def test_calibrate_TG_function(self):
        self.actualSetup("stokesIQUV", "linear", f=[100.0, 50, 0.0, 0.0])
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_visibility(self.vis)
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(gt, 10.0, 180550721)
        original = self.vis.copy(deep=True)
        self.vis = apply_gaintable(self.vis, gt)

        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 0
        controls["T"]["timeslice"] = 0.0
        controls["T"]["phase_only"] = True
        controls["G"]["first_selfcal"] = 0
        controls["G"]["timeslice"] = 0.0
        controls["G"]["phase_only"] = False

        calibrated_vis, gaintables = calibrate_chain(
            self.vis, original, calibration_context="TG", controls=controls
        )
        # We test the G residual because it is calibrated last
        residual = numpy.max(gaintables["G"].residual)
        assert residual < 1e-6, "Max T residual = %s" % residual

    def test_calibrate_B_function(self):
        self.actualSetup(
            "stokesIQUV", "linear", f=[100.0, 50, 0.0, 0.0], vnchan=32
        )
        # Prepare the corrupted visibility data_models
        gt = create_gaintable_from_visibility(self.vis)
        log.info("Created gain table: %.3f GB" % (gt.gaintable_acc.size()))
        gt = simulate_gaintable(gt, 10.0, 180550721)
        original = self.vis.copy(deep=True)
        self.vis = apply_gaintable(self.vis, gt)

        # Now get the control dictionary and calibrate
        controls = create_calibration_controls()
        controls["T"]["first_selfcal"] = 10
        controls["T"]["timeslice"] = 0.0
        controls["T"]["phase_only"] = True

        controls["G"]["first_selfcal"] = 10
        controls["G"]["timeslice"] = 0.0
        controls["G"]["phase_only"] = True

        controls["B"]["first_selfcal"] = 0
        controls["B"]["timeslice"] = 0.0
        controls["B"]["phase_only"] = False

        calibrated_vis, gaintables = calibrate_chain(
            self.vis, original, calibration_context="B", controls=controls
        )
        residual = numpy.max(gaintables["B"].residual)
        assert residual < 1e-6, "Max B residual = %s" % residual


if __name__ == "__main__":
    unittest.main()
