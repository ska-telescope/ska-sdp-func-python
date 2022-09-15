"""Unit tests for image deconvolution vis MSMFS


"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    deconvolve_list,
    restore_list,
    create_pb,
    image_scatter_channels,
    image_gather_channels,
    weight_visibility,
    taper_visibility_gaussian,
    qa_image,
)
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.imaging.base import (
    create_image_from_visibility,
)
from rascil.processing_components.imaging.imaging import (
    predict_visibility,
    invert_visibility,
)

from rascil.processing_components.imaging.primary_beams import create_low_test_beam
from rascil.processing_components.simulation import create_low_test_image_from_gleam
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.visibility.base import create_visibility

log = logging.getLogger("rascil-logger")

log.setLevel(logging.INFO)


class TestImageDeconvolutionMSMFS(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")
        self.persist = os.getenv("RASCIL_PERSIST", False)
        self.niter = 1000
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.lowcore = decimate_configuration(self.lowcore, skip=3)
        self.nchan = 6
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.linspace(0.9e8, 1.1e8, self.nchan)
        self.channel_bandwidth = numpy.array(
            self.nchan * [self.frequency[1] - self.frequency[0]]
        )
        self.phasecentre = SkyCoord(
            ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_visibility(
            config=self.lowcore,
            times=self.times,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
            zerow=True,
        )
        self.vis["vis"].data *= 0.0

        # Create model
        self.test_model = create_low_test_image_from_gleam(
            npixel=256,
            cellsize=0.001,
            phasecentre=self.vis.phasecentre,
            frequency=self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            flux_limit=1.0,
        )
        beam = create_low_test_beam(self.test_model)
        if self.persist:
            export_image_to_fits(
                beam, "%s/test_deconvolve_mmclean_beam.fits" % self.results_dir
            )
        self.test_model["pixels"].data *= beam["pixels"].data
        if self.persist:
            export_image_to_fits(
                self.test_model,
                "%s/test_deconvolve_mmclean_model.fits" % self.results_dir,
            )
        self.vis = predict_visibility(self.vis, self.test_model, context="2d")
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(
            self.vis,
            npixel=512,
            cellsize=0.001,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.vis = weight_visibility(self.vis, self.model)
        self.vis = taper_visibility_gaussian(self.vis, 0.002)
        self.dirty, sumwt = invert_visibility(self.vis, self.model, context="2d")
        self.psf, sumwt = invert_visibility(
            self.vis, self.model, context="2d", dopsf=True
        )
        if self.persist:
            export_image_to_fits(
                self.dirty, "%s/test_deconvolve_mmclean-dirty.fits" % self.results_dir
            )
        if self.persist:
            export_image_to_fits(
                self.psf, "%s/test_deconvolve_mmclean-psf.fits" % self.results_dir
            )
        self.dirty = image_scatter_channels(self.dirty)
        self.psf = image_scatter_channels(self.psf)
        window = numpy.ones(shape=self.model["pixels"].shape, dtype=bool)
        window[..., 65:192, 65:192] = True
        self.innerquarter = create_image_from_array(
            window,
            self.model.image_acc.wcs,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.innerquarter = image_scatter_channels(self.innerquarter)
        self.sensitivity = create_pb(self.model, "LOW")
        self.sensitivity = image_scatter_channels(self.sensitivity)

    def test_deconvolve_mmclean_no_taylor(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_no_taylor", 12.806085871833158, -0.14297206892008504
        )

    def test_deconvolve_mmclean_no_taylor_edge(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="no_edge",
            window_edge=32,
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_no_taylor_edge", 12.806085871833158, -0.1429720689200851
        )

    def test_deconvolve_mmclean_no_taylor_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=1,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_notaylor_noscales", 12.874215203967717, -0.14419436344642067
        )

    def test_deconvolve_mmclean_linear(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear", 15.207396524333546, -0.14224980487729696
        )

    def test_deconvolve_mmclean_linear_sensitivity(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            sensitivity=self.sensitivity,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        if self.persist:
            sensitivity = image_gather_channels(self.sensitivity)
            export_image_to_fits(
                sensitivity,
                "%s/test_deconvolve_mmclean_linear_sensitivity.fits" % self.results_dir,
            )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear_sensitivity", 15.207396524333546, -0.14224980487729716
        )

    def test_deconvolve_mmclean_linear_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=2,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_linear_noscales", 15.554039669750269, -0.14697685168807129
        )

    def test_deconvolve_mmclean_quadratic(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic", 15.302992891627193, -0.15373682171426403
        )

    def test_deconvolve_mmclean_quadratic_noscales(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic_noscales", 15.69172353540307, -0.1654330930047646
        )

    def save_and_check_images(self, tag, flux_max=0.0, flux_min=0.0):
        """Save the images with standard names

        :param tag: Informational, unique tag
        :return:
        """
        cmodel = image_gather_channels(self.cmodel)
        if self.persist:
            comp = image_gather_channels(self.comp)
            export_image_to_fits(
                comp,
                f"{self.results_dir}/test_deconvolve_{tag}_deconvolved.fits",
            )
            residual = image_gather_channels(self.residual)
            export_image_to_fits(
                residual,
                f"{self.results_dir}/test_deconvolve_{tag}_residual.fits",
            )
            export_image_to_fits(
                cmodel,
                f"{self.results_dir}/test_deconvolve_{tag}_restored.fits",
            )
        qa = qa_image(cmodel)
        numpy.testing.assert_allclose(
            qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}"
        )

    def test_deconvolve_mmclean_quadratic_psf_support(self):
        self.comp, self.residual = deconvolve_list(
            self.dirty,
            self.psf,
            niter=self.niter,
            gain=0.1,
            algorithm="mmclean",
            scales=[0, 3, 10],
            threshold=0.01,
            nmoment=3,
            findpeak="RASCIL",
            fractional_threshold=0.01,
            window_shape="quarter",
            psf_support=32,
        )
        self.cmodel = restore_list(self.comp, self.psf, self.residual)
        self.save_and_check_images(
            "mmclean_quadratic_psf", 15.322874439605584, -0.23892365313457908
        )


if __name__ == "__main__":
    unittest.main()
