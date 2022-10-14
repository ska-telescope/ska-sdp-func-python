"""Unit tests for individual functions in imaging_qa_main.py

"""
# pylint: disable=bad-string-format-type

import glob
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_array_almost_equal

from rascil.apps.imaging_qa_main import (
    cli_parser,
    analyze_image,
    correct_primary_beam,
    read_skycomponent_from_txt,
)
from rascil.data_models.polarisation_data_models import PolarisationFrame
from rascil.processing_components.image import (
    create_image,
)
from rascil.processing_components.imaging.primary_beams import (
    create_pb,
    create_low_test_beam,
)
from rascil.processing_components.parameters import rascil_path
from rascil.processing_components.simulation import (
    create_mid_simulation_components,
    find_pb_width_null,
)
from rascil.processing_components.skycomponent import (
    apply_beam_to_skycomponent,
    fit_skycomponent_spectral_index,
)
from rascil.processing_components.util.coordinate_support import hadec_to_azel

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestCIChecker(unittest.TestCase):
    def setUp(self):

        # Generate mock objects for functions in imaging_qa,
        # including skycomponents, primary beam image, restored image
        self.tempdir_root = tempfile.TemporaryDirectory(dir=rascil_path("test_results"))
        self.results_dir = self.tempdir_root.name

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.nchan = 8
        self.central_frequency = numpy.array([1e9])
        self.component_frequency = numpy.linspace(0.8e9, 1.2e9, self.nchan)
        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.npixel = 512
        self.cellsize = 0.0001
        self.bandwidth_array = numpy.full(self.nchan, 1e7)

        # Below are parameters used to generate primary beam
        hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(
            pbtype="MID", frequency=self.central_frequency
        )

        hwhm = hwhm_deg * numpy.pi / 180.0
        self.pbradius = 1.5 * hwhm
        self.pb_npixel = 256
        fov_deg = 8.0 * 1.36e9 / self.central_frequency
        d2r = numpy.pi / 180.0
        self.pb_cellsize = d2r * fov_deg / self.pb_npixel
        self.flux_limit = 0.001

        # List of skycomponents to test
        self.components = create_mid_simulation_components(
            self.phasecentre,
            self.component_frequency,
            self.flux_limit,
            self.pbradius,
            self.pb_npixel,
            self.pb_cellsize,
            apply_pb=False,
        )[0]

        # Test restored image
        self.multi_chan_image = create_image(
            npixel=self.npixel,
            cellsize=self.cellsize,
            polarisation_frame=PolarisationFrame("stokesI"),
            frequency=self.component_frequency,
            channel_bandwidth=self.bandwidth_array,
            phasecentre=self.phasecentre,
        )

        self.restored_image_multi = (
            self.results_dir + "/test_imaging_qa_functions_nchan8_restored.fits"
        )

        self.txtfile = self.results_dir + "/test_imaging_qa_functions.txt"

        pbmodel = create_image(
            npixel=self.pb_npixel,
            cellsize=self.pb_cellsize,
            phasecentre=self.phasecentre,
            frequency=self.component_frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        self.pb_mid = create_pb(
            pbmodel, "MID", pointingcentre=self.phasecentre, use_local=False
        )
        self.components_with_pb_mid = apply_beam_to_skycomponent(
            self.components, self.pb_mid
        )

        az, el = hadec_to_azel(0.0 * u.deg, self.phasecentre.dec, -27.0 * u.deg)
        beam_low_local = create_low_test_beam(
            self.multi_chan_image, use_local=True, azel=(az, el)
        )
        self.pb_low = create_low_test_beam(self.multi_chan_image, use_local=False)
        self.pb_low["pixels"].data = beam_low_local["pixels"].data
        self.components_with_pb_low = apply_beam_to_skycomponent(
            self.components, self.pb_low
        )

        parser = cli_parser()
        self.args = parser.parse_args([])

    def persist_data_files(self):
        """Persist the temporary data files"""
        to_copy = self.results_dir + "/test_imaging_qa_functions*"
        for f in glob.glob(to_copy):
            shutil.copy(f, rascil_path("test_results"))

    def test_correct_primary_beam_sensitivity(self):

        # Test using sensitivity image
        # After adding and dividing primary beam, the flux should stay roughly the same

        sensitivity_image = (
            self.results_dir + "/test_imaging_qa_functions_sensitivity.fits"
        )
        self.pb_mid.export_to_fits(sensitivity_image)

        reversed_comp = correct_primary_beam(
            None, sensitivity_image, self.components_with_pb_mid, "MID"
        )

        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux = [c.flux[self.nchan // 2][0] for c in reversed_comp]

        assert_array_almost_equal(orig_flux, reversed_flux, decimal=3)

    def test_correct_primary_beam_restored_mid(self):

        self.multi_chan_image.export_to_fits(self.restored_image_multi)
        # Test using restored image
        reversed_comp_rest = correct_primary_beam(
            self.restored_image_multi,
            None,
            self.components_with_pb_mid,
            telescope="MID",
        )
        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux_rest = [c.flux[self.nchan // 2][0] for c in reversed_comp_rest]

        assert_array_almost_equal(orig_flux, reversed_flux_rest, decimal=1)

        if self.persist:
            self.persist_data_files()

    def test_correct_primary_beam_restored_low(self):

        self.multi_chan_image.export_to_fits(self.restored_image_multi)
        # Test using restored image
        reversed_comp_rest = correct_primary_beam(
            self.restored_image_multi,
            None,
            self.components_with_pb_low,
            telescope="LOW",
        )
        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        reversed_flux_rest = [c.flux[self.nchan // 2][0] for c in reversed_comp_rest]

        assert_array_almost_equal(orig_flux, reversed_flux_rest, decimal=1)

        if self.persist:
            self.persist_data_files()

    def test_correct_primary_beam_none(self):

        # If we don't feed either sensitivity or restored image, should return the same components back
        reversed_comp_none = correct_primary_beam(
            None, None, self.components_with_pb_mid, "MID"
        )
        assert self.components_with_pb_mid == reversed_comp_none

        if self.persist:
            self.persist_data_files()

    def write_txt_file(self):

        f = open(self.txtfile, "w")
        f.write(
            "# RA(deg), Dec(deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz), Spectral index\n"
        )
        for cmp in self.components:
            coord_ra = cmp.direction.ra.degree
            coord_dec = cmp.direction.dec.degree
            spec_indx = fit_skycomponent_spectral_index(cmp)
            f.write(
                "%.6f, %.6f, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e \n"
                % (
                    coord_ra,
                    coord_dec,
                    cmp.flux[self.nchan // 2],
                    0.0,
                    0.0,
                    0.0,
                    self.central_frequency,
                    spec_indx,
                )
            )
        f.close()

    def test_read_skycomponent_from_txt_multi(self):

        self.write_txt_file()
        components_read = read_skycomponent_from_txt(
            self.txtfile, self.component_frequency
        )
        assert len(components_read) == len(self.components)
        orig_ra = [c.direction.ra.degree for c in self.components]
        read_ra = [c.direction.ra.degree for c in components_read]

        assert_array_almost_equal(orig_ra, read_ra)

        if self.persist:
            self.persist_data_files()

    def test_read_skycomponent_from_txt_single(self):

        self.write_txt_file()
        # Test single channel components
        # The flux for each component should only have one value
        components_read_single = read_skycomponent_from_txt(
            self.txtfile, self.central_frequency
        )

        for c in components_read_single:
            assert c.flux.shape == (1, 1)

        orig_flux = [c.flux[self.nchan // 2][0] for c in self.components]
        read_flux = [c.flux[0][0] for c in components_read_single]

        assert_array_almost_equal(orig_flux, read_flux)

        if self.persist:
            self.persist_data_files()

    def test_wrong_restored(self):

        # This part tests for no image input
        self.args.ingest_fitsname_restored = None
        self.assertRaises(FileNotFoundError, lambda: analyze_image(self.args))

    @patch("rascil.apps.imaging_qa_main.imaging_qa_bdsf")
    def test_analyze_image_exceptions(self, mock_checker):
        mock_checker.return_value = Mock()
        self.args.ingest_fitsname_restored = self.restored_image_multi
        self.args.restart = "False"

        self.multi_chan_image.export_to_fits(self.restored_image_multi)

        result = analyze_image(self.args)

        # call_args_list returns the input for function imaging_qa_bdsf
        # Assert using automatic beam size
        assert mock_checker.call_args_list[0][0][2] == (1.0, 1.0, 0.0)

        # Assert returning None values
        assert result == (None, None)

        if self.persist:
            self.persist_data_files()


if __name__ == "__main__":
    unittest.main()
