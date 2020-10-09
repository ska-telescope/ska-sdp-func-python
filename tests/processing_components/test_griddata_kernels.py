""" Unit tests for image operations


"""
import functools
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_image
from rascil.processing_components.griddata import \
    apply_bounding_box_convolutionfunction, calculate_bounding_box_convolutionfunction
from rascil.processing_components.griddata.kernels import create_pswf_convolutionfunction, \
    create_awterm_convolutionfunction, create_box_convolutionfunction
from rascil.processing_components.griddata.convolution_functions import export_convolutionfunction_to_fits
from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.imaging.primary_beams import create_pb_generic

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)


class TestGridDataKernels(unittest.TestCase):
    
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.001, phasecentre=self.phasecentre,
                                  polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    
    def test_fill_box_to_convolutionfunction(self):
        gcf, cf = create_box_convolutionfunction(self.image,
                                  polarisation_frame=PolarisationFrame("linear"))
        assert numpy.max(numpy.abs(cf.data.values)) > 0.0
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_box_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_box_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - 1.0) < 1e-15, "Peak is incorrect %s" % str(
            cf.data[peak_location] - 1.0)
        assert peak_location == (0, 0, 0, 0, 0, 2, 2), peak_location
    
    def test_fill_pswf_to_convolutionfunction(self):
        oversampling = 127
        support = 8
        gcf, cf = create_pswf_convolutionfunction(self.image, oversampling=oversampling, support=support,
                                  polarisation_frame=PolarisationFrame("linear"))
        assert numpy.max(numpy.abs(cf.data.values)) > 0.0
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_pswf_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_pswf_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - 0.18712109669890534 + 0j) < 1e-7, cf.data[peak_location]
        
        assert peak_location == (0, 0, 0, 63, 63, 4, 4), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
    
    def test_fill_pswf_to_convolutionfunction_nooversampling(self):
        oversampling = 1
        support = 8
        gcf, cf = create_pswf_convolutionfunction(self.image, oversampling=oversampling, support=support,
                                  polarisation_frame=PolarisationFrame("linear"))
        
        assert numpy.max(numpy.abs(cf.data.values)) > 0.0
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_pswf_nooversampling_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_pswf_nooversampling_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert numpy.abs(cf.data.values[peak_location] - 0.18712109669890536 + 0j) < 1e-7, cf.data.values[peak_location]
        assert peak_location == (0, 0, 0, 0, 0, 4, 4), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
    
    def test_fill_wterm_to_convolutionfunction(self):
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=None, nw=21, wstep=80.0, oversampling=8,
                                                    support=60, use_aaf=True,
                                                    polarisation_frame=PolarisationFrame("linear"))
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_wterm_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_wterm_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert peak_location == (0, 0, 10, 4, 4, 30, 30), peak_location
        assert numpy.abs(cf.data.values[peak_location] - (0.18709898-0.j)) < 1e-7, \
            cf.data[peak_location]
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
        
        cf_clipped = apply_bounding_box_convolutionfunction(cf, 5e-2)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data.values)), cf_clipped.shape)
        assert peak_location == (0, 0, 10, 4, 4, 12, 12), peak_location
        
        assert numpy.abs(cf_clipped.data[peak_location] - (0.18709898-0.j)) < 1e-7, \
            cf_clipped.data[peak_location]
        u_peak, v_peak = cf_clipped.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak

    def test_fill_awterm_to_convolutionfunction(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.image)
        if self.persist:
            export_image_to_fits(pb, "%s/test_convolutionfunction_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=21, wstep=80, oversampling=8,
                                                    support=60, use_aaf=True,
                                  polarisation_frame=PolarisationFrame("linear"))
        
        assert numpy.max(numpy.abs(cf.data.values)) > 0.0
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_awterm_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_awterm_cf.fits" % self.dir)
        # cf_image = convert_convolutionfunction_to_image(cf)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_cf.fits" % self.dir)
        
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert peak_location == (0, 0, 10, 4, 4, 30, 30), peak_location
        assert numpy.abs(cf.data.values[peak_location] - (0.027645606329255775-0j)) < 1e-7, \
            cf.data.values[peak_location]
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
        
        bboxes = calculate_bounding_box_convolutionfunction(cf)
        assert len(bboxes) == 21, len(bboxes)
        assert len(bboxes[0]) == 3, len(bboxes[0])
        assert bboxes[-1][0] == 20, bboxes[-1][0]
        
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert peak_location == (0, 0, 10, 4, 4, 30, 30), peak_location
        
        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=5e-2)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data.values)), cf_clipped.shape)
        assert peak_location == (0, 0, 10, 4, 4, 23, 23), peak_location
        # cf_image = convert_convolutionfunction_to_image(cf_clipped)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_clipped_cf.fits" % self.dir)
    
    def test_fill_aterm_to_convolutionfunction(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.image)
        if self.persist:
            export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=1, wstep=1e-7, oversampling=16,
                                                    support=16, use_aaf=True,
                                  polarisation_frame=PolarisationFrame("linear"))
        # cf_image = convert_convolutionfunction_to_image(cf)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_cf.fits" % self.dir)
        #
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_aterm_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert numpy.abs(cf.data.values[peak_location] - 0.02722692-0.j) < 1e-7, cf.data[peak_location]
        assert peak_location == (0, 0, 0, 8, 8, 8, 8), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
        
        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=5e-2)
        # cf_image = convert_convolutionfunction_to_image(cf_clipped)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_clipped_cf.fits" % self.dir)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data.values)), cf_clipped.shape)
        assert peak_location == (0, 0, 0, 8, 8, 7, 7), peak_location
    
    def test_compare_wterm_symmetry(self):
        _, cf = create_awterm_convolutionfunction(self.image, nw=11, wstep=8, oversampling=8,
                                                  support=60, use_aaf=True,
                                  polarisation_frame=PolarisationFrame("linear"))
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert peak_location == (0, 0, 5, 4, 4, 30, 30), peak_location
        assert numpy.abs(cf.data[peak_location] - (0.18704646-0.j)) < 1e-7, cf.data[peak_location]
        
        # Side to side in u,v
        p1 = (0, 0, 5, 4, 4, 25, 25)
        p2 = (0, 0, 5, 4, 4, 35, 35)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        p1 = (0, 0, 5, 4, 4, 25, 25)
        p2 = (0, 0, 5, 4, 4, 25, 35)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        p1 = (0, 0, 5, 4, 4, 25, 25)
        p2 = (0, 0, 5, 4, 4, 35, 35)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        p1 = (0, 0, 0, 4, 4, 25, 25)
        p2 = (0, 0, 0, 4, 4, 35, 35)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        p1 = (0, 0, 0, 4, 4, 25, 25)
        p2 = (0, 0, 0, 4, 4, 25, 35)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        p1 = (0, 0, 0, 4, 4, 25, 25)
        p2 = (0, 0, 0, 4, 4, 35, 25)
        assert numpy.abs(cf.data[p1] - cf.data[p2]) < 1e-15
        
        # w, -w must be conjugates
        p1 = (0, 0, 5 - 3, 4, 4, 25, 25)
        p2 = (0, 0, 5 + 3, 4, 4, 25, 25)
        assert numpy.abs(cf.data[p1] - numpy.conjugate(cf.data[p2])) < 1e-15
        
        p1 = (0, 0, 5 - 3, 4, 4, 25, 25)
        p2 = (0, 0, 5 + 3, 4, 4, 35, 35)
        assert numpy.abs(cf.data[p1] - numpy.conjugate(cf.data[p2])) < 1e-15
    
    def test_fill_aterm_to_convolutionfunction_noover(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.image)
        
        if self.persist:
            export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=1, wstep=1e-7, oversampling=1,
                                                    support=32, use_aaf=True,
                                  polarisation_frame=PolarisationFrame("linear"))
        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_noover_gcf.fits" % self.dir)
            export_convolutionfunction_to_fits(cf, "%s/test_convolutionfunction_aterm_pb_cf.fits" % self.dir)

        # cf_image = convert_convolutionfunction_to_image(cf)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_noover_cf.fits" % self.dir)
        #
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data.values)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - 0.02722652-0.j) < 1e-7, cf.data[peak_location]
        assert peak_location == (0, 0, 0, 0, 0, 16, 16), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
        

        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=5e-2)
        # cf_image = convert_convolutionfunction_to_image(cf_clipped)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_clipped_noover_cf.fits" % self.dir)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data.values)), cf_clipped.shape)
        assert peak_location == (0, 0, 0, 0, 0, 6, 6), peak_location


if __name__ == '__main__':
    unittest.main()
