""" Unit tests for image operations


"""
import logging
import unittest
import functools

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from processing_library.image.operations import create_image
from processing_components.convolution_function.kernels import create_pswf_convolutionfunction, \
    create_awterm_convolutionfunction, create_box_convolutionfunction
from processing_components.convolution_function.operations import convert_convolutionfunction_to_image, \
    create_convolutionfunction_from_image, apply_bounding_box_convolutionfunction, \
    calculate_bounding_box_convolutionfunction
from processing_components.image.operations import export_image_to_fits
from processing_components.imaging.primary_beams import create_pb_generic

log = logging.getLogger(__name__)


class TestGridDataKernels(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.0005, phasecentre=self.phasecentre)
    
    def test_create_convolutionfunction(self):
        cf = create_convolutionfunction_from_image(self.image, nz=1)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_cf.fits" % self.dir)

    def test_fill_box_to_convolutionfunction(self):
        gcf, cf = create_box_convolutionfunction(self.image)
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_box_gcf.fits" % self.dir)

        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_box_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location]-1.0) < 1e-15, "Peak is incorrect %s" % str(cf.data[peak_location]-1.0)
        assert peak_location == (0, 0, 0, 0, 0, 2, 2), peak_location

    def test_fill_pswf_to_convolutionfunction(self):
        oversampling=8
        support=6
        gcf, cf = create_pswf_convolutionfunction(self.image, oversampling=oversampling, support=support)
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_pswf_gcf.fits" % self.dir)

        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_pwsf_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location]-0.1871722034219655+0j) < 1e-7, cf.data[peak_location]
        
        assert peak_location == (0, 0, 0, 4, 4, 3, 3), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak)<1e-7, u_peak
        assert numpy.abs(v_peak)<1e-7, u_peak


    def test_fill_pswf_to_convolutionfunction_nooversampling(self):
        oversampling=1
        support=6
        gcf, cf = create_pswf_convolutionfunction(self.image, oversampling=oversampling, support=support)
        
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_pswf_nooversampling_gcf.fits" % self.dir)

        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_pwsf_nooversampling_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location]-0.18712109669890536+0j) < 1e-7, cf.data[peak_location]
        assert peak_location == (0, 0, 0, 0, 0, 3, 3), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak)<1e-7, u_peak
        assert numpy.abs(v_peak)<1e-7, u_peak
        

    def test_fill_wterm_to_convolutionfunction(self):
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=None,  nw=21, wstep=100.0, oversampling=4,
                                                    support=100, use_aaf=True)
        export_image_to_fits(gcf, "%s/test_convolutionfunction_wterm_gcf.fits" % self.dir)
    
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert peak_location == (0, 0, 10, 2, 2, 50, 50), peak_location
        assert numpy.abs(cf.data[peak_location] - (0.18704594760349508-0j)) < 1e-7, \
            cf.data[peak_location]
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak

        
        cf_clipped = apply_bounding_box_convolutionfunction(cf)
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_clipped_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 10, 2, 2, 40, 40), peak_location
        
        assert numpy.abs(cf_clipped.data[peak_location] - (0.18704594760349508-0j)) < 1e-7, \
            cf_clipped.data[peak_location]
        u_peak, v_peak = cf_clipped.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak


    def test_fill_wterm_to_convolutionfunction_nopswf(self):
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=None, nw=21, wstep=100.0, oversampling=8,
                                                    support=50, use_aaf=False)
        export_image_to_fits(gcf, "%s/test_convolutionfunction_wterm_nopswf_gcf.fits" % self.dir)

        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_wterm_nopswf_cf.fits" % self.dir)

    def test_fill_awterm_to_convolutionfunction(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.image)
        export_image_to_fits(pb, "%s/test_convolutionfunction_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=21, wstep=100.0, oversampling=4,
                                                    support=100, use_aaf=True)
        
        assert numpy.max(numpy.abs(cf.data)) > 0.0
        export_image_to_fits(gcf, "%s/test_convolutionfunction_awterm_gcf.fits" % self.dir)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_cf.fits" % self.dir)

        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert peak_location == (0, 0, 10, 2, 2, 50, 50), peak_location
        assert numpy.abs(cf.data[peak_location] - (0.07761529943522588-0j)) < 1e-7, \
            cf.data[peak_location]
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak

        bboxes = calculate_bounding_box_convolutionfunction(cf)
        assert len(bboxes) == 21, len(bboxes)
        assert len(bboxes[0]) == 3, len(bboxes[0])
        assert bboxes[-1][0] == 20, bboxes[-1][0]
        
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert peak_location == (0, 0, 10, 2, 2, 50, 50), peak_location
        
        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=1e-4)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 10, 2, 2, 29, 29), peak_location
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_awterm_clipped_cf.fits" % self.dir)

    def test_fill_aterm_to_convolutionfunction(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.image)
        export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=1, wstep=1e-7, oversampling=16,
                                                    support=32, use_aaf=True)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_cf.fits" % self.dir)
    
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - 0.07761522554113436-0j) < 1e-7, cf.data[peak_location]
        assert peak_location == (0, 0, 0, 8, 8, 16, 16), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
    
        export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_gcf.fits" % self.dir)
    
        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=0.001)
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_clipped_cf.fits" % self.dir)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 0, 8, 8, 6, 6), peak_location

    def test_compare_aterm_kernels(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        _, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, oversampling=16, support=32,
                                                  use_aaf=True)
        cf.data = numpy.real(cf.data)
        _, cf_noover = create_awterm_convolutionfunction(self.image, make_pb=make_pb, oversampling=1,
                                                    support=32, use_aaf=True)
        cf_noover.data = numpy.real(cf_noover.data)
        cf.data[...] -= cf_noover.data[0,0,0,0,0]
        cf.data[...] /= numpy.max(cf_noover.data)
        assert numpy.abs(cf.data[0,0,0,8,8,16,16]) < 1e-6, cf.data[0,0,0,8,8,16,16]

    def test_compare_wterm_kernels(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.image)
        _, cf = create_awterm_convolutionfunction(self.image, nw=16, wstep=50, oversampling=16,
                                                    support=32, use_aaf=True)
        cf.data = numpy.real(cf.data)
        _, cf_noover = create_awterm_convolutionfunction(self.image, nw=16, wstep=50, oversampling=1,
                                                    support=32, use_aaf=True)
        cf_noover.data = numpy.real(cf_noover.data)
        cf.data[...] -= cf_noover.data[0,0,0,0,0]
        cf.data[...] /= numpy.max(cf_noover.data)
        assert numpy.abs(cf.data[0,0,0,8,8,16,16]) < 5e-6, cf.data[0,0,0,8,8,16,16]

    def test_fill_aterm_to_convolutionfunction_noover(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.image)
    
        export_image_to_fits(pb, "%s/test_convolutionfunction_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=1, wstep=1e-7, oversampling=1,
                                                    support=32, use_aaf=True)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_noover_cf.fits" % self.dir)
    
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - 0.0776153022780847+0j) < 1e-7, cf.data[peak_location]
        assert peak_location == (0, 0, 0, 0, 0, 16, 16), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak
    
        export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_noover_gcf.fits" % self.dir)
    
        cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=0.001)
        cf_image = convert_convolutionfunction_to_image(cf_clipped)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_clipped_noover_cf.fits" % self.dir)
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf_clipped.data)), cf_clipped.shape)
        assert peak_location == (0, 0, 0, 0, 0, 5, 5), peak_location


if __name__ == '__main__':
    unittest.main()
