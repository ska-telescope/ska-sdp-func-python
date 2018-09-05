""" Unit tests for image operations


"""
import functools
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from processing_components.convolution_function.kernels import create_awterm_convolutionfunction, \
    create_pswf_convolutionfunction, create_box_convolutionfunction
from processing_components.convolution_function.operations import convert_convolutionfunction_to_image
from processing_components.griddata.gridding import grid_visibility_to_griddata, \
    fft_griddata_to_image, fft_image_to_griddata, \
    degrid_visibility_from_griddata
from processing_components.griddata.operations import create_griddata_from_image
from processing_components.image.operations import export_image_to_fits
from processing_components.image.operations import smooth_image
from processing_components.imaging.base import normalize_sumwt
from processing_components.imaging.base import predict_skycomponent_visibility
from processing_components.imaging.primary_beams import create_pb_generic
from processing_components.simulation.testing_support import create_named_configuration, create_unittest_model, \
    create_unittest_components, ingest_unittest_visibility
from processing_components.skycomponent.operations import insert_skycomponent
from processing_components.visibility.operations import qa_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

log = logging.getLogger(__name__)


class TestGridDataGridding(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def actualSetUp(self, zerow=True):
        self.npixel = 512
        self.cellsize = 0.0009
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = 1
        self.vis_list = list()
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        
        if self.freqwin == 1:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([4e7])
        else:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        
        self.vis_pol = PolarisationFrame('linear')
        self.image_pol = PolarisationFrame('stokesIQUV')
        
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        
        flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low,
                                              self.frequency,
                                              self.channelwidth,
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre,
                                              block=False,
                                              zerow=zerow)
        
        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=self.cellsize,
                                           npixel=self.npixel, nchan=self.freqwin)
        self.components = create_unittest_components(self.model, flux, applypb=False)
        self.model = insert_skycomponent(self.model, self.components)
        self.vis = predict_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_gridding_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel.fits' % self.dir)
        pb = create_pb_generic(self.model, diameter=35.0, blockage=0.0)
        self.cmodel.data *= pb.data
        export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel_pb.fits' % self.dir)
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_griddata_invert_pswf(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=128)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_pswf.fits' % self.dir)
        self.check_peaks(im, 98.93575439937361, tol=1e-7)
    
    def test_griddata_invert_pswf_w(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=32)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_pswf_w.fits' % self.dir)
        self.check_peaks(im, 99.84916864175699, tol=1e-7)
    
    def test_griddata_invert_aterm(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.model)
        export_image_to_fits(pb, "%s/test_gridding_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=1, oversampling=16, support=16,
                                                    use_aaf=False)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_aterm.fits' % self.dir)
        self.check_peaks(im, 99.0672945056528, tol=1e-7)
    
    def test_griddata_invert_aterm_noover(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.model)
        export_image_to_fits(pb, "%s/test_gridding_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=1, oversampling=1, support=16,
                                                    use_aaf=True)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_aterm_noover.fits' % self.dir)
        self.check_peaks(im, 99.06949151553816)
    
    def test_griddata_invert_pswf_nooversampling(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=1)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_pswf_nooversampling.fits' % self.dir)
        self.check_peaks(im, 99.0694915155382, tol=1e-7)
    
    def test_griddata_invert_box(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_box_convolutionfunction(self.model)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_box.fits' % self.dir)
        self.check_peaks(im, 99.06949151553818, tol=1e-7)
    
    def test_griddata_invert_fast(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_box_convolutionfunction(self.model)
        griddata = create_griddata_from_image(self.model)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_fast.fits' % self.dir)
        self.check_peaks(im, 99.06949151553818, tol=1e-7)
    
    def check_peaks(self, im, peak=99.6754, tol=1e-3):
        assert numpy.abs(im.data[(0, 0, self.npixel // 2, self.npixel // 2)] - peak) < tol, \
            im.data[(0, 0, self.npixel // 2, self.npixel // 2)]
    
    def test_griddata_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0, oversampling=4, support=50,
                                                    use_aaf=True)
        
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_gridding_wterm_cf.fits" % self.dir)
        
        griddata = create_griddata_from_image(self.model, nw=1)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_wterm.fits' % self.dir)
        self.check_peaks(im, 99.96890306008716)
    
    def test_griddata_invert_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.model)
        export_image_to_fits(pb, "%s/test_gridding_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0, oversampling=4,
                                                    support=100, use_aaf=True)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_gridding_awterm_cf.fits" % self.dir)
        
        griddata = create_griddata_from_image(self.model, nw=100, wstep=8.0)
        griddata, sumwt = grid_visibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        im = fft_griddata_to_image(griddata, gcf)
        im = normalize_sumwt(im, sumwt)
        export_image_to_fits(im, '%s/test_gridding_dirty_awterm.fits' % self.dir)
        self.check_peaks(im, 99.96885125478173)
    
    def test_griddata_predict_pswf(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_pswf_convolutionfunction(self.model, support=6, oversampling=128)
        griddata = create_griddata_from_image(self.model)
        griddata = fft_image_to_griddata(self.model, griddata, gcf)
        newvis = degrid_visibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert numpy.abs(qa.data['rms'] - 163.47064019979751) < 1e-3, str(qa)
        import matplotlib.pyplot as plt
        r = numpy.sqrt(newvis.u**2+newvis.v**2)
        for pol in range(4):
            plt.plot(r, numpy.abs(newvis.vis[:,pol]))
        plt.title('pswf')
        plt.show()

    def test_griddata_predict_box(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_box_convolutionfunction(self.model)
        griddata = create_griddata_from_image(self.model)
        griddata = fft_image_to_griddata(self.model, griddata, gcf)
        newvis = degrid_visibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert numpy.abs(qa.data['rms'] - 192.28683248691866) < 1e-3, str(qa)
        import matplotlib.pyplot as plt
        r = numpy.sqrt(newvis.u**2+newvis.v**2)
        for pol in range(4):
            plt.plot(r, numpy.abs(newvis.vis[:,pol]))
        plt.title('box')
        plt.show()

    def test_griddata_predict_aterm(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        griddata = create_griddata_from_image(self.model)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=1,
                                                    oversampling=16, support=16,
                                                    use_aaf=True)
        griddata = fft_image_to_griddata(self.model, griddata, gcf)
        newvis = degrid_visibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert numpy.abs(qa.data['rms'] - 44.08765413622353) < 1e-3, str(qa)
        import matplotlib.pyplot as plt
        r = numpy.sqrt(newvis.u**2+newvis.v**2)
        for pol in range(4):
            plt.plot(r, numpy.abs(newvis.vis[:,pol]))
        plt.title('aterm')
        plt.show()

    def test_griddata_predict_wterm(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0, oversampling=4, support=50,
                                                    use_aaf=True)

        griddata = create_griddata_from_image(self.model, nw=100, wstep=8.0)
        griddata = fft_image_to_griddata(self.model, griddata, gcf)
        newvis = degrid_visibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert numpy.abs(qa.data['rms'] - 163.3050979060864) < 1e-3, str(qa)
        import matplotlib.pyplot as plt
        r = numpy.sqrt(newvis.u**2+newvis.v**2)
        for pol in range(4):
            plt.plot(r, numpy.abs(newvis.vis[:,pol]))
        plt.title('wterm')
        plt.show()


    def test_griddata_predict_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0)
        pb = make_pb(self.model)
        export_image_to_fits(pb, "%s/test_gridding_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                    oversampling=4, support=50, use_aaf=True)
        griddata = create_griddata_from_image(self.model, nw=100, wstep=8.0)
        griddata = fft_image_to_griddata(self.model, griddata, gcf)
        newvis = degrid_visibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert numpy.abs(qa.data['rms'] - 43.199054679082124) < 1e-3, str(qa)
        import matplotlib.pyplot as plt
        r = numpy.sqrt(newvis.u**2+newvis.v**2)
        for pol in range(4):
            plt.plot(r, numpy.abs(newvis.vis[:,pol]))
        plt.title('awterm')
        plt.show()


if __name__ == '__main__':
    unittest.main()
