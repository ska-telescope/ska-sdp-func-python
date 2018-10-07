""" Unit tests for pipelines
"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame

from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow, invert_list_serial_workflow, \
    subtract_list_serial_workflow, zero_list_serial_workflow

from wrappers.serial.image.operations import export_image_to_fits, smooth_image
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent
from wrappers.serial.simulation.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_model, insert_unittest_errors, create_unittest_components
from wrappers.serial.griddata.kernels import create_awterm_convolutionfunction
from wrappers.serial.griddata.convolution_functions import apply_bounding_box_convolutionfunction

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging(unittest.TestCase):
    def setUp(self):
        
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def tearDown(self):
        pass
    
    def actualSetUp(self, add_errors=False, freqwin=3, block=False, dospectral=True, dopol=False, zerow=False,
                    makegcfcf=False):
        
        self.npixel = 256
        self.cellsize = 0.0005
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 11
        self.times = numpy.linspace(-2.0, +2.0, self.ntimes) * numpy.pi / 12.0
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis_list = [ingest_unittest_visibility(self.low,
                                                    [self.frequency[freqwin]],
                                                    [self.channelwidth[freqwin]],
                                                    self.times,
                                                    self.vis_pol,
                                                    self.phasecentre, block=block,
                                                    zerow=zerow)
                         for freqwin, _ in enumerate(self.frequency)]
        
        self.model_list = [create_unittest_model(self.vis_list[freqwin],
                                                 self.image_pol,
                                                 npixel=self.npixel,
                                                 cellsize=self.cellsize)
                           for freqwin, _ in enumerate(self.frequency)]
        
        self.components_list = [create_unittest_components(self.model_list[freqwin],
                                                           flux[freqwin, :][numpy.newaxis, :], single=True)
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.model_list = [insert_skycomponent(self.model_list[freqwin],
                                               self.components_list[freqwin])
                           for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_list = [predict_skycomponent_visibility(self.vis_list[freqwin],
                                                         self.components_list[freqwin])
                         for freqwin, _ in enumerate(self.frequency)]
        centre=self.freqwin // 2
        # Calculate the model convolved with a Gaussian.
        self.model = self.model_list[centre]
        
        self.cmodel = smooth_image(self.model)
        export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
        export_image_to_fits(self.cmodel, '%s/test_imaging_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_list = [insert_unittest_errors(self.vis_list[i])
                             for i, _ in enumerate(self.frequency)]
        
        self.components = self.components_list[centre]

        if makegcfcf:
            self.gcfcf = [create_awterm_convolutionfunction(self.model, nw=121, wstep=8.0,
                                                           oversampling=8,
                                                           support=60,
                                                           use_aaf=True)]
            self.gcfcf_clipped = [(self.gcfcf[0][0], apply_bounding_box_convolutionfunction(self.gcfcf[0][1],
                                                                                       fractional_level=1e-3))]

            self.gcfcf_joint = [create_awterm_convolutionfunction(self.model, nw=21, wstep=8.0,
                                                                 oversampling=8,
                                                                 support=60,
                                                                 use_aaf=True)]
            
        else:
            self.gcfcf = None
            self.gcfcf_clipped = None
            self.gcfcf_joint = None
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=1.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.components)
            assert separation / cellsize < positionthreshold, "Component differs in position %.3f pixels" % \
                                                              separation / cellsize
    
    def _predict_base(self, context='2d', extra='', fluxthreshold=1.0, facets=1, vis_slices=1,
                      gcfcf=None, **kwargs):
        
        centre=self.freqwin // 2
        vis_list = zero_list_serial_workflow(self.vis_list)
        vis_list = predict_list_serial_workflow(vis_list, self.model_list, context=context,
                                                vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
        vis_list = subtract_list_serial_workflow(self.vis_list, vis_list)
        
        dirty = invert_list_serial_workflow(vis_list, self.model_list, context='2d', dopsf=False,
                                            gcfcf=gcfcf, normalize=True)[centre]
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Residual image is empty"
        export_image_to_fits(dirty[0], '%s/test_imaging_predict_%s%s_serial_dirty.fits' %
                             (self.dir, context, extra))
        
        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, context, extra='', fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     facets=1, vis_slices=1, gcfcf=None, **kwargs):
        
        centre = self.freqwin // 2
        dirty = invert_list_serial_workflow(self.vis_list, self.model_list, context=context,
                                            dopsf=False, normalize=True, facets=facets, vis_slices=vis_slices,
                                            gcfcf=gcfcf, **kwargs)[centre]
        
        export_image_to_fits(dirty[0], '%s/test_imaging_invert_%s%s_serial_dirty.fits' %
                             (self.dir, context, extra))
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"
        
        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)

    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(context='2d')

    @unittest.skip("Facets need overlap")
    def test_predict_facets(self):
        self.actualSetUp()
        self._predict_base(context='facets', fluxthreshold=17.0, facets=4)

    @unittest.skip("Timeslice predict needs better interpolation and facets need overlap")
    def test_predict_facets_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='facets_timeslice', fluxthreshold=19.0, facets=8, vis_slices=self.ntimes)

    @unittest.skip("Facets need overlap")
    def test_predict_facets_wprojection(self, makegcfcf=True):
        self.actualSetUp()
        self._predict_base(context='facets', extra='_wprojection', facets=8, fluxthreshold=15.0,
                           gcfcf=self.gcfcf_joint)

    @unittest.skip("Facets need overlap")
    def test_predict_facets_wstack(self):
        self.actualSetUp()
        self._predict_base(context='facets_wstack', fluxthreshold=15.0, facets=8, vis_slices=101)

    def test_predict_timeslice(self):
        self.actualSetUp()
        self._predict_base(context='timeslice', fluxthreshold=17.0, vis_slices=self.ntimes)

    def test_predict_timeslice_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='timeslice', extra='_wprojection', fluxthreshold=17.0,
                           vis_slices=self.ntimes, gcfcf=self.gcfcf_joint)

    def test_predict_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='2d', extra='_wprojection', fluxthreshold=3.5,
                           gcfcf=self.gcfcf)

    def test_predict_wprojection_clip(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='2d', extra='_wprojection_clipped', fluxthreshold=3.5,
                           gcfcf=self.gcfcf_clipped)

    def test_predict_wstack(self):
        self.actualSetUp()
        self._predict_base(context='wstack', fluxthreshold=2.0, vis_slices=101)

    def test_predict_wstack_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._predict_base(context='wstack', extra='_wprojection', fluxthreshold=8.0, vis_slices=11,
                           gcfcf=self.gcfcf)

    def test_predict_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=101)

    def test_predict_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._predict_base(context='wstack', extra='_spectral', fluxthreshold=4.0, vis_slices=101)

    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(context='2d', positionthreshold=2.0, check_components=False)

    @unittest.skip("Facets need overlap")
    def test_invert_facets(self):
        self.actualSetUp()
        self._invert_base(context='facets', positionthreshold=2.0, check_components=True, facets=8)

    @unittest.skip("Facets need overlap")
    def test_invert_facets_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='facets_timeslice', check_components=True, vis_slices=self.ntimes,
                          positionthreshold=5.0, flux_threshold=1.0, facets=8)

    @unittest.skip("Facets need overlap")
    def test_invert_facets_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='facets', extra='_wprojection', check_components=True,
                          positionthreshold=2.0, facets=4, gcfcf=self.gcfcf)

    @unittest.skip("Facets need overlap")
    def test_invert_facets_wstack(self):
        self.actualSetUp()
        self._invert_base(context='facets_wstack', positionthreshold=1.0, check_components=False, facets=4,
                          vis_slices=101)

    def test_invert_timeslice(self):
        self.actualSetUp()
        self._invert_base(context='timeslice', positionthreshold=1.0, check_components=True,
                          vis_slices=self.ntimes)

    def test_invert_timeslice_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='timeslice', extra='_wprojection', positionthreshold=1.0,
                          check_components=True, vis_slices=self.ntimes // 2, gcfcf=self.gcfcf_joint)

    def test_invert_wprojection(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='2d', extra='_wprojection', positionthreshold=2.0, gcfcf=self.gcfcf)

    def test_invert_wprojection_clip(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='2d', extra='_wprojection_clipped', positionthreshold=2.0,
                          gcfcf=self.gcfcf_clipped)

    def test_invert_wprojection_wstack(self):
        self.actualSetUp(makegcfcf=True)
        self._invert_base(context='wstack', extra='_wprojection', positionthreshold=1.0, vis_slices=11,
                          gcfcf=self.gcfcf_joint)

    def test_invert_wstack(self):
        self.actualSetUp()
        self._invert_base(context='wstack', positionthreshold=1.0, vis_slices=101)

    def test_invert_wstack_spectral(self):
        self.actualSetUp(dospectral=True)
        self._invert_base(context='wstack', extra='_spectral', positionthreshold=2.0,
                          vis_slices=101)

    def test_invert_wstack_spectral_pol(self):
        self.actualSetUp(dospectral=True, dopol=True)
        self._invert_base(context='wstack', extra='_spectral_pol', positionthreshold=2.0,
                          vis_slices=101)

    def test_zero_list(self):
        self.actualSetUp()
        
        centre = self.freqwin // 2
        vis_list = zero_list_serial_workflow(self.vis_list)
        
        assert numpy.max(numpy.abs(vis_list[centre].vis)) < 1e-15, numpy.max(numpy.abs(vis_list[centre].vis))
    
        predicted_vis_list = [predict_skycomponent_visibility(vis_list[freqwin], self.components_list[freqwin])
                              for freqwin, _ in enumerate(self.frequency)]
        assert numpy.max(numpy.abs(predicted_vis_list[centre].vis)) > 0.0, \
            numpy.max(numpy.abs(predicted_vis_list[centre].vis))

        diff_vis_list = subtract_list_serial_workflow(self.vis_list, predicted_vis_list)
        
        assert numpy.max(numpy.abs(diff_vis_list[centre].vis)) < 1e-15, numpy.max(numpy.abs(diff_vis_list[centre].vis))


if __name__ == '__main__':
    unittest.main()
