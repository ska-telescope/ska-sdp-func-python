"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from workflows.serial.calibration.calibration_serial import calibrate_list_serial_workflow
from wrappers.serial.calibration.calibration_control import create_calibration_controls
from wrappers.serial.imaging.base import predict_skycomponent_visibility
from wrappers.serial.simulation.testing_support import create_named_configuration, ingest_unittest_visibility, \
    create_unittest_components, insert_unittest_errors, create_unittest_model
from wrappers.serial.visibility.base import copy_visibility
from wrappers.serial.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestCalibrateGraphs(unittest.TestCase):
    
    def setUp(self):
        from data_models.parameters import arl_path
        self.dir = arl_path('test_results')
    
    def tearDown(self):
        pass
    
    def actualSetUp(self, nfreqwin=3, dospectral=True, dopol=False,
                    amp_errors=None, phase_errors=None, zerow=True):
        
        if amp_errors is None:
            amp_errors = {'T': 0.0, 'G': 0.1}
        if phase_errors is None:
            phase_errors = {'T': 1.0, 'G': 0.0}
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.vis_list = list()
        self.ntimes = 1
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if self.freqwin > 1:
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
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
        self.blockvis_list = [ingest_unittest_visibility(self.low,
                                                         [self.frequency[i]],
                                                         [self.channelwidth[i]],
                                                         self.times,
                                                         self.vis_pol,
                                                         self.phasecentre, block=True,
                                                         zerow=zerow)
                              for i in range(nfreqwin)]
        
        self.vis_list = [convert_blockvisibility_to_visibility(bv) for bv in
                         self.blockvis_list]
        
        self.model_imagelist = [create_unittest_model
                                (self.vis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
                                for i in range(nfreqwin)]
        
        self.components_list = [create_unittest_components
                                (self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :])
                                for freqwin, m in enumerate(self.model_imagelist)]
        
        self.blockvis_list = [predict_skycomponent_visibility
                              (self.blockvis_list[freqwin], self.components_list[freqwin])
                              for freqwin, _ in enumerate(self.blockvis_list)]
        
        self.error_blockvis_list = [copy_visibility(v) for v in self.blockvis_list]
        self.error_blockvis_list = [insert_unittest_errors
                                    (self.error_blockvis_list[i], amp_errors=amp_errors, phase_errors=phase_errors,
                                     calibration_context="TG")
                                    for i in range(self.freqwin)]
        
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 1.0
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_calibrate_serial(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                           calibration_context='T', controls=controls, do_selfcal=True,
                                           global_solution=False)
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 1e-3
    
    def test_calibrate_serial_global(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'
        
        calibrate_list = \
            calibrate_list_serial_workflow(self.error_blockvis_list, self.blockvis_list,
                                           calibration_context='T', controls=controls, do_selfcal=True,
                                           global_solution=True)
        
        assert numpy.max(calibrate_list[1]['T'].residual) < 7e-6, numpy.max(calibrate_list[1]['T'].residual)
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 1e-3
        assert numpy.max(calibrate_list[1]['T'].residual) < 1e-6, numpy.max(calibrate_list[1]['T'].residual)


if __name__ == '__main__':
    unittest.main()
