"""Unit tests for pipelines expressed via dask.delayed


"""

import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from tests.workflows import ARLExecuteTestCase
from workflows.arlexecute.calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from wrappers.arlexecute.calibration.calibration_control import create_calibration_controls
from wrappers.arlexecute.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.simulation.testing_support import ingest_unittest_visibility
from processing_components.simulation.configurations import create_named_configuration
from wrappers.arlexecute.simulation.testing_support import simulate_gaintable
from wrappers.arlexecute.visibility.base import copy_visibility

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestCalibrateGraphs(unittest.TestCase):
    
    def setUp(self):
        client = get_dask_Client(memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        arlexecute.set_client(client)
    
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
        self.blockvis_list = [arlexecute.execute(ingest_unittest_visibility, nout=1)(self.low,
                                                                                     [self.frequency[i]],
                                                                                     [self.channelwidth[i]],
                                                                                     self.times,
                                                                                     self.vis_pol,
                                                                                     self.phasecentre, block=True,
                                                                                     zerow=zerow)
                              for i in range(nfreqwin)]
        self.blockvis_list = arlexecute.compute(self.blockvis_list, sync=True)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 1.0 + 0.0j
        
        self.error_blockvis_list = [arlexecute.execute(copy_visibility(v)) for v in self.blockvis_list]
        gt = arlexecute.execute(create_gaintable_from_blockvisibility)(self.blockvis_list[0])
        gt = arlexecute.execute(simulate_gaintable)(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1,
                                                    leakage=0.0, seed=180555)
        self.error_blockvis_list = [arlexecute.execute(apply_gaintable)(self.error_blockvis_list[i], gt)
                                    for i in range(self.freqwin)]
        
        self.error_blockvis_list = arlexecute.compute(self.error_blockvis_list, sync=True)
    
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 0.0
    
    def test_time_setup(self):
        self.actualSetUp()

    def test_calibrate_arlexecute(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
    
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'
    
        calibrate_list = \
            calibrate_list_arlexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                               calibration_context='T', controls=controls, do_selfcal=True,
                                               global_solution=False)
        calibrate_list = arlexecute.compute(calibrate_list, sync=True)
    
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        assert numpy.max(numpy.abs(calibrate_list[0][0].vis - self.blockvis_list[0].vis)) < 2e-6

    def test_calibrate_arlexecute_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)

        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j

        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'

        calibrate_list = \
            calibrate_list_arlexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                               calibration_context='T', controls=controls, do_selfcal=True,
                                               global_solution=False)
        calibrate_list = arlexecute.compute(calibrate_list, sync=True)
        assert len(calibrate_list[1][0]) > 0

    def test_calibrate_arlexecute_global(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'
        
        calibrate_list = \
            calibrate_list_arlexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                               calibration_context='T', controls=controls, do_selfcal=True,
                                               global_solution=True)

        calibrate_list = arlexecute.compute(calibrate_list, sync=True)

        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].vis - self.blockvis_list[0].vis))
        assert err < 2e-6, err

    def test_calibrate_arlexecute_global_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
    
        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j
    
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timescale'] = 'auto'
    
        calibrate_list = \
            calibrate_list_arlexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                               calibration_context='T', controls=controls, do_selfcal=True,
                                               global_solution=True)

        calibrate_list = arlexecute.compute(calibrate_list, sync=True)
        assert len(calibrate_list[1][0]) > 0

if __name__ == '__main__':
    unittest.main()
