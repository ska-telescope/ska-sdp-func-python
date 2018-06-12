""" Unit tests for json helpers


"""
import logging
import unittest

from data_models.parameters import arl_path
from processing_components.external_interface.execution_helper import initialise_config_wrapper, \
    initialise_logging_wrapper

class TestWrappers(unittest.TestCase):
    
    def test_initialise_config(self):
        config = initialise_config_wrapper(arl_path("tests/processing_components/test_json_helpers.json"))
        for key in ['execute', 'component', 'logging', 'inputs', 'outputs', 'imaging', 'image', 'deconvolution',
                    'create_vislist']:
            assert key in config.keys(), "Key %s not in configuration"
            
        log = logging.getLogger(__name__)
        initialise_logging_wrapper(config)

        log.info('Test message')
        log.info(str(config))
        
if __name__ == '__main__':
    unittest.main()
