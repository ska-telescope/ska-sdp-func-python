""" Unit tests for quality assessment


"""
import logging
import unittest

from ska_sdp_datamodels import QualityAssessment

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestQualityAssessment(unittest.TestCase):
    def test_qa(self):
        qa = QualityAssessment(
            origin="foo", data={"rms": 100.0, "median": 10.0}, context="test of qa"
        )
        log.debug(str(qa))


if __name__ == "__main__":
    unittest.main()
