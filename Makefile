include .make/base.mk
include .make/python.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-func-python

# flake8 switches
# W503: line break before binary operator
# E203: whitespace before ':'
# E501: line too long
# These need to be eventually addressed
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=W503,E203,E501

# W0511: fixme (don't report TODOs)
# R0801: duplicate-code (some are duplicated between the main function and utils
#		 these will eventually need to be resolved
PYTHON_SWITCHES_FOR_PYLINT = --disable=W0511,R0801


