include .make/base.mk
include .make/python.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-func-python

# flake8 switches
# W503: line break before binary operator
# E203: whitespace before ':'
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=W503,E203

# W0511: fixme (don't report TODOs)
# R0801: duplicate-code (some are duplicated between the main function and utils
#		 these will eventually need to be resolved
# C0103: invalid-name
# The following needs major refactor work to be fixed
# R0914: too-many-locals
# R0913: too-many-arguments
# R0912: too-many-branches
# TODO: review all these!!
PYTHON_SWITCHES_FOR_PYLINT = --disable=W0511,R0801,C0103,R0914,R0913,R0912,R1702,W0404,W0621,E0611,C0301,W0631,W0632,R0915,W0707,W0105,W1202,W0201,C0115,R0902,R1735,W0622


