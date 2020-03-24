"""
MSkyHough
=====
MSkyHough: A user-friendly Search Management for Sky-Hough.
The aim of MSkyHough is to provide user friendly interface to perform searches
for the Sky-Hough pipeline. It is primarily designed and built to define search
set-ups and run them in the LVC clustes.
It provides tools to generate the necesary graphical interpretation of the status
of the search as well as from the results.
It allows to perform searches, injections and generate the tau statistics.
The code, and many examples are hosted at https://git.ligo.org/miquel.oliver/MSkyHough
For installation instructions see
...
"""

from __future__ import absolute_import

#from . import folder,...,...

from .phenom import phenom_simulator
from .siers import siers_simulator
from .utilitis import data_loader
from Simulations_COVID19 import utilitis

__version__ = utilitis.get_version_information()
