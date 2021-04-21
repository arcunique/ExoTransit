ExoTransit
======

[![Build Status](https://img.shields.io/badge/release-1.0.0-orange)](https://github.com/arcunique/Cplotter)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-371/)

ExoTransit is a Python package for retrieval of the properties of an exoplanet by modeling its transit light curves 
using Markov chain Monte Carlo technique
on the basis of the transit model of Eric & Agol (2002). The code also allows simultaneous modeling of correlated noise
with Gaussian process (GP) regression. The code also allows users to detrend the light curves simultaneously by choosing
arbitrary detrending variables and trend functions. 

One of the greatest fetaure of the code is that it has option for monitoring the progress of walkers over iterations.
Also users can set the program to stop automatically by calculating the auto-correlation time.

Author
------
* Aritra Chakrabarty (IIA, Bangalore)

Requirements
------------
* python>3.6
* numpy
* matplotlib 
* scipy
* emcee
* george

Instructions on installation and use
------------------------------------
Presently, the code is only available on [Github](https://github.com/arcunique/ExoTransit). Either download the code or
use the following line on terminal to install using pip:\
pip install git+https://github.com/arcunique/ExoTransit  #installs from the current master on this repo.

To use the retrieval function use the __model_transit_lightcurve__ class defined. It can be imported by

from ExoTransit.retrieve import model_transit_lightcurve

Documentation of this package is underway. Some example Jupyter notebooks can be found in the __run2learn__ directory of this 
package which demonstrate how to use the classes and functions. This package has already been 
used to perform modeling of transit light curves of some hot Jupiters, results from which can be found to be 
demonstrated in [here](https://doi.org/10.3847/1538-3881/ab24dd).






