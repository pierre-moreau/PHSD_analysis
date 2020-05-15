# PHSD_analysis

To analyze and plot heavy-ion collision data produced by the [Parton-Hadron-String Dynamics (PHSD)](https://theory.gsi.de/~ebratkov/phsd-project/PHSD/index1.html) model.

## Basic usage

This package can be installed with: ``pip install .`` along with all the necessary packages indicated in [requirements.txt](requirements.txt). 

It can also be used without installation: ``python -m PHSD_analysis [--help]``. 

The ``-h, --help`` argument details how to modify the parameters of the analysis, such as the midrapidity region, or the binning of particle spectra. In particular, the ``--folder`` argument indicates in which folder to look for the ``inputPHSD`` and ``phsd.dat`` files. The folder argument can also contain several PHSD simulations, each of them contained in a ``job_[0-9+]`` folder.