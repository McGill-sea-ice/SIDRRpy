# SIDDRR.py
Python utility to open and manipulate SIDRR data. 

## Overview

This repository shares code to load, manipulate and analyse coverage characteristics of the Sea Ice Deformation and Rotation Rates (SIDRR) dataset.
This code is used to produce figures and analysis in Plante et al. (2024).

## Installation

Start by cloning the repository:

```bash
# Check if your SSH key is configured on GitHub
ssh -T git@github.com
# Clone the project
git clone git@github.com:McGill-sea-ice/SIDRR.py.git
```

This project uses a [**conda environment**][conda]. Start by accessing the project folder:

[conda]: https://docs.conda.io/en/latest/miniconda.html

```bash
cd SIDRRpy
```

Create and activate the project's environment (this installs the dependencies):

```bash
conda env create -f environment.yaml
conda activate icetrackdefs
```

Install the Cartopy shapefiles (this would be done automatically by Cartopy, but the URL hardcoded in the Cartopy version we used to require is [out of service][1]):
~~~bash
conda activate icetrackdefs
wget -q https://raw.githubusercontent.com/SciTools/cartopy/master/tools/cartopy_feature_download.py -O $CONDA_PREFIX/bin/cartopy_feature_download.py
python cartopy_feature_download.py physical
~~~


[1]: https://github.com/SciTools/cartopy/pull/1833
## Usage

In order to generate analysis of the SIDRR dataset, modify the namelist (namelist.ini), and use the following commands:

```bash
# Activate the virtual environment
conda activate icetrackdefs
# Launch the code
python DatasetAnalysis.py
# Deactivate the environment when you are done
conda deactivate
```

## Input Data Location

The SIDRR dataset can be found on Zenodo (NOT YET!)

