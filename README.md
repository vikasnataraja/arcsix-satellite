# Satellite Pipeline for NASA ARCSIX 2024 Aircraft Campaign

<p align="center">
  <img src="https://espo.nasa.gov/sites/default/files/images/final_arcsix_logo_p3.png" alt="NASA ARCSIX Logo" width="225" height="200"/>
</p>

Imagery and videos are published at: https://lasp.colorado.edu/data/store/airs/arcsix/

NASA ARCSIX: https://espo.nasa.gov/arcsix/

EaR³T (Education and Research 3D Radiative Transfer Toolbox) recommended for downloading satellite data: https://github.com/hong-chen/er3t

## About
This repository is the processing pipeline for visualizing satellite imagery in near real time specifically tailored to the Arctic for the NASA ARCSIX 2024 Aircraft Campaign. Note that this is a tool meant for flight planning and decision making, and not meant to be used as a scientific data set. 

## How to Install
```
conda create --name arcsix --file conda_requirements.txt
conda activate arcsix
pip3 install -r pip_requirements.txt
```
Tested with Python 3.10.14 on RedHat Enterprise Linux 8. 

`conda` can be notoriously slow on different OS. Try other channels on `conda` or try `pip` instead. If none of those work, grab a coffee and let `conda` do its thing:-)

Note that downloading satellite files (MODIS and VIIRS) is not a part of this repository and assumes the user already has them downloaded. It is recommended to use <a href="https://github.com/hong-chen/er3t">EaR3T</a> to download satellite files. Visit that page, set up a separate environment in a different directory. Once setup of EaR3T is complete, run `sdown` (<a href="[https://github.com/hong-chen/er3t](https://er3t.readthedocs.io/en/latest/source/tutorial/tool.html#satellite-products-download-tool-sdown)">EaR3T Documentation</a>).

This repository expects the following file structure:
```
some_dir
├── arcsix-satellite
│    ├── data/
│    ├── *.py
│    ├── output/
|    │    ├── viz_region_1/
│    │    │    ├── true_color/
│    │    │    ├── false_color_721/
│    │    │    ├── false_color_367/
│    │    │    ├── false_color_ir/
│    │    │    ├── false_color_cirrus/
│    │    │    ├── water_path/
│    │    │    ├── ice_path/
│    │    │    ├── optical_thickness/
│    │    │    ├── cloud_phase/
│    │    │    ├── cloud_top_height_temperature/
│    │    ├── viz_region_2/
│    │    │    ├── true_color/
│    │    │    ├── false_color_721/
│    │    │    ├── false_color_367/
│    │    │    ├── false_color_ir/
│    │    │    ├── false_color_cirrus/
│    │    │    ├── water_path/
│    │    │    ├── ice_path/
│    │    │    ├── optical_thickness/
│    │    │    ├── cloud_phase/
│    │    │    ├── cloud_top_height_temperature/
│    │    ├── ...
├── er3t
│    ├── sat-data/
│    │   ├── region_1/
│    │   |    ├── 2024-06-20
│    │   |    ├── 2024-06-21
│    │   |    ├── ...
│    │   ├── region_2/
│    │   |    ├── 2024-06-20
│    │   |    ├── 2024-06-21
│    │   |    ├── ...
```

`arcsix-satellite/output/` is an example directory that will be created after the run and will contain all the Level 1 and Level 2 visualized png outputs.


## How to Run
Put on your Nike Air Max and assuming that the above folder structure is implemented and that the satellite files are in the appropriate directories (`sat-data`), run the following command:

```
python3 visualize_satellites.py --fdir /path/to/some_dir/er3t/sat-data/region_1 --outdir /path/to/some_dir/arcsix-satellite/output/viz_region_1/ --mode lincoln --nrt --ndir_recent 1 --buoys /path/to/some_dir/arcsix-satellite/data/buoys/buoys.json
```

## Benchmarks

*As of July 4, 2024*:

Average time taken for downloading and producing 10 different kinds of L1 and L2 imagery per overpass = 3 minutes


## Authors and Credits
<a href="mailto:Vikas.HanasogeNataraja@lasp.colorado.edu">Vikas Nataraja</a>, Laboratory for Atmospheric and Space Physics, Dept. of Atmospheric and Oceanic Sciences, University of Colorado Boulder

<a href="mailto:Hong.Chen@lasp.colorado.edu">Hong Chen</a>, Laboratory for Atmospheric and Space Physics, University of Colorado Boulder

<a href="mailto:Sebastian.Schmidt@lasp.colorado.edu">Sebastian Schmidt</a>, Laboratory for Atmospheric and Space Physics, Dept. of Atmospheric and Oceanic Sciences, University of Colorado Boulder

## Acknowledgements
Some of the functionalities, particularly in `arctic_gridding_utils.py`, have been adapted from <a href="https://github.com/hong-chen/er3t">EaR3T</a>.

This work utilized the Alpine high performance computing resource at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, and Colorado State University.

Data storage supported by the University of Colorado Boulder ‘PetaLibrary’.

Public server link created and maintained by the Laboratory for Atmospheric and Space Physics at the University of Colorado Boulder.


