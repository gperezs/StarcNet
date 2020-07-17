# StarcNet: Machine Learning for Star Cluster Classification

Pytorch code for classification of star clusters from galaxy images by the Hubble Space Telescope (HST). The target galaxies used in this project are provided by the [Legacy ExtraGalactic UV Survey (LEGUS)](https://archive.stsci.edu/prepds/legus/). 

The approximate running time of StarcNet in a Galaxy with around 3,000 objects is ~4 mins (~4 secs with a GPU). 

![title_image](title_im.jpg)

## Getting started

### Prerequisites

**1. Installing Git:** To install Git follow the instructions [Here](https://gist.github.com/derhuerst/1b15ff4652a867391f03).

**2. Downloading the repository:** To download this repository run:
```
git clone https://github.com/gperezs/StarcNet.git
cd StarcNet
```

**3. Installing Anaconda:** We recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version.

**4. Anaconda Virtual environment:** To set up a virtual environment, run:
```
conda create -n starcnet python=3.*
```

To activate and enter the environment, run:
```
source activate starcnet
```

To install requirements, run:
```
conda install --yes --file requirements.txt 
```

**5. PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).

-------------------------
## Run StarcNet

StarcNet will classify objects from a galaxy or list of galaxies in `targets.txt`. It can run using catalogs saved locally or using online LEGUS catalogs. The original code comes ready to download catalog and classify star clusters from galaxy NGC1566. StarcNet predictions are saved into `output/predictions.csv`.

To run StarcNet demo with NGC1566:

```
bash run_starcnet.sh 1
```

### Run StarcNet with local data

1. Save the 5 mosaic's `.FITS` files into `legus/frc_fits_files/` folder.
2. Save catalog `.tab` file into `legus/tab_files/`.
3. Name of galaxy should be in `targets.txt`.
4. Name of the mosaic(s) files should be in `frc_fits_links.txt`.
5. Name of .tab file with the cluster catalog(s) (with object coordinates) should be in `tab_links.txt`.
6. Run `bash run_starcnet.sh`


### Run StarcNet with online LEGUS catalogs

1. Name of galaxy should be in `targets.txt`.
2. Links to the mosaic(s) files should be in `frc_fits_links.txt`.
3. Links to the cluster catalog(s) (with objects coordinates) should be in `tab_links.txt`.
4. Run `bash run_starcnet.sh 1`


### Preditions visualization

To produce the galaxy image with predictions run:
```
python src/run_visualization.py
```

The output visualization is saved into `output/predictions.png`.

-------------------------

### Acknowledgements

This work is supported by the [National Science Foundation (NSF)](https://nsf.gov/index.jsp) of the United States under the award [\#1815267](https://nsf.gov/awardsearch/showAward?AWD_ID=1815267).
