
<h1 align="center">
  <br>
  <a><img width="200" src="logo.png" alt="StarcNet"></a>
</h1>

<h4 align="center">Machine Learning for Star Cluster Classification</h4>

<p align="center">
    <a href="https://github.com/gperezs/StarcNet/commits/master">
    <img src="https://img.shields.io/github/last-commit/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/gperezs/StarcNet/issues">
    <img src="https://img.shields.io/github/issues-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/gperezs/StarcNet/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests"></a>
</p>


PyTorch code for classification of star clusters from galaxy images
taken by the Hubble Space Telescope (HST) using StarcNet. 
StarcNet is a convolutional neural network (CNN) trained to classify
5-band galaxy images into four morphological classes. 
The target galaxies used in this project are provided by the [Legacy
ExtraGalactic UV Survey
(LEGUS)](https://archive.stsci.edu/prepds/legus/).
The running time of StarcNet in a Galaxy of 3,000
objects is about 4 mins on a CPU (4 secs with a GPU).

![title_image](title_im.jpg)


## Table of contents
* [Installing / Getting started](#installing-/-getting-started)
	* [Using Anaconda](#using-anaconda)
	* [Using Virtualenv](#using-virtualenv)
* [Run StarcNet](#run-starcnet)
	* [Using local data](#run-starcnet-with-local-data)
	* [Using LEGUS catalogs](#run-starcnet-with-online-legus-catalogs)
* [Acknowledgements](#acknowledgements)


## Installing / Getting started

1. **Clone the repository:** To download this repository run:
```
$ git clone https://github.com/gperezs/StarcNet.git
$ cd StarcNet
```

In the following sections we show two ways to setup StarcNet. Use the one that suits you best: 
* [Using virtualenv](#using-virtualenv)
* [Using Anaconda](#using-anaconda)

### Using virtualenv

2. **Install virtualenv:** To install virtualenv run after installing pip:

```
$ sudo pip3 install virtualenv 
```

3. **Virtualenv  environment:** To set up and activate the virtual environment,
run:
```
$ virtualenv -p /usr/bin/python3 venv3
$ source venv3/bin/activate
```

To install requirements, run:
```
$ pip install -r requirements.txt
```

4. **PyTorch:** To install pytorch run:
```
$ pip install torch torchvision
```

-------
### Using Anaconda

2. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
$ conda create -n starcnet python=3.*
$ source activate starcnet
```

To install requirements, run:
```
$ conda install --yes --file requirements.txt
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).


## Run StarcNet

StarcNet will classify objects from a single galaxy or list of
galaxies in `targets.txt`. 
It can run using catalogs saved locally or using online LEGUS
catalogs. 
The code comes ready to download catalog and classify star clusters
from galaxy NGC1566 (See [demo](demo.ipynb) ipython notebook file). 
StarcNet predictions are
saved into `output/predictions.csv`.


To run StarcNet on NGC1566:

```
$ bash run_starcnet.sh 1
```

To produce the predictions run:
```
$ python src/run_visualization.py
```

The output is saved into `output/predictions.png`.


### Run StarcNet with local data

1. Save the 5 mosaic's `.FITS` files into `legus/frc_fits_files/` folder.
2. Save catalog `.tab` file into `legus/tab_files/`.
3. Name of galaxy should be in `targets.txt`.
4. Name of the mosaic(s) files should be in `frc_fits_links.txt`.
5. Name of .tab file with the cluster catalog(s) (with object coordinates) should be in `tab_links.txt`.
6. Run `bash run_starcnet.sh`

**Note:** The `.tab` file must have 3 columns, first one with ids and the last two with the coordinates. If your catalog only has the two columns of the coordinates you can use `src/add_ids_to_coords.py` file.

### Run StarcNet with online LEGUS catalogs

1. Name of galaxy should be in `targets.txt`.
2. Links to the mosaic(s) files should be in `frc_fits_links.txt`.
3. Links to the cluster catalog(s) (with objects coordinates) should be in `tab_links.txt`.
4. Run `bash run_starcnet.sh 1`

## Acknowledgements

This work is supported by the [National Science Foundation (NSF)](https://nsf.gov/index.jsp) of the United States under the award [\#1815267](https://nsf.gov/awardsearch/showAward?AWD_ID=1815267).
