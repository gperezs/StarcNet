import os
import sys
import pickle
import numpy as np
from astropy.io import fits


def create_target_db(tab=None, target=None, sz=22):
    """
    Function for loading of CAT annotations, FITS file, and creation of numpy array
    with all star cluster candidates. Size of output 'slices' is (N x 1 x sz xsz),
    were N is the number of valid star cluster candidates of the current galaxy.

    input:
      - cat:       CAT complete filename
      - target:    target galaxy (e.g.: eso486-g021, ngc4449, etc.)
      - sz:        window size for visualization (window size: sz x sz)

    output:
      - slices:    numpy array of size (N x 1 x sz x sz) with star cluster candidates
      - coords:    numpy array of size (N,) with star cluster candidate coordinates

    [GPS - 01/22/2019]
    """

    # targets with ACS filters
    acs435814 = ['ngc628-e']
    acs555814 = ['ngc4395-s','ngc7793-w','ugc4305','ugc4459','ugc5139']
    acs606814 = ['ic4247','ngc3738','ngc5238','ngc5474','ngc5477','ugc1249','ugc685','ugc7408','ugca281']
    acs435555814 =['ngc1313-e','ngc1313-w','ngc4449','ngc5194-ngc5195-mosaic','ngc5253','ngc628-c']

    # load .tab file:
    if 'ngc1313' in target: # has different column number for class in .readme file
        sid,x,y=np.loadtxt('legus/tab_files/'+tab,usecols=(0,1,2),unpack=True)
    elif any(ext in target for ext in ('ngc3351','ngc4242','ngc45')):
        sid,x,y=np.loadtxt('legus/tab_files/'+tab,usecols=(0,1,2),unpack=True)
    else:
        sid,x,y=np.loadtxt('legus/tab_files/'+tab,usecols=(0,1,2),unpack=True)

    # Load FITS data
    file_names = [file for file in sorted(os.listdir('legus/frc_fits_files/')) if target in file]

    fits_image_filename1 = [file for file in file_names if '275' in file]
    fits_image_filename2 = [file for file in file_names if '336' in file]
    fits_image_filename3 = [file for file in file_names if '435' in file or '438' in file]
    fits_image_filename4 = [file for file in file_names if '555' in file or '606' in file]
    fits_image_filename5 = [file for file in file_names if '814' in file]

    hdul1 = fits.open('legus/frc_fits_files/'+fits_image_filename1[0])
    hdul2 = fits.open('legus/frc_fits_files/'+fits_image_filename2[0])
    hdul3 = fits.open('legus/frc_fits_files/'+fits_image_filename3[0])
    hdul4 = fits.open('legus/frc_fits_files/'+fits_image_filename4[0])
    hdul5 = fits.open('legus/frc_fits_files/'+fits_image_filename5[0])

    # Working with Image Data
    data1 = hdul1[0].data
    data2 = hdul2[0].data
    data3 = hdul3[0].data
    data4 = hdul4[0].data
    data5 = hdul5[0].data

    w = int(np.floor(sz/2)) # used for slicing
    ty, tx = np.shape(data1)
    good = np.where(np.logical_and(np.logical_and(x > w+1 , (tx - x) > w+1) ,
                     np.logical_and(y > w+1 , (ty - y) > w+1))) \
    
    # to discard candidates close to border ([x,y] < w)
    sid, x, y = sid[good], x[good], y[good]
    slices = []
    if sz%2 == 1:
        ww = 1
    else:
        ww = 0
    for i in range(len(x)): # create array with all slices of current galaxy
        obj_slices = np.zeros((5,sz,sz))
        obj_slices[0,:,:] = data1[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[1,:,:] = data2[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[2,:,:] = data3[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[3,:,:] = data4[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        obj_slices[4,:,:] = data5[int(y[i])-w:int(y[i])+w+ww, int(x[i])-w:int(x[i])+w+ww]
        slices.extend([obj_slices])
    slices = np.asarray(slices) # Array of size (N x 5 x sz x sz)
    
    # save coordinates of each candidate and ids in .tab file
    coords = np.concatenate((np.asarray(x, dtype=np.int)[:,np.newaxis], \
                        np.asarray(y, dtype=np.int)[:,np.newaxis]),axis=1)
    ids = np.asarray(sid, dtype=np.int)
    return slices, coords, ids


def load_db(file_name):
    """
    Function for loading dataset from .dat dictionary.

    input:
      - file_name: file name and directory of current galaxy dataset file

    output:
      - dset['data']: numpy array with image data of size (N x 5 x sz x sz)
      - dset['label']: numpy array with labels of size (N,)
      - dset['coordinates']: numpy array with coordinates of each object of size (N, 2)

    where N is the number of candidates of current galaxy and sz is the slice size.

    [GPS - 01/22/2019]
    """
    with open(file_name, 'rb') as infile:
        dset = pickle.load(infile)
    return dset['data'], dset['coordinates'], dset['ids']


def get_name(name):
    """
    Remove \n character at the en of line (if exists)
    """
    if name[-1] == '\n':
        return name[0:-1]
    else:
        return name


def get_tab_filenames(targets):
    """
    Retrieve tab filenames from a target list
    """
    tab_filenames = []
    tabs = os.listdir('legus/tab_files')
    for target in targets:
        found = False
        for tab in tabs:
            if ('_'+get_name(target)+'_') in tab:
                tab_filenames.append(tab)
                found = True
        if not found:
            sys.exit('NOT FOUND: tab file not found for galaxy %s'%(target))
    return tab_filenames

