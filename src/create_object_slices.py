import os
import sys
import time
import pickle
import numpy as np
import argparse
from os.path import isfile, join

sys.path.insert(0, './src/utils')
import data_utils as du

"""
Script for reading target galaxy data from FITS files and annotations from CAT files.
Slices with annotations are save in a single .dat dictionary file per target galaxy.
Slices are of size (N x 1 x sz x sz), where N is the number of candidates detected by 
SExtractor of target galaxy, and sz is the size of the slice of each star cluster 
candidate (--slice-size).

[GPS - 01/22/2019]
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Create candidate slices')
    parser.add_argument('--slice-size', type=int, default=22, 
                        help='window size for visualization (slice size: sz x sz)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    sz = args.slice_size
    
    dataset_info = 'raw_'
   
    dirm = 'data/'
    targets_txt = 'targets.txt'

    # dir to save created dataset of current target galaxy:
    data_dir = dirm+dataset_info+str(sz)+'x'+str(sz)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    with open(targets_txt, "r+") as x:
        targets = x.readlines()
        tab_filenames = du.get_tab_filenames(targets)
        print('creating object arrays for galaxies %s'%(targets))
        for i in range(len(targets)): # create one dataset file per target
            target = du.get_name(targets[i]) # target galaxy (e.g.: ic4247, ngc4449, etc.)
            tab = du.get_name(tab_filenames[i]) 

            tin = time.time()
            slices, coords, ids = du.create_target_db(tab, target, sz)
            data = {'data':slices, 'coordinates':coords, 'ids':ids}
              
            with open(os.path.join(data_dir, target)+'.dat', 'wb') as outfile:
                pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

