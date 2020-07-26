import os
import sys
import time
import random
import pickle
import numpy as np
import argparse
from os.path import isfile, join

sys.path.insert(0, './src/utils')
import data_utils as du

"""
Script for creating dataset to predict. Separate sets with different galaxies. 

[GPS - 03/09/2019]
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
    tabs_txt = 'tab_links.txt'

    # dir with created slices of galaxy targets:
    data_dir = dirm+dataset_info+str(sz)+'x'+str(sz)

    tin = time.time()
    files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    files = sorted(files)

    print('creating dataset...')
    data = np.array([], dtype=np.int64).reshape(0,5,sz,sz)
    coords = np.array([], dtype=np.int64).reshape(0,2)
    ids = np.array([], dtype=np.int64).reshape(0,)
    galaxies = np.array([], dtype=np.int64).reshape(0,)
    for i in range(len(files)):
        t_ini = time.time()
        file_name = join(data_dir,files[i])
        target_data, target_coords, target_ids = du.load_db(file_name)
        data = np.concatenate((data, target_data), axis=0)
        coords = np.concatenate((coords, target_coords), axis=0)
        ids = np.concatenate((ids, target_ids), axis=0)
        strs = [files[i][:-4] for x in range(len(target_ids))]
        galaxies = np.concatenate((galaxies, strs), axis=0)
    
    # Save test set
    db_name = 'test_'+dataset_info+str(sz)+'x'+str(sz)
    dataset = {'data':data, 'coordinates':coords, 'galaxies':galaxies, 'ids':ids}
    with open(os.path.join('data', db_name)+'.dat', 'wb') as outfile:
                    pickle.dump(dataset, outfile, pickle.HIGHEST_PROTOCOL)
    print('dataset shape: %s' % (str(data.shape)))
