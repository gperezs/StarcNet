import os
import sys
import csv
import pickle
import numpy as np

with open('data/test_raw_32x32.dat', 'rb') as infile:
    dset = pickle.load(infile)

data, ids, galaxies, coords = dset['data'], dset['ids'], dset['galaxies'], dset['coordinates']

scores = np.load('output/scores.npy')
preds = np.argmax(scores,axis=1)

with open('output/predictions.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Galaxy', 'Id', 'X', 'Y','Prediction'])
    for i in range(len(ids)):
        filewriter.writerow([galaxies[i], ids[i], coords[i][0], coords[i][1], preds[i]+1])
