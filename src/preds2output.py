import os
import sys
import csv
import pickle
import numpy as np
from scipy.special import softmax

with open('data/test_raw_32x32.dat', 'rb') as infile:
    dset = pickle.load(infile)

data, ids, galaxies, coords = dset['data'], dset['ids'], dset['galaxies'], dset['coordinates']

scores = np.load('output/scores.npy')
scores = softmax(scores,axis=1)
preds = np.argmax(scores,axis=1)

with open('output/predictions.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Galaxy', 'Id', 'X', 'Y','Prediction'])
    for i in range(len(ids)):
        filewriter.writerow([galaxies[i], ids[i], coords[i][0], coords[i][1], preds[i]+1])

targets = np.unique(galaxies)
for target in targets:
    idxg = np.where(galaxies == target)
    idsg = ids[idxg]
    coordsg = coords[idxg]
    predsg = preds[idxg]
    scoresg = scores[idxg]
    
    with open(os.path.join('output',target+'.tab'), 'w') as taboutput:
        writer = csv.writer(taboutput, delimiter='\t')
        for i in range(len(idsg)):
            row = [idsg[i], coordsg[i,0], coordsg[i,1], predsg[i]+1]
            row.append('[%.2f - %.2f - %.2f - %.2f]'%(scores[i][0],scores[i][1],scores[i][2],scores[i][3]))
            writer.writerow(row)

