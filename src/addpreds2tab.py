import os
import csv
import numpy as np
from scipy.special import softmax

folder = 'legus/tab_files/'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
files = sorted(files)
filename = files[0]

scores = np.load('output/scores.npy')
scores = softmax(scores,axis=1)
preds = np.argmax(scores,axis=1)

with open(os.path.join(folder,filename),'r') as csvinput:
    with open(os.path.join('output','output.tab'), 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter='\t')
        reader = csv.reader(csvinput, delimiter='\t')

        for i in range(len(preds)):
            row = next(reader)
            row.append(str(preds[i]+1))
            writer.writerow(row)

with open(os.path.join(folder,filename),'r') as csvinput:
    with open(os.path.join('output','output_scores.tab'), 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter='\t')
        reader = csv.reader(csvinput, delimiter='\t')

        for i in range(len(preds)):
            row = next(reader)
            row.append(str(preds[i]+1))
            row.append('[%.2f - %.2f - %.2f - %.2f]'%(scores[i][0],scores[i][1],scores[i][2],scores[i][3]))
            writer.writerow(row)
