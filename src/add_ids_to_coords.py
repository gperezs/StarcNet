import os
import sys
import csv
import pickle
import numpy as np
from scipy.special import softmax

with open('legus/cat_UBVI_avgapcor_magcut_ci1.4_ngc4258s.coo', 'r') as read_obj:
    # Create a csv.reader object from the input file object
    csv_reader = csv.reader(read_obj)
    cont = 1
    arr = np.empty((0,3), float)
    for row in csv_reader:
        # Append the default text in the row / list
        spl = row[0].split()
        rownp = np.array((cont, spl[0], spl[1]))
        cont += 1
        arr = np.concatenate((arr, rownp[np.newaxis,:]),axis=0)    
    np.savetxt('legus/cat_UBVI_avgapcor_magcut_ci1.4_ngc4258s.tab', arr, fmt='%s')
        

