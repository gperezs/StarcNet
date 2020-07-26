from PIL import Image
import os
import csv
import numpy as np
from astropy.io import fits
from scipy.special import softmax
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.draw import circle_perimeter

# class1: red, class2: green, class3: blue, class4: white
def draw_circle(img,score,pred,coord,rad=10,linewidth=3,predint=1):
    score = softmax(score)
    magn = score[pred]
    for i in range(linewidth):
        rr, cc = circle_perimeter(int(coord[1]),int(coord[0]),rad+i)
        if pred == 4: #for background
            img[rr, cc] = 1
        elif pred == 0:
            img[rr, cc, 0] = magn
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = 0
        elif pred == 1:
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = magn
            img[rr, cc, 2] = 0
        elif pred == 2:
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = magn
        elif pred == 3:
            img[rr, cc, 0] = magn
            img[rr, cc, 1] = magn
            img[rr, cc, 2] = 0
    return img


def create_image(target=None):
    # targets with ACS filters
    acs435814 = ['ngc628-e']
    acs555814 = ['ngc4395-s','ngc7793-w','ugc4305','ugc4459','ugc5139']
    acs606814 = ['ic4247','ngc3738','ngc5238','ngc5474','ngc5477','ugc1249','ugc685','ugc7408','ugca281']
    acs435555814 =['ngc1313-e','ngc1313-w','ngc4449','ngc5194-ngc5195-mosaic','ngc5253','ngc628-c']

    # if ACS filter is used
    if target in acs435814:
        #print('acs 435 - 814')
        g1, g2, g3, g4, g5 = 1, 1, 0.474, 1, 0.471
    elif target in acs555814:
        #print('acs 555 - 814')
        g1, g2, g3, g4, g5 = 1, 1, 1, 1.083, 0.471
    elif target in acs606814:
        #print('acs 606 - 814')
        g1, g2, g3, g4, g5 = 1, 1, 1, 0.689, 0.471
    elif target in acs435555814:
        #print('acs 435 - 555 - 814')
        g1, g2, g3, g4, g5 = 1, 1, 0.474, 1.083, 0.471
    else:
        #print('no acs')
        g1, g2, g3, g4, g5 = 1, 1, 1, 1, 1

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

    datac = np.zeros((data1.shape[0],data1.shape[1],3), dtype=np.float64)
    datac[:,:,2] = (21.63*g1*data1 + 8.63*g2*data2) / 2.
    datac[:,:,1] = (4.52*g3*data3)
    datac[:,:,0] = (1.23*g4*data4 + g5*data5) / 2.
    datac = np.clip(datac, 0,1)
    plt.imsave(os.path.join(savedir,target+".png"), datac)


if __name__ == '__main__':

    scoresdir = 'output'
    savedir = 'output/visualizations'

    scores = np.load(os.path.join(scoresdir,'scores.npy'))
    preds = np.argmax(scores, axis=1)

    coords = []
    galaxies = []
    with open('output/predictions.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            coords.append(row[2:4])
            galaxies.append(row[0])
    
    coords = np.asarray(coords[1::])
    targets = np.unique(galaxies)[1::]
    galaxies = np.asarray(galaxies[1::])

    for target in targets:
        print('creating %s visualization...'%(target))
        idxg = np.where(galaxies == target)
        coordsg = coords[idxg]
        predsg = preds[idxg]
        scoresg = scores[idxg]
       
        create_image(target)
        img = mpimg.imread(os.path.join(savedir,target+'.png')) 
        
        for j in range(len(predsg)):
            img = draw_circle(img,scoresg[j,:],predsg[j],coordsg[j,:])

        img = (img - img.min())/(img.max() - img.min())
        tp = 0.7
        cbox = mpimg.imread('src/cbox.png')
        img[100:568,100:1100,0:3] = cbox*tp + img[100:568,100:1100,0:3]*(1-tp)

        plt.imsave(os.path.join(savedir,target+'_predictions.png'),img)    
        img_small = resize(img, (int(img.shape[0]/(img.shape[1]/2000)),2000), anti_aliasing=False, mode='reflect')
        plt.imsave(os.path.join(savedir,target+'_predictions.jpg'),img_small)
        print("visualization saved to '%s'"%(os.path.join(savedir,target+'_predictions.png')))

