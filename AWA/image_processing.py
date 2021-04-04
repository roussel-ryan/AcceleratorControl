import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature, measure, filters, morphology
import logging

def zero_surrounded(array):
    return not (array[0,:].any() or
                array[-1,:].any() or
                array[:,0].any() or
                array[:,-1].any())


def check_image(image, verbose = True):
    '''
    Note: use this with a sub-image ROI

    check for the following 
    - if there is a beam
    - if the beam is entirely inside the ROI
    - if the image is saturated


    '''

    logger = logging.getLogger(__name__)
    
    #apply a gaussian filter
    if test:
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(image)
    gaussian_image = filters.gaussian(image, 100, mode = 'reflect')
    if test:
        ax[1].imshow(gaussian_image)

    #if the normalized difference is below 0.01 then there is no beam / beam is too big
    width = (np.max(gaussian_image) - np.min(gaussian_image)) / np.mean(gaussian_image)
    if width < 0.01:
        logger.warning('no beam detected')
        return 0

    #apply triangle threshold to determine beam locations
    triangle_threshold = filters.threshold_triangle(image)
    thresholded_image = np.where(image > triangle_threshold, 1, 0)

    #dilate image
    dilated_binary = morphology.binary_dilation(thresholded_image,
                                                morphology.disk(10))

    if test:
        ax[2].imshow(dilated_binary)

    #if there is no beam on the edges
    if zero_surrounded(dilated_binary):
        return 1
    else:
        logger.warning('ROI is clipping the beam envelope')
        return 0


    
    

    

if __name__ == '__main__':
    data = np.genfromtxt('test_images/onenc250200.txt',names = True)

    size = 50e-3
    bins = 700
    img, xedges, yedges = np.histogram2d(data['x'], data['y'],
                                     range = np.array(((-0.0,1.0),
                                                       (-0.0,1.0)))*size/2.0,
                                     bins=(bins,bins))

    img = img / np.max(img)
    img = 0.15*np.random.rand(*img.shape) + img
    
    print(check_image(img))
    plt.show()
