import numpy as np
import matplotlib.pyplot as plt
from skimage import data, feature, measure, filters, morphology
import logging

def zero_surrounded(array):
    return not (array[0,:].any() or
                array[-1,:].any() or
                array[:,0].any() or
                array[:,-1].any())


def check_image(image, verbose = False):
    '''
    Note: use this with a sub-image ROI

    check for the following 
    - if there is a beam
    - if the beam is entirely inside the ROI
    - if the image is saturated


    '''

    logger = logging.getLogger(__name__)
    
    smoothed_image = filters.gaussian(image, 3)

    
    #apply a gaussian filter
    if verbose:
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(image)
    #gaussian_image = filters.gaussian(smoothed_image, 100, mode = 'reflect')
    if verbose:
        ax[1].imshow(smoothed_image)

    #if the normalized difference is below 0.01 then there is no beam / beam is too big
    #width = (np.max(gaussian_image) - np.min(gaussian_image)) / np.mean(gaussian_image)
    #logger.debug(f'smoothed amplitude: {width}')    
    #if width < 0.01:
    #    logger.warning('no beam detected')
    #    return 0

    #apply triangle threshold to determine beam locations
    triangle_threshold = filters.threshold_triangle(smoothed_image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    thresholded_image = np.where(smoothed_image > triangle_threshold*1.1, 1, 0)

    #dilate image
    dilated_binary = morphology.binary_dilation(thresholded_image,
                                                morphology.disk(10))

    if verbose:
        ax[2].imshow(dilated_binary)
        plt.show()

    #if there is no beam on the edges
    if zero_surrounded(dilated_binary):
        return 1
    else:
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(image)
        ax[1].imshow(smoothed_image)
        ax[2].imshow(dilated_binary)
        plt.show()
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
