import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, feature, measure, filters, morphology,transform
import logging
import cv2

def zero_surrounded(array):
    return not (array[0,:].any() or
                array[-1,:].any() or
                array[:,0].any() or
                array[:,-1].any())

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def check_image(image, verbose = False, n_blobs_required = 1):
    '''
    Note: use this with a sub-image ROI

    check for the following 
    - if there is a beam
    - if the beam is entirely inside the ROI
    - if the image is saturated


    '''

    logger = logging.getLogger(__name__)
    
    image = image.astype(np.float)
    
    smoothed_image = filters.gaussian(image, 3)
    #apply a gaussian filter
    #if the normalized difference is below 0.01 then there is no beam / beam is too big
    #width = (np.max(image) - np.min(image)) / np.mean(image)
    #logger.debug(f'smoothed amplitude: {width}')    
    #if width < 0.01:
    #    logger.warning('no beam detected')
    #    return 0

    #apply triangle threshold to determine beam locations
    triangle_threshold = filters.threshold_triangle(smoothed_image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    thresholded_image = np.where(smoothed_image > triangle_threshold * 1.5, 1, 0)
    
    dilated_binary, n_blobs = get_blobs(thresholded_image)
    
    verbose = True
    if verbose:

        fig,ax = plt.subplots(1,4)
        c = ax[0].imshow(image)
        ax[1].imshow(smoothed_image)
        ax[2].imshow(dilated_binary)        
        #add ellipses
        
        
        fig.colorbar(c)
        plt.show()
    
    if n_blobs < n_blobs_required:
        logger.warning(f'too few blobs detected! detecting {n_blobs} n_blobs')
        return 0
    
    
        
    #get the projected std of both x and y
    proj_x = np.sum(thresholded_image, axis = 0)
    proj_y = np.sum(thresholded_image, axis = 1)
    
    #calculate stds
    x_len = len(proj_x)
    y_len = len(proj_y)
    std_x = weighted_std(np.arange(x_len), proj_x)
    std_y = weighted_std(np.arange(y_len), proj_y)
    
    std_scale = 2.5
    side_scale = 0.65
    logger.debug(std_x*std_scale)
    logger.debug(x_len * side_scale)
    logger.debug(std_y*std_scale)
    logger.debug(y_len * side_scale)
    if (std_x*std_scale > side_scale* x_len) or (std_y*std_scale > side_scale*y_len):
        logger.warning('beam too diffuse')
        return 0
    
    #if there is no beam on the edges
    if zero_surrounded(dilated_binary):
        return 1
    else:
        
        logger.warning('ROI is clipping the beam envelope')
        return 0

def get_blobs(image):
    #segment a binary image into regions, which are deleted if they have 
    #less than a given size
    labels, n_blobs = ndimage.label(image)
    new_image = image.copy()
    
    min_size = 500
    n_good_blobs = n_blobs
    for i in range(1, n_blobs + 1):
        counts = np.count_nonzero(i == labels)
        if not counts > min_size:
            new_image = np.where(i == labels, 0, new_image)
            n_good_blobs -= 1
            
    return new_image, n_good_blobs
    
def rotate_beamlets(image):
    logger = logging.getLogger(__name__)
    
    image = image.astype(np.float)
    
    smoothed_image = filters.gaussian(image, 3)

    #apply triangle threshold to determine beam locations
    triangle_threshold = filters.threshold_triangle(smoothed_image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')

    blob_fitting_image = np.where(smoothed_image > triangle_threshold*1.0 , 1 ,0).astype('uint8')
    
    #find contours
    cnts, huers = cv2.findContours(blob_fitting_image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    angles = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            ellipse = cv2.fitEllipse(cnt)
            angles += [ellipse[2]]
        
    angles = np.array(angles)
    
    if len(angles) == 0:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(image)
        ax[1].imshow(blob_fitting_image)
        
    mean_angle = np.mean(angles)
    rotated_image = transform.rotate(image, mean_angle - 90.0)
    
    
        
    return rotated_image, mean_angle - 90.0
    

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
