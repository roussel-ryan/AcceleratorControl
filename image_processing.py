import numpy as np
from scipy import ndimage
from skimage import feature
from skimage.segmentation import watershed



def id_blobs(image, threshold = 0.01):
    '''
    identify blobs in the image to do image segmentation
    
    Arguments:
    ----------
    image : ndarray, (n,m)
         ndarray of image

    threshold : float
         threshold used to determine blobs, should be between 0 -> 1 (normalized)
    
    max_radius_px : float
         maximum blob radius in pixels

    '''

    #identifies connected regions above a certain threshold 
    labeled, n_objects = ndimage.label(image > threshold)

    return labeled, n_objects

def calculate_moments(labeled, image, n_objects):
    '''
    calculate the means and covariance matricies of the image inside of 
    labeled objects
    
    Arguments:
    ----------
    labeled : ndarray, (n,m)
         labeled array showing segmented image labels 

    image : ndarray, (n,m)
         ndarray of image

    n_objects : int
         number of identified objects

    '''

    vals = np.arange(1,n_obj+1)
        
    means = []
    variances = []
    for i in vals:
        coords = np.nonzero(labeled == i)        
        mean = [np.sum(coords[j]*image[coords]) /
                  np.sum(image[coords]) for j in [0,1]]

        means += [mean]

        #calculate weighted var
        cov = np.cov(coords,aweights = image[coords])
        variances += [cov]
        
    return means, variances
