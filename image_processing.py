import numpy as np
from scipy import ndimage
from skimage import feature, filters
from skimage.segmentation import watershed

def measure_spot(image, center, radius, sigma = 3, threshold = 0.1):
    '''
    process image to get beam spot size

    Arguments:
    ----------
    image : ndarray, (n,m)
         ndarray of image

    center : ndarray (2,)
         center of screen in px
    
    radius : float
         radius of screen in pixels

    '''

    #create a mask and mask the image array
    mask = create_mask(image, center, radius)
    image = np.ma.masked_array(image, mask = mask)

    #fill in masked portions of the image with zeros
    fimage = image.filled(0)
    
    #apply gaussian filter
    fimage = filter_image(fimage, sigma)

    #id and fit the beam
    lbls, n_blobs = id_blobs(fimage, threshold = threshold)
    means, variances = calculate_moments(lbls, fimage, n_blobs)

    #return beam size
    return np.sqrt(np.diag(variances[0]))


def create_mask(image, center, radius):
    '''
    set all elements outside of circular mask to zero

    Arguments:
    ----------
    image : ndarray, (n,m)
         ndarray of image

    center : ndarray (2,)
         center of circle in px
    
    radius : float
         radius in pixels

    '''
    shape = array.shape
    r = 0.875*radius
    mask = np.ones_like(array)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.sqrt((j - center[0])**2 + (i - center[1])**2) < r:
                mask[i][j] = 0
        
    return mask

def filter_image(image, sigma):
    '''
    apply a gaussian filter to the image

    '''
    return filters.gaussian(image, sigma)
    
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
