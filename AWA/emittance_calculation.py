import matplotlib.pyplot as plt
import numpy as np
import logging

from scipy.signal import find_peaks
from skimage import filters

def calculate_emittance(image, scale, slit_sep, drift):
    '''
    calculate emittance using processed beam image

    Arguments
    ---------
    image : ndarray, size (N,M)
        Post-processed screen image with screen removed and noise present

    scale : float
        Pixel scaling factor in meters per pixel

    slit_sep : float
        Slit seperation in meters

    drift : float
        Longitudinal drift distance between screen and slits

    threshold : float in range [0,1), optional
        Fractional threshold to set proj to zero as a fraction of global maximum, 
        default 0.0 (no change)
    '''

    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    #get projection
    orig_proj = np.sum(image, axis = 1)
    orig_proj = orig_proj / np.max(orig_proj)

    #apply threshold
    #proj = np.where(proj > threshold * np.max(proj), proj, 0.0)
    try:
        triangle_threshold = filters.threshold_triangle(orig_proj)
    except ValueError:
        fig,ax = plt.subplots(1,2)
        ax[0].plot(orig_proj)
        ax[1].imshow(image)
        print(image)
    logger.debug(f'triangle_threshold: {triangle_threshold}')
    proj = np.where(orig_proj > triangle_threshold, orig_proj, 0)

    #plot proj if in debugging
    if 1:
        fig,ax = plt.subplots(1,2)
        ax[0].plot(orig_proj)
        ax[1].plot(proj)
        
        plt.show()

    #we assume that the beam is divergent, as a result the peaks should be at least
    #2 mm apart
    peaks,_ = find_peaks(proj, distance = 0.5e-3 / scale)

    if len(peaks) < 5:
        logger.warning(f'detected only {len(peaks)} peaks '
                       '-- emittance might be underestimated')

    logger.debug(f'peak finding found {len(peaks)} peaks')
    logger.debug(f'found peaks at {peaks} px')
    logger.debug(f'mean seperation {np.mean(peaks[1:] - peaks[:-1])*scale:.2e}')
    logger.debug(f'rms seperation {np.std(peaks[1:] - peaks[:-1])*scale:.2e}')
    
    #calculate mid points and number of blobs
    mid_pts = (peaks[1:] + peaks[:-1])/2
    mid_pts = mid_pts.astype(np.int)
    mid_pts = np.array([0, *mid_pts, len(proj)])

    x_screen_px = np.arange(len(proj))
    n_blobs = len(mid_pts) - 1

    
    #calculate gaussian fit stats
    a = np.empty(n_blobs) # relative intensity
    b = np.empty(n_blobs) #'y' peak position
    c = np.empty(n_blobs) #'y' sigma

    for n in range(n_blobs):        
        sub_proj = proj[mid_pts[n] : mid_pts[n+1]] 
        sub_x = x_screen_px[mid_pts[n] : mid_pts[n+1]]

        a[n] = np.sum(sub_proj)
        b[n] = np.average(sub_x, weights = sub_proj)
        c[n] = np.sqrt(np.average((sub_x - b[n])**2, weights=sub_proj))

    #sort peaks by central position
    sorted_idx = np.argsort(b)
    a,b,c = a[sorted_idx],b[sorted_idx],c[sorted_idx]

    #convert pixel lengths to meters
    b = b * scale
    c = c * scale

    #define slit locations
    x_slit_m = np.linspace(-(n_blobs-1)/2*slit_sep, (n_blobs-1)/2*slit_sep, n_blobs)

    #calculate beam centroid at slits
    ixi = np.sum(a * x_slit_m) / np.sum(a)

    #center b coords on beam center on screen
    b = b - np.average(b, weights = a)

    #calculate rms spread at slits
    ixxi = np.sum(a * (x_slit_m - ixi)**2) / np.sum(a)

    #calc mean divergence of each beamlet
    xp = (b - x_slit_m) / drift

    #calc mean divergence of beam at slits
    ixpi = np.sum(a * xp) / np.sum(a)

    #calc spread in divergence for each beamlet
    sp = c / drift

    #calc rms divergence at slits
    ixpxpi = np.sum(a * sp**2 + a * (xp - ixpi)**2) / np.sum(a)

    #calc correlation term at slits
    ixxpi = (np.sum(a * x_slit_m * xp) - np.sum(a) * ixi *ixpi) / np.sum(a)

    logger.debug('emittance calculation results')
    logger.debug(f'ixi:{ixi}')
    logger.debug(f'ixpi:{ixpi}')
    logger.debug(f'ixxi:{ixxi}')
    logger.debug(f'ixpxpi:{ixpxpi}')
    logger.debug(f'ixxpi:{ixxpi}')

    #calculate emittance
    emittance = np.sqrt(ixxi * ixpxpi - ixxpi**2)
    logger.info(f'calculated emittance: {emittance:.2e}, n_peaks:{len(peaks)}')

    return emittance


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    #testing
    minscreen2mm = np.genfromtxt('test_images/onenc400200.txt',names=True)
    img, _, _ = np.histogram2d(minscreen2mm['x'], minscreen2mm['y'],
                               range = np.array(((-25.0,25.0),
                                                 (-25.0,25.0)))*1e-3,
                               bins=(700,700))

    scale = 50e-3 / 700
    drift = 1.5748
    slit_sep = 0.002

    calculate_emittance(img, scale, slit_sep, drift, threshold = 0.0)
