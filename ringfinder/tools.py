# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:53:28 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np
import math
import configparser
from scipy.ndimage.measurements import center_of_mass
from skimage.feature import peak_local_max
try:
    import skimage.filters as filters
except ImportError:
    import skimage.filter as filters
from skimage.transform import probabilistic_hough_line

from ringfinder.neurosimulations import simAxon


def saveConfig(main, filename=None):

    if filename is None:
        filename = os.path.join(os.getcwd(), 'config')

    config = configparser.ConfigParser()

    config['Loading'] = {
        'STORM px nm': main.STORMPxEdit.text(),
        'STORM magnification': main.magnificationEdit.text(),
        'STED px nm': main.STEDPxEdit.text()}

    config['Analysis'] = {
        'ROI size nm': main.roiSizeEdit.text(),
        'Gaussian sigma filter nm': main.sigmaEdit.text(),
        'nsigmas threshold': main.intThresEdit.text(),
        'Lines min length nm': main.lineLengthEdit.text(),
        'Ring periodicity nm': main.wvlenEdit.text(),
        'Sinusoidal pattern power': main.sinPowerEdit.text(),
        'Angular step deg': main.thetaStepEdit.text(),
        'Delta angle deg': main.deltaThEdit.text(),
        'Discrimination threshold': main.corrThresEdit.text(),
        'Area threshold %': main.minAreaEdit.text()}

    with open(filename, 'w') as configfile:
        config.write(configfile)


def saveDefaultConfig(filename=None):

    if filename is None:
        filename = os.path.join(os.getcwd(), 'config')

    config = configparser.ConfigParser()

    config['Loading'] = {
        'STORM px nm': '13.3', 'STORM magnification': '10', 'STED px nm': '20'}

    config['Analysis'] = {
        'ROI size nm': '1000', 'Gaussian sigma filter nm': '100',
        'nsigmas threshold': '0.5', 'Lines min length nm': '300',
        'Ring periodicity nm': '180', 'Sinusoidal pattern power': '6',
        'Angular step deg': '3', 'Delta angle deg': '20',
        'Discrimination threshold': '0.2', 'Area threshold %': '20'}

    with open(filename, 'w') as configfile:
        config.write(configfile)


def loadConfig(main, filename=None):

    if filename is None:
        filename = os.path.join(os.getcwd(), 'config')

    config = configparser.ConfigParser()
    config.read(filename)

    loadConfig = config['Loading']
    main.STORMPxEdit.setText(loadConfig['STORM px nm'])
    main.magnificationEdit.setText(loadConfig['STORM magnification'])
    main.STEDPxEdit.setText(loadConfig['STED px nm'])

    analysisConfig = config['Analysis']
    main.roiSizeEdit.setText(analysisConfig['ROI size nm'])
    main.sigmaEdit.setText(analysisConfig['Gaussian sigma filter nm'])
    main.intThresEdit.setText(analysisConfig['nsigmas threshold'])
    main.lineLengthEdit.setText(analysisConfig['Lines min length nm'])
    main.wvlenEdit.setText(analysisConfig['Ring periodicity nm'])
    main.sinPowerEdit.setText(analysisConfig['Sinusoidal pattern power'])
    main.thetaStepEdit.setText(analysisConfig['Angular step deg'])
    main.deltaThEdit.setText(analysisConfig['Delta angle deg'])
    main.corrThresEdit.setText(analysisConfig['Discrimination threshold'])
    main.minAreaEdit.setText(analysisConfig['Area threshold %'])


def pearson(a, b):
    """2D pearson coefficient of two matrixes a and b"""

    # Subtracting mean values
    an = a - np.mean(a)
    bn = b - np.mean(b)

    # Vectorized versions of c, d, e
    c_vect = an*bn
    d_vect = an*an
    e_vect = bn*bn

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/math.sqrt(np.sum(d_vect)*np.sum(e_vect))

    return r_out


def cosTheta(a, b):
    """Angle between two vectors a and b"""

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0

    cosTheta = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cosTheta


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    nrows = np.int(nrows)
    ncols = np.int(ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1, 2)
               .reshape(h, w))


def firstNmax(coord, image, N):
    """Returns the first N max in an image from an array of coord of the max
       in the image"""

    if np.shape(coord)[0] < N:
        return []
    else:
        aux = np.zeros(np.shape(coord)[0])
        for i in np.arange(np.shape(coord)[0]):
            aux[i] = image[coord[i, 0], coord[i, 1]]

        auxmax = aux.argsort()[-N:][::-1]

        coordinates3 = []
        for i in np.arange(0, N):
            coordinates3.append(coord[auxmax[i]])

        coord3 = np.asarray(coordinates3)

        return coord3


def arrayExt(array):
    """Extends an array in a specific way"""

    y = array[::-1]
    z = []
    z.append(y)
    z.append(array)
    z.append(y)
    z = np.array(z)
    z = np.reshape(z, 3*np.size(array))

    return z


def getDirection(data, binary, minLen, debug=False):
    """Returns the direction (angle) of the neurite in the image data.

    binary: boolean array with same shape as data. True means that the pixel
        belongs to a neuron and False means background.
    minLen: minimum  line length in px."""

    th0, sigmaTh, lines = linesFromBinary(binary, minLen, debug)
    if debug:
        text = 'Angle = {0:.1f} +- {1:.1f} from {2:.0f} lines'
        print(text.format(th0, sigmaTh, len(lines)))

    try:

        if len(lines) > 1:

            # TO DO: find optimal threshold, 20 is arbitrary
            if sigmaTh < 15:
                return th0, lines
            else:
                if debug:
                    print('sigmaTh too high')
                return None, lines

        else:
            if debug:
                print('Only one line found')
            return None, lines

    except:
        # if sigmaTh is None (no lines), this happens
        if debug:
            print('No lines were found')
        return None, lines

def linesFromBinary(binaryData, minLen, debug=False):

    # find edges
    edges = filters.sobel(binaryData)

    # get directions
    lines = probabilistic_hough_line(edges, threshold=10, line_length=minLen,
                                     line_gap=3)

    if lines == []:
        if debug:
            print('No lines detected with Hough line algorithm')
        return None, None, lines

    else:
        angleArr = np.zeros(len(lines))
        for l in np.arange(len(lines)):
            p0, p1 = lines[l]

            # get the m coefficient of the lines and the angle
            try:
                m = (p1[0] - p0[0])/(p1[1] - p0[1])
                angle = (180/np.pi)*np.arctan(m)
            except ZeroDivisionError:
                angle = 90

            angleArr[l] = angle

        # Before calculating the mean angle, we have to make sure we're using
        # the same quadrant for all the angles. We refer all the angles to the
        # first one
        opt = np.array([180, 0, -180])
        for i in np.arange(1, len(angleArr)):
            dists = np.abs(angleArr[0] - (opt + angleArr[i]))
            angleArr[i] += opt[np.argmin(dists)]

        mean, std = np.mean(angleArr), np.std(angleArr)

        # We like angles in [0, 180)
        if mean < 0:
            mean += 180
            
        # histogram method for getDirection
 
        hrange = (-180, 180)
        arr = np.histogram(angleArr, bins=45, range=hrange)
 
        dig = (arr[0] != 0).astype(int)
        angleGroups = [np.split(arr[0], np.where(np.diff(dig) != 0)[0] + 1),
                       np.split(arr[1], np.where(np.diff(dig) != 0)[0] + 1)]
        angleGroupsSum = [np.sum(np.array(b)) for b in angleGroups[0]]
        biggerAngleGroup = angleGroups[1][np.argmax(angleGroupsSum)]
 
        if debug: 
            print('Total number of lines: {}, Number of lines in biggest group: {}'.format(np.sum(angleGroupsSum), np.max(angleGroupsSum)))
 
        if np.max(angleGroupsSum)/np.sum(angleGroupsSum) > 0.49:
            mean = np.mean(biggerAngleGroup)
            std = np.std(biggerAngleGroup)
 
            # We like angles in [0, 180)
            if mean < 0:
                mean += 180
 
        else:
            mean = None
            std = None

        return mean, std, lines


# First try at a method for directly fit the rings profile. Current problem
# is that it finds the longest line within the neuron and this is not parallel
# to the neuron. It should find the longest ~50nm thick rectangle instead.
def fitMethod(data, mask, thres, minLen, thStep, deltaTh, wvlen, sinPow,
              developer=False):

    mask = blocksMask[25]
    y0, x0 = np.array(center_of_mass(~mask), dtype=int)
    yMax, xMax = mask.shape
    if ~mask[y0, x0]:

        theta = np.arange(0, math.pi, thStep*math.pi/180)
        m = np.tan(theta)

        # Taking care of infinite slope case
        theta = theta[np.abs(m) < 1000]
        m = m[np.abs(m) < 1000]

        lineLength = 0
        for i in np.arange(len(theta)):

            # Find right intersection between line and neuron edge
            x, y = x0, y0
            while ~mask[y, x] and x < xMax:
                print(x, y)
                x += 1
                y = int(m[i]*(x - x0) + y0)

                # If it gets out of the image
                if abs(y) >= mask.shape[0] or y < 0:
                    x -= 1
                    y = int(m[i]*(x - x0) + y0)
                    break
            yq, xq = y, x

            # Find left intersection between line and neuron edge
            y, x = y0, x0
            while ~mask[y, x] and x > 0:
                print(x, y)
                x -= 1
                y = int(m[i]*(x - x0) + y0)

                # If it gets out of the image
                if abs(y) >= mask.shape[0] or y < 0:
                    x += 1
                    y = int(m[i]*(x - x0) + y0)
                    break
            yp, xp = y, x

            # vertical line case

            # We keep coordinates of longest line
            newLength = math.sqrt((xq - xp)**2 + (yq - yp)**2)
            if lineLength < newLength:
                lineLength = newLength
                x2, y2 = xq, yq
                x1, y1 = xp, yp


def corrMethod(data, mask, minLen, thStep, deltaTh, wvlen, sinPow,
               developer=False):
    """Searches for rings by correlating the image data with a given
    sinusoidal pattern

    data: 2D image data
    thres: discrimination threshold for the correlated data.
    sigma: gaussian filter sigma to blur the image, in px
    minLen: minimum line length in px.
    thStep: angular step size
    deltaTh: maximum pattern rotation angle for correlation matching
    wvlen: wavelength of the ring pattern, in px
    sinPow: power of the pattern function
    developer (bool): enables additional output of algorithms

    returns:

    corrMax: the maximum (in function of the rotated angle) correlation value
    at the image data
    thetaMax: simulated axon's rotation angle with maximum correlation value
    phaseMax: simulated axon's phase with maximum correlation value at thetaMax
    rings (bool): ring presence"""

    # phase steps are set to 20, TO DO: explore this parameter
    phase = np.arange(0, 21, 1)

    corrPhase = np.zeros(np.size(phase))

    # line angle calculated
    th0, lines = getDirection(data, np.invert(mask), minLen, developer)

    if th0 is None:

        theta = np.arange(0, 180, thStep)
        # result = np.nan means there's no neuron in the block
        corrPhaseArg = np.zeros(np.size(theta))
        corrPhaseArg[:] = np.nan
        corrTheta = np.zeros(np.size(theta))
        corrTheta[:] = np.nan
        corrMax = np.nan
        thetaMax = np.nan
        phaseMax = np.nan

    else:

        try:
            if developer:
                theta = np.arange(np.min([th0 - deltaTh, 0]), 180, thStep)
            else:
                theta = np.arange(th0 - deltaTh, th0 + deltaTh, thStep)

        except TypeError:
            th0 = 90
            deltaTh = 90
            theta = np.arange(0, 180, thStep)

        corrPhaseArg = np.zeros(np.size(theta))
        corrTheta = np.zeros(np.size(theta))

        subImgSize = np.shape(data)[0]

        # for now we correlate with the full sin2D pattern
        for t in np.arange(len(theta)):
            for p in phase:
                # creates simulated axon
                axonTheta = simAxon(subImgSize, wvlen, theta[t], p*.025, a=0,
                                    b=sinPow).data
                axonThetaMasked = np.ma.array(axonTheta, mask=mask)
                dataMasked = np.ma.array(data, mask=mask)

                # saves correlation for the given phase p
                corrPhase[p] = pearson(dataMasked, axonThetaMasked)

            # saves the correlation for the best p, and given angle i
            corrTheta[t - 1] = np.max(corrPhase)
            corrPhaseArg[t - 1] = .025*np.argmax(corrPhase)

        # get theta, phase and correlation with greatest correlation value
        # Find indices within (th0 - deltaTh, th0 + deltaTh)
        ix = np.where(np.logical_and(th0 - deltaTh <= theta,
                                     theta <= th0 + deltaTh))
        i = np.argmax(corrTheta[ix])
        thetaMax = theta[ix][i]
        phaseMax = corrPhaseArg[ix][i]
        corrMax = np.max(corrTheta[ix])

    return th0, corrTheta, corrMax, thetaMax, phaseMax


def FFTMethod(data, thres=0.4):
    """A method for actin/spectrin ring finding. It performs FFT 2D analysis
    and looks for maxima at 180 nm in the frequency spectrum."""

    # calculate new fft2
    fft2output = np.real(np.fft.fftshift(np.fft.fft2(data)))

    # take abs value and log10 for better visualization
    fft2output = np.abs(np.log10(fft2output))

    # calculate local intensity maxima
    coord = peak_local_max(fft2output, min_distance=2, threshold_rel=thres)

    # take first 3 max
    coord = firstNmax(coord, fft2output, N=3)

    # size of the subimqge of interest
    A = np.shape(data)[0]

    # max and min radius in pixels, 9 -> 220 nm, 12 -> 167 nm
    rmin, rmax = (9, 12)

    # auxarrays: ringBool, D

    # ringBool is checked to define wether there are rings or not
    ringBool = []

    # D saves the distances of local maxima from the centre of the fft2
    D = []

    # loop for calculating all the distances d, elements of array D
    for i in np.arange(0, np.shape(coord)[0]):
        d = np.linalg.norm([A/2, A/2], coord[i])
        D.append(d)
        if A*(rmin/100) < d < A*(rmax/100):
            ringBool.append(1)

    # condition for ringBool: all elements d must correspond to
    # periods between 170 and 220 nm
    rings = np.sum(ringBool) == np.shape(coord)[0]-1 and np.sum(ringBool) > 0

    return fft2output, coord, (rmin, rmax), rings


def pointsMethod(self, data, thres=.3):
    """A method for actin/spectrin ring finding. It finds local maxima in the
    image (points) and then if there are three or more in a row considers that
    to be rings."""

    points = peak_local_max(data, min_distance=6, threshold_rel=thres)
    points = firstNmax(points, data, N=7)

    D = []

    if points == []:
        rings = False

    else:
        dmin = 8
        dmax = 11

        # look up every point
        for i in np.arange(0, np.shape(points)[0]-1):
            # calculate the distance of every point to the others
            for j in np.arange(i + 1, np.shape(points)[0]):
                d1 = np.linalg.norm(points[i], points[j])
                # if there are two points at the right distance then
                if dmin < d1 < dmax:
                    for k in np.arange(0, np.shape(points)[0]-1):
                        # check the distance between the last point
                        # and the other points in the list
                        if k != i & k != j:
                            d2 = np.linalg.norm(points[j], points[k])

                        else:
                            d2 = 0

                        # calculate the angle between vector i-j
                        # and j-k with i, j, k points
                        v1 = points[i]-points[j]
                        v2 = points[j]-points[k]
                        t = cosTheta(v1, v2)

                        # if point k is at right distance from point j and
                        # the angle is flat enough
                        if dmin < d2 < dmax and np.abs(t) > 0.8:
                            # save the three points and plot the connections
                            D.append([points[i], points[j], points[k]])

                        else:
                            pass

        rings = len(D) > 0

    return points, D, rings
