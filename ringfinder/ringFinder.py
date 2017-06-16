# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import time
import math
import numpy as np
from scipy import ndimage as ndi
import tifffile as tiff
from PIL import Image

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import matplotlib.pyplot as plt
import matplotlib.colors
# from matplotlib import rc

import ringfinder.utils as utils
import ringfinder.tools as tools
import ringfinder.pyqtsubclass as pyqtsub

# rc('text', usetex=True)
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})


class Gollum(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0
        self.testData = False

        self.setWindowTitle('Gollum: the Ring Finder')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Run')
        batchSTORMAct = QtGui.QAction('Analyze batch of STORM images...', self)
        batchSTEDAct = QtGui.QAction('Analyze batch of STED images...', self)
        batchSTORMAct.triggered.connect(self.batchSTORM)
        batchSTEDAct.triggered.connect(self.batchSTED)
        fileMenu.addAction(batchSTORMAct)
        fileMenu.addAction(batchSTEDAct)
        fileMenu.addSeparator()

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.closeAllWindows)
        fileMenu.addAction(exitAction)

        self.folderStatus = QtGui.QLabel('Ready', self)
        self.statusBar().addPermanentWidget(self.folderStatus, 1)
        self.fileStatus = QtGui.QLabel('Ready', self)
        self.statusBar().addPermanentWidget(self.fileStatus)

        # Main Widgets' layout
        self.mainLayout = QtGui.QGridLayout()
        self.cwidget.setLayout(self.mainLayout)

        # Image with correlation results
        self.corrImgWidget = pg.GraphicsLayoutWidget()
        self.corrImgItem = pg.ImageItem()
        self.corrVb = self.corrImgWidget.addViewBox(col=0, row=0)
        self.corrVb.setAspectLocked(True)
        self.corrVb.addItem(self.corrImgItem)
        self.corrImgHist = pg.HistogramLUTItem()
        self.corrImgHist.gradient.loadPreset('thermal')
        self.corrImgHist.setImageItem(self.corrImgItem)
        self.corrImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.corrImgWidget.addItem(self.corrImgHist)
        self.corrResult = pg.ImageItem()
        self.corrVb.addItem(self.corrResult)

        # Image with ring results
        self.ringImgWidget = pg.GraphicsLayoutWidget()
        self.ringImgItem = pg.ImageItem()
        self.ringVb = self.ringImgWidget.addViewBox(col=0, row=0)
        self.ringVb.setAspectLocked(True)
        self.ringVb.addItem(self.ringImgItem)
        self.ringImgHist = pg.HistogramLUTItem()
        self.ringImgHist.gradient.loadPreset('thermal')
        self.ringImgHist.setImageItem(self.ringImgItem)
        self.ringImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.ringImgWidget.addItem(self.ringImgHist)
        self.ringResult = pg.ImageItem()
        self.ringVb.addItem(self.ringResult)

        # Separate frame for loading controls
        loadFrame = QtGui.QFrame(self)
        loadFrame.setFrameStyle(QtGui.QFrame.Panel)
        loadLayout = QtGui.QGridLayout()
        loadFrame.setLayout(loadLayout)
        loadTitle = QtGui.QLabel('<strong>Load image</strong>')
        loadTitle.setTextFormat(QtCore.Qt.RichText)
        loadLayout.addWidget(loadTitle, 0, 0, 1, 2)
        loadLayout.addWidget(QtGui.QLabel('STORM pixel [nm]'), 1, 0)
        self.STORMPxEdit = QtGui.QLineEdit()
        loadLayout.addWidget(self.STORMPxEdit, 1, 1)
        loadLayout.addWidget(QtGui.QLabel('STORM magnification'), 2, 0)
        self.magnificationEdit = QtGui.QLineEdit()
        loadLayout.addWidget(self.magnificationEdit, 2, 1)
        self.loadSTORMButton = QtGui.QPushButton('Load STORM Image')
        loadLayout.addWidget(self.loadSTORMButton, 3, 0, 1, 2)
        loadLayout.addWidget(QtGui.QLabel('STED pixel [nm]'), 4, 0)
        self.STEDPxEdit = QtGui.QLineEdit()
        loadLayout.addWidget(self.STEDPxEdit, 4, 1)
        self.loadSTEDButton = QtGui.QPushButton('Load STED Image')
        loadLayout.addWidget(self.loadSTEDButton, 5, 0, 1, 2)
        loadLayout.setColumnMinimumWidth(1, 40)
        loadFrame.setFixedHeight(180)

        # Ring finding method settings frame
        self.intThrLabel = QtGui.QLabel('#sigmas threshold from mean')
        self.intThresEdit = QtGui.QLineEdit()
        self.sigmaEdit = QtGui.QLineEdit()
        self.lineLengthEdit = QtGui.QLineEdit()
        self.roiSizeEdit = QtGui.QLineEdit()
        self.corrSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.corrSlider.setMinimum(0)
        self.corrSlider.setMaximum(250)   # Divide by 1000 to get corr value
        self.corrSlider.setValue(200)
        self.corrThresEdit = QtGui.QLineEdit()
        self.minAreaEdit = QtGui.QLineEdit()
        self.corrSlider.valueChanged[int].connect(self.sliderChange)
        self.showCorrMapCheck = QtGui.QCheckBox('Show coefficient map', self)
        self.thetaStepEdit = QtGui.QLineEdit()
        self.deltaThEdit = QtGui.QLineEdit()
        self.sinPowerEdit = QtGui.QLineEdit()
        self.corrButton = QtGui.QPushButton('Run analysis')
        self.corrButton.setCheckable(True)
        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0, 1, 2)
        wvlenLabel = QtGui.QLabel('Rings periodicity [nm]')
        self.wvlenEdit = QtGui.QLineEdit()
        settingsLayout.addWidget(wvlenLabel, 1, 0)
        settingsLayout.addWidget(self.wvlenEdit, 1, 1)
        corrThresLabel = QtGui.QLabel('Discrimination threshold')
        settingsLayout.addWidget(corrThresLabel, 2, 0)
        settingsLayout.addWidget(self.corrThresEdit, 2, 1)
        settingsLayout.addWidget(self.corrSlider, 3, 0, 1, 2)
        settingsLayout.addWidget(self.showCorrMapCheck, 4, 0, 1, 2)
        settingsLayout.addWidget(self.corrButton, 5, 0, 1, 2)
        settingsLayout.setColumnMinimumWidth(1, 40)
        settingsFrame.setFixedHeight(180)

        # Load settings configuration and then connect the update
        try:
            tools.loadConfig(self)
        except:
            tools.saveDefaultConfig()
            tools.loadConfig(self)
        self.STORMPxEdit.editingFinished.connect(self.updateConfig)
        self.magnificationEdit.editingFinished.connect(self.updateConfig)
        self.STEDPxEdit.editingFinished.connect(self.updateConfig)
        self.roiSizeEdit.editingFinished.connect(self.updateConfig)
        self.sigmaEdit.editingFinished.connect(self.updateConfig)
        self.intThresEdit.editingFinished.connect(self.updateConfig)
        self.lineLengthEdit.editingFinished.connect(self.updateConfig)
        self.wvlenEdit.editingFinished.connect(self.updateConfig)
        self.sinPowerEdit.editingFinished.connect(self.updateConfig)
        self.thetaStepEdit.editingFinished.connect(self.updateConfig)
        self.deltaThEdit.editingFinished.connect(self.updateConfig)
        self.corrThresEdit.editingFinished.connect(self.updateConfig)

        self.buttonWidget = QtGui.QWidget()
        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        # layout of the three widgets
        self.mainLayout.addWidget(self.buttonWidget, 1, 0)
        corrLabel = QtGui.QLabel('Pearson coefficient')
        corrLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.mainLayout.addWidget(corrLabel, 0, 1)
        self.mainLayout.addWidget(self.corrImgWidget, 1, 1, 2, 1)
        ringLabel = QtGui.QLabel('Rings')
        ringLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.mainLayout.addWidget(ringLabel, 0, 2)
        self.mainLayout.addWidget(self.ringImgWidget, 1, 2, 2, 1)
        self.mainLayout.setColumnMinimumWidth(1, 600)
        self.mainLayout.setColumnMinimumWidth(2, 600)
        self.buttonWidget.setFixedWidth(225)

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)
        self.sigmaEdit.textChanged.connect(self.updateMasks)
        self.corrButton.clicked.connect(self.ringFinder)

        # Load sample STED image
        folder = os.path.join(os.getcwd(), 'ringfinder')
        if os.path.exists(folder):
            self.folder = folder
            self.loadSTED(os.path.join(folder, 'spectrinSTED.tif'))
        else:
            self.folder = os.getcwd()
            self.loadSTED(os.path.join(os.getcwd(), 'spectrinSTED.tif'))

    def updateConfig(self):
        tools.saveConfig(self)

    def sliderChange(self, value):
        self.corrThresEdit.setText(str(np.round(0.001*value, 2)))
        self.corrEditChange(str(value/1000))

    def corrEditChange(self, text):
        self.corrSlider.setValue(1000*float(text))
        if self.analyzed:
            self.corrThres = float(text)
            self.ringsBig = np.nan_to_num(self.localCorrBig) > self.corrThres
            self.ringsBig = self.ringsBig.astype(float)
            self.ringResult.setImage(np.fliplr(np.transpose(self.ringsBig)))

    def loadSTED(self, filename=None):
        self.loadImage(np.float(self.STEDPxEdit.text()), 'STED',
                       filename=filename)

    def loadSTORM(self, filename=None):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those 3px borders before running the
        # analysis.
        mag = np.float(self.magnificationEdit.text())
        load = self.loadImage(np.float(self.STORMPxEdit.text()), 'STORM',
                              crop=int(3*mag), filename=filename)
        if load:
            self.corrImgHist.setLevels(0, 3)
            self.ringImgHist.setLevels(0, 3)

    def loadImage(self, pxSize, tt, crop=0, filename=None):

        try:
            if not(isinstance(filename, str)):
                filetypes = ('Tiff file', '*.tif;*.tiff')
                self.filename = utils.getFilename('Load ' + tt + ' image',
                                                  [filetypes], self.folder)
            else:
                self.filename = filename

            if self.filename is not None:

                self.corrButton.setChecked(False)
                self.analyzed = False

                self.folder = os.path.split(self.filename)[0]
                self.crop = np.int(crop)
                self.pxSize = pxSize
                self.corrVb.clear()
                self.corrResult.clear()
                self.ringVb.clear()
                self.ringResult.clear()

                im = Image.open(self.filename)
                self.inputData = np.array(im).astype(np.float64)
                self.initShape = self.inputData.shape
                bound = (np.array(self.initShape) - self.crop).astype(np.int)
                self.inputData = self.inputData[self.crop:bound[0],
                                                self.crop:bound[1]]
                self.shape = self.inputData.shape

                # We need 1um n-sized subimages
                self.subimgPxSize = int(1000/self.pxSize)
                self.n = (np.array(self.shape)/self.subimgPxSize).astype(int)

                # If n*subimgPxSize < shape, we crop the image
                self.remanent = np.array(self.shape) - self.n*self.subimgPxSize
                self.inputData = self.inputData[:self.n[0]*self.subimgPxSize,
                                                :self.n[1]*self.subimgPxSize]
                self.shape = self.inputData.shape

                self.nblocks = np.array(self.inputData.shape)/self.n
                self.blocksInput = tools.blockshaped(self.inputData,
                                                     *self.nblocks)

                self.updateMasks()
                self.corrVb.addItem(self.corrImgItem)
                self.ringVb.addItem(self.ringImgItem)
                showIm = np.fliplr(np.transpose(self.inputData))
                self.corrImgItem.setImage(showIm)
                self.ringImgItem.setImage(showIm)

                self.grid = pyqtsub.Grid(self.corrVb, self.shape, self.n)

                self.corrVb.setLimits(xMin=-0.05*self.shape[0],
                                      xMax=1.05*self.shape[0], minXRange=4,
                                      yMin=-0.05*self.shape[1],
                                      yMax=1.05*self.shape[1], minYRange=4)
                self.ringVb.setLimits(xMin=-0.05*self.shape[0],
                                      xMax=1.05*self.shape[0], minXRange=4,
                                      yMin=-0.05*self.shape[1],
                                      yMax=1.05*self.shape[1], minYRange=4)

                self.dataMean = np.mean(self.inputData)
                self.dataStd = np.std(self.inputData)

                self.corrVb.addItem(self.corrResult)
                self.ringVb.addItem(self.ringResult)

                return True

            else:
                return False

        except OSError:
            self.fileStatus.setText('No file selected!')

    def updateMasks(self):
        """Binarization of image. """

        self.gaussSigma = np.float(self.sigmaEdit.text())/self.pxSize
        thr = np.float(self.intThresEdit.text())

        if self.testData:
            self.blocksInputS = [ndi.gaussian_filter(b, self.gaussSigma)
                                 for b in self.blocksInput]
            self.blocksInputS = np.array(self.blocksInputS)
            self.meanS = np.mean(self.blocksInputS, (1, 2))
            self.stdS = np.std(self.blocksInputS, (1, 2))
            thresholds = self.meanS + thr*self.stdS
            thresholds = thresholds.reshape(np.prod(self.n), 1, 1)
            mask = self.blocksInputS < thresholds
            self.blocksMask = np.array([bI < np.mean(bI) + thr*np.std(bI)
                                       for bI in self.blocksInputS])

            self.mask = tools.unblockshaped(mask, *self.inputData.shape)
            self.inputDataS = tools.unblockshaped(self.blocksInputS,
                                                  *self.shape)
        else:
            self.inputDataS = ndi.gaussian_filter(self.inputData,
                                                  self.gaussSigma)
            self.blocksInputS = tools.blockshaped(self.inputDataS,
                                                  *self.nblocks)
            self.meanS = np.mean(self.inputDataS)
            self.stdS = np.std(self.inputDataS)
            self.mask = self.inputDataS < self.meanS + thr*self.stdS
            self.blocksMask = tools.blockshaped(self.mask, *self.nblocks)

        self.showImS = np.fliplr(np.transpose(self.inputDataS))
        self.showMask = np.fliplr(np.transpose(self.mask))

    def ringFinder(self, show=True, batch=False):
        """RingFinder handles the input data, and then evaluates every subimg
        using the given algorithm which decides if there are rings or not.
        Subsequently gives the output data and plots it"""

        if self.corrButton.isChecked() or batch:

            self.corrResult.clear()
            self.ringResult.clear()

            # for each subimg, we apply the correlation method for ring finding
            intThr = np.float(self.intThresEdit.text())
            minLen = np.float(self.lineLengthEdit.text())/self.pxSize
            thetaStep = np.float(self.thetaStepEdit.text())
            deltaTh = np.float(self.deltaThEdit.text())
            wvlen = np.float(self.wvlenEdit.text())/self.pxSize
            sinPow = np.float(self.sinPowerEdit.text())
            cArgs = minLen, thetaStep, deltaTh, wvlen, sinPow

            # Single-core code
            self.localCorr = np.zeros(len(self.blocksInput))
            thres = self.meanS + intThr*self.stdS
            if not(self.testData):
                thres = thres*np.ones(self.blocksInput.shape)

            for i in np.arange(len(self.blocksInput)):
                block = self.blocksInput[i]
                blockS = self.blocksInputS[i]
                mask = self.blocksMask[i]
                # Block may be excluded from the analysis for two reasons.
                # Firstly, because the intensity for all its pixels may be
                # too low. Secondly, because the part of the block that
                # belongs toa neuron may be below an arbitrary 20% of the
                # block. We apply intensity threshold to smoothed data so we
                # don't catch tiny bright spots outside neurons
                neuronFrac = 1 - np.sum(mask)/np.size(mask)
                areaThres = 0.01*float(self.minAreaEdit.text())
                if np.any(blockS > thres[i]) and neuronFrac > areaThres:
                    output = tools.corrMethod(block, mask, *cArgs)
                    angle, corrTheta, corrMax, theta, phase = output
                    # Store results
                    self.localCorr[i] = corrMax
                else:
                    self.localCorr[i] = np.nan

            self.localCorr = self.localCorr.reshape(*self.n)
            self.updateGUI(self.localCorr)

        else:
            self.corrResult.clear()
            self.ringResult.clear()

    def updateGUI(self, localCorr):

        self.analyzed = True
        self.localCorr = localCorr

        # code for visualization of the output
        mag = np.array(self.inputData.shape)/self.n
        self.localCorrBig = np.repeat(self.localCorr, mag[0], 0)
        self.localCorrBig = np.repeat(self.localCorrBig, mag[1], 1)
        showIm = 100*np.fliplr(np.transpose(self.localCorrBig))
        self.corrResult.setImage(np.nan_to_num(showIm))
        self.corrResult.setZValue(10)    # make sure this image is on top
        self.corrResult.setOpacity(0.5)

        self.corrThres = float(self.corrThresEdit.text())
        self.ringsBig = np.nan_to_num(self.localCorrBig) > self.corrThres
        self.ringsBig = self.ringsBig.astype(float)
        self.ringResult.setImage(np.fliplr(np.transpose(self.ringsBig)))
        self.ringResult.setZValue(10)    # make sure this image is on top
        self.ringResult.setOpacity(0.5)

        if self.showCorrMapCheck.isChecked():
            plt.figure(figsize=(10, 8))
            data = self.localCorr.reshape(*self.n)
            data = np.flipud(data)
            maskedData = np.ma.array(data, mask=np.isnan(data))

            mx = np.max(maskedData)
            mn = np.min(maskedData)
            mp = (self.corrThres - mn)/(mx - mn)
            mp = np.min((mp, 1))
            mp = np.max((mp, 0))

            cmap = shiftedColorMap(matplotlib.cm.PuOr, midpoint=mp,
                                   name='shifted')
            heatmap = plt.pcolor(maskedData, cmap=cmap)
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.2f' % data[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',)
            plt.colorbar(heatmap)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.show()

    def batch(self, function, tech):
        try:
            filenames = utils.getFilenames('Load ' + tech + ' images',
                                           [('Tiff file', '*.tif;*.tiff')],
                                           self.folder)
            nfiles = len(filenames)
            function(filenames[0])
            corrArray = np.zeros((nfiles, self.n[0], self.n[1]))

            # Expand correlation array so it matches data shape
            corrExp = np.empty((nfiles, self.initShape[0], self.initShape[1]),
                               dtype=np.single)
            corrExp[:] = np.nan
            ringsExp = np.empty((nfiles, self.initShape[0], self.initShape[1]),
                                dtype=np.single)
            ringsExp[:] = np.nan

            path = os.path.split(filenames[0])[0]
            folder = os.path.split(path)[1]
            self.folderStatus.setText('Processing folder ' + path)
            print('Processing folder', path)
            t0 = time.time()

            # Make results directory if it doesn't exist
            resultsDir = os.path.join(path, 'results')
            if not os.path.exists(resultsDir):
                os.makedirs(resultsDir)
            resNames = [utils.insertFolder(p, 'results') for p in filenames]

            for i in np.arange(nfiles):
                print(os.path.split(filenames[i])[1])
                self.fileStatus.setText(os.path.split(filenames[i])[1])
                function(filenames[i])
                self.ringFinder(False, batch=True)
                corrArray[i] = self.localCorr

                bound = (np.array(self.initShape) - self.crop).astype(np.int)
                edge = bound - self.remanent
                corrExp[i, self.crop:edge[0],
                        self.crop:edge[1]] = self.localCorrBig

                # Save correlation values array
                corrName = utils.insertSuffix(resNames[i], '_correlation')
                tiff.imsave(corrName, corrExp[i], software='Gollum',
                            imagej=True,
                            resolution=(1000/self.pxSize, 1000/self.pxSize),
                            metadata={'spacing': 1, 'unit': 'um'})

            # Saving ring images
            ringsExp[corrExp < self.corrThres] = 0
            ringsExp[corrExp >= self.corrThres] = 1
            for i in np.arange(nfiles):
                # Save correlation values array
                ringName = utils.insertSuffix(resNames[i], '_rings')
                tiff.imsave(ringName, ringsExp[i], software='Gollum',
                            imagej=True,
                            resolution=(1000/self.pxSize, 1000/self.pxSize),
                            metadata={'spacing': 1, 'unit': 'um'})

            # save configuration file in the results folder
            tools.saveConfig(self, os.path.join(resultsDir, 'config'))

            # plot histogram of the correlation values
            hrange = (np.min(np.nan_to_num(corrArray)),
                      np.max(np.nan_to_num(corrArray)))
            y, x, _ = plt.hist(corrArray.flatten(), bins=20, range=hrange)
            x = (x[1:] + x[:-1])/2

            # Save data array as txt
            corrArrayFlat = corrArray.flatten()
            validCorr = corrArrayFlat[~np.isnan(corrArrayFlat)]
            validArr = np.repeat(np.arange(nfiles), np.prod(self.n))
            validArr = validArr[~np.isnan(corrArrayFlat)]
            valuesTxt = os.path.join(resultsDir, folder + 'corr_values.txt')
            corrByN = np.stack((validCorr, validArr), 1)
            np.savetxt(valuesTxt, corrByN, fmt='%f\t%i')

            groupedCorr = [validCorr[np.where(validArr == i)]
                           for i in np.arange(nfiles)]
            groupedCorr = [x for x in groupedCorr if len(x) > 0]
            nfiles = len(groupedCorr)
            meanCorrs = [np.mean(d) for d in groupedCorr]

            validCorrRing = validCorr[np.where(validCorr > self.corrThres)]
            validArrRing = validArr[np.where(validCorr > self.corrThres)]
            groupedCorrRing = [validCorrRing[np.where(validArrRing == i)]
                               for i in np.arange(nfiles)]
            ringFracs = [len(groupedCorrRing[i])/len(groupedCorr[i])
                         for i in np.arange(nfiles)]
            meanRingCorrs = np.array([np.mean(d) for d in groupedCorrRing])
            meanRingCorrs = meanRingCorrs[~np.isnan(meanRingCorrs)]

            n = corrArray.size - np.count_nonzero(np.isnan(corrArray))
            nring = np.sum(validCorr > self.corrThres)
            validRingCorr = validCorr[validCorr > self.corrThres]
            ringFrac = nring/n

            # Err estimation: stat err (binomial distribution, p=ringFrac)
            statVar = ringFrac*(1 - ringFrac)/n
            fracStd = math.sqrt(statVar)

            statCorrVar = np.var(validCorr)/n
            corrStd = math.sqrt(statCorrVar)

            statRingCorrVar = np.var(validRingCorr)/nring
            ringCorrStd = math.sqrt(statRingCorrVar)

            # Plotting
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 7.5))
            plt.bar(x, y, align='center', width=(x[1] - x[0]), color="#3F5D7D")
            plt.plot((self.corrThres, self.corrThres), (0, np.max(y)), '--',
                     color='r', linewidth=2)
            text = ('Pearson coefficient threshold = {0:.2f} \n'
                    'n = {1}; nrings = {2} \n'
                    'PSS fraction = {3:.2f} $\pm$ {4:.2f} \n'
                    'mean coefficient = {5:.3f} $\pm$ {6:.3f}\n'
                    'mean ring coefficient = {7:.3f} $\pm$ {8:.3f}')
            text = text.format(self.corrThres, n, nring,
                               np.mean(ringFracs), fracStd,
                               np.mean(meanCorrs), corrStd,
                               np.mean(meanRingCorrs), ringCorrStd)
            plt.text(0.75*plt.axis()[1], 0.83*plt.axis()[3], text,
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white'), fontsize=20)
            plt.xlabel('Pearson correlation coefficient', fontsize=35)
            plt.tick_params(axis='both', labelsize=25)
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(resultsDir, folder + 'corr_hist.pdf'),
                        dpi=300)
            plt.savefig(os.path.join(resultsDir, folder + 'corr_hist.png'),
                        dpi=300)
            plt.close()

            folder = os.path.split(path)[1]
            text = 'Folder ' + folder + ' done in {0:.0f} seconds'
            print(text.format(time.time() - t0))
            self.folderStatus.setText(text.format(time.time() - t0))
            self.fileStatus.setText('                 ')

        except IndexError:
            self.fileStatus.setText('No file selected!')

    def batchSTORM(self):
        self.batch(self.loadSTORM, 'STORM')

    def batchSTED(self):
        self.batch(self.loadSTED, 'STED')


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    http://stackoverflow.com/questions/7404116/
    defining-the-midpoint-of-a-colormap-in-matplotlib
    '''

    if midpoint == stop:
        newcmap = truncate_colormap(cmap, 0, 0.5)

    elif midpoint == start:
        newcmap = truncate_colormap(cmap, 0.5, 1)

    else:
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    plt.register_cmap(cmap=newcmap)

    return newcmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = Gollum()
    win.show()
    app.exec_()
