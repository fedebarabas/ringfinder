# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas

# TODO: QTHREADS ARE NOT WORKING AS THEY WERE SUPPOSED TO
"""

import os
import time
import numpy as np
from scipy import ndimage as ndi
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import ringfinder.utils as utils
import ringfinder.tools as tools


class Gollum(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0

        self.setWindowTitle('Gollum: the Ring Finder')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Run')
        batchSTORMAction = QtGui.QAction('Analyze batch of STORM images...',
                                         self)
        batchSTEDAction = QtGui.QAction('Analyze batch of STED images...',
                                        self)

        batchSTORMAction.triggered.connect(self.batchSTORM)
        batchSTEDAction.triggered.connect(self.batchSTED)
        fileMenu.addAction(batchSTORMAction)
        fileMenu.addAction(batchSTEDAction)
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
        self.corrResult.setZValue(10)    # make sure this image is on top
        self.corrResult.setOpacity(0.5)
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
        self.ringResult.setZValue(10)    # make sure this image is on top
        self.ringResult.setOpacity(0.5)
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
        self.STORMPxEdit = QtGui.QLineEdit('13.3')
        loadLayout.addWidget(self.STORMPxEdit, 1, 1)
        loadLayout.addWidget(QtGui.QLabel('STORM magnification'), 2, 0)
        self.magnificationEdit = QtGui.QLineEdit('10')
        loadLayout.addWidget(self.magnificationEdit, 2, 1)
        self.loadSTORMButton = QtGui.QPushButton('Load STORM Image')
        loadLayout.addWidget(self.loadSTORMButton, 3, 0, 1, 2)
        loadLayout.addWidget(QtGui.QLabel('STED pixel [nm]'), 4, 0)
        self.STEDPxEdit = QtGui.QLineEdit('20')
        loadLayout.addWidget(self.STEDPxEdit, 4, 1)
        self.loadSTEDButton = QtGui.QPushButton('Load STED Image')
        loadLayout.addWidget(self.loadSTEDButton, 5, 0, 1, 2)
        loadLayout.setColumnMinimumWidth(1, 40)
        loadFrame.setFixedHeight(180)

        # Ring finding method settings frame
        self.intThrLabel = QtGui.QLabel('#sigmas threshold from mean')
        self.intThresEdit = QtGui.QLineEdit()
#        gaussianSigmaLabel = QtGui.QLabel('Gaussian filter sigma [nm]')
        self.sigmaEdit = QtGui.QLineEdit()
#        minLenLabel = QtGui.QLabel('Direction lines min length [nm]')
        self.lineLengthEdit = QtGui.QLineEdit('300')
        defaultThreshold = 0.12
        self.corrThresEdit = QtGui.QLineEdit(str(defaultThreshold))
        self.corrSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.corrSlider.setMinimum(0)
        self.corrSlider.setMaximum(250)   # Divide by 1000 to get corr value
        self.corrSlider.setValue(1000*defaultThreshold)
        self.corrSlider.valueChanged[int].connect(self.sliderChange)
        self.showCorrMapCheck = QtGui.QCheckBox('Show correlation map', self)
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaAngleEdit = QtGui.QLineEdit('20')
        self.sinPowerEdit = QtGui.QLineEdit('6')
        self.wvlenEdit = QtGui.QLineEdit('180')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.corrButton.setCheckable(True)
        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0, 1, 2)
#        settingsLayout.addWidget(self.intThrLabel, 1, 0)
#        settingsLayout.addWidget(self.intThresEdit, 1, 1)
#        gaussianSigmaLabel = QtGui.QLabel('Gaussian filter sigma [nm]')
#        settingsLayout.addWidget(gaussianSigmaLabel, 2, 0)
#        settingsLayout.addWidget(self.sigmaEdit, 2, 1)
#        minLenLabel = QtGui.QLabel('Direction lines min length [nm]')
#        settingsLayout.addWidget(minLenLabel, 3, 0)
#        settingsLayout.addWidget(self.lineLengthEdit, 3, 1)
        settingsLayout.addWidget(QtGui.QLabel('Correlation threshold'), 4, 0)
        settingsLayout.addWidget(self.corrThresEdit, 4, 1)
        settingsLayout.addWidget(self.corrSlider, 5, 0, 1, 2)
        settingsLayout.addWidget(self.showCorrMapCheck, 6, 0, 1, 2)
#        settingsLayout.addWidget(QtGui.QLabel('Angular step [°]'), 6, 0)
#        settingsLayout.addWidget(self.thetaStepEdit, 6, 1)
#        settingsLayout.addWidget(QtGui.QLabel('Delta Angle [°]'), 7, 0)
#        settingsLayout.addWidget(self.deltaAngleEdit, 7, 1)
#        powLabel = QtGui.QLabel('Sinusoidal pattern power')
#        settingsLayout.addWidget(powLabel, 8, 0)
#        settingsLayout.addWidget(self.sinPowerEdit, 8, 1)
#        wvlenLabel = QtGui.QLabel('wvlen of corr pattern [nm]')
#        settingsLayout.addWidget(wvlenLabel, 9, 0)
#        settingsLayout.addWidget(self.wvlenEdit, 9, 1)
        settingsLayout.addWidget(self.corrButton, 10, 0, 1, 2)
        loadLayout.setColumnMinimumWidth(1, 40)
        settingsFrame.setFixedHeight(150)

        self.buttonWidget = QtGui.QWidget()
        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        # layout of the three widgets
        self.mainLayout.addWidget(self.buttonWidget, 1, 0)
        self.mainLayout.addWidget(QtGui.QLabel('Correlation'), 0, 1)
        self.mainLayout.addWidget(self.corrImgWidget, 1, 1, 2, 1)
        self.mainLayout.addWidget(QtGui.QLabel('Rings'), 0, 2)
        self.mainLayout.addWidget(self.ringImgWidget, 1, 2, 2, 1)
        self.mainLayout.setColumnMinimumWidth(1, 600)
        self.mainLayout.setColumnMinimumWidth(2, 600)
        self.buttonWidget.setFixedWidth(200)

        # Load sample STED image
        self.initialdir = os.getcwd()
        self.loadSTED(os.path.join(self.initialdir, 'labnanofisica',
                                   'ringfinder', 'spectrinSTED.tif'))

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)
        self.sigmaEdit.textChanged.connect(self.changeSigma)
        self.corrButton.clicked.connect(self.ringFinder)

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
        prevSigma = self.sigmaEdit.text()
        prevThres = self.intThresEdit.text()
        self.sigmaEdit.setText('100')
        self.intThresEdit.setText('0.5')
        load = self.loadImage(np.float(self.STEDPxEdit.text()), 'STED',
                              filename=filename)
        if not(load):
            self.sigmaEdit.setText(prevSigma)
            self.intThresEdit.setText(prevThres)

    def loadSTORM(self, filename=None):
        prevSigma = self.sigmaEdit.text()
        prevThres = self.intThresEdit.text()
        self.corrImgHist.setLevels(0, 3)
        self.ringImgHist.setLevels(0, 3)
        self.sigmaEdit.setText('100')
        self.intThresEdit.setText('0.5')
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those 3px borders before running the
        # analysis.
        mag = np.float(self.magnificationEdit.text())
        load = self.loadImage(np.float(self.STORMPxEdit.text()), 'STORM',
                              crop=3*mag, filename=filename)
        if not(load):
            self.sigmaEdit.setText(prevSigma)
            self.intThresEdit.setText(prevThres)

    def loadImage(self, pxSize, tt, crop=0, filename=None):

        try:

            if not(isinstance(filename, str)):
                self.filename = utils.getFilename('Load ' + tt + ' image',
                                                  [('Tiff file', '.tif')],
                                                  self.initialdir)
            else:
                self.filename = filename

            if self.filename is not None:

                self.corrButton.setChecked(False)
                self.analyzed = False

                self.initialdir = os.path.split(self.filename)[0]
                self.crop = np.int(crop)
                self.pxSize = pxSize

                im = Image.open(self.filename)
                self.inputData = np.array(im).astype(np.float64)
                self.initShape = self.inputData.shape
                bound = (np.array(self.initShape) - self.crop).astype(np.int)
                self.inputData = self.inputData[self.crop:bound[0],
                                                self.crop:bound[1]]
                self.shape = self.inputData.shape
                self.changeSigma()

                # We need 1um n-sized subimages
                self.subimgPxSize = 1000/self.pxSize
                self.n = (np.array(self.shape)/self.subimgPxSize).astype(int)

                self.dataMean = np.mean(self.inputData)
                self.dataStd = np.std(self.inputData)

                self.updateInput(self.inputData, self.n)

                return True

            else:
                return False

        except OSError:
            self.fileStatus.setText('No file selected!')
#            print("No file selected!")

    def updateInput(self, inputData, n):
        self.corrVb.clear()
        self.corrResult.clear()
        self.ringVb.clear()
        self.ringResult.clear()

        self.corrVb.addItem(self.corrImgItem)
        self.ringVb.addItem(self.ringImgItem)
        self.corrVb.addItem(self.corrResult)
        self.ringVb.addItem(self.ringResult)

        self.corrImgItem.setImage(np.fliplr(np.transpose(inputData)))
        self.ringImgItem.setImage(np.fliplr(np.transpose(inputData)))

        shape = inputData.shape
        self.grid = tools.Grid(self.corrVb, shape, n)
        self.corrVb.setLimits(xMin=-0.05*shape[0], xMax=1.05*shape[0],
                              yMin=-0.05*shape[1], yMax=1.05*shape[1],
                              minXRange=4, minYRange=4)
        self.ringVb.setLimits(xMin=-0.05*shape[0], xMax=1.05*shape[0],
                              yMin=-0.05*shape[1], yMax=1.05*shape[1],
                              minXRange=4, minYRange=4)

    def changeSigma(self):
        self.gaussSigma = np.float(self.sigmaEdit.text())/self.pxSize
        self.inputDataS = ndi.gaussian_filter(self.inputData,
                                              self.gaussSigma)
        self.meanS = np.mean(self.inputDataS)
        self.stdS = np.std(self.inputDataS)

        # binarization of image
        thr = np.float(self.intThresEdit.text())
        self.mask = self.inputDataS < self.meanS + thr*self.stdS

    def ringFinder(self, show=True, batch=False):
        """RingFinder handles the input data, and then evaluates every subimg
        using the given algorithm which decides if there are rings or not.
        Subsequently gives the output data and plots it."""

        if self.corrButton.isChecked():

            # shape the data into the subimg that we need for the analysis
            nblocks = np.array(self.inputData.shape)/self.n
            blocksInput = tools.blockshaped(self.inputData, *nblocks)
            blocksInputS = tools.blockshaped(self.inputDataS, *nblocks)
            blocksMask = tools.blockshaped(self.mask, *nblocks)

            # for each subimg, we apply the correlation method for ring finding
            intThr = np.float(self.intThresEdit.text())
            corrThres = np.float(self.corrThresEdit.text())
            minLen = np.float(self.lineLengthEdit.text())/self.pxSize
            thetaStep = np.float(self.deltaAngleEdit.text())
            deltaTh = np.float(self.deltaAngleEdit.text())
            wvlen = np.float(self.wvlenEdit.text())/self.pxSize
            sinPow = np.float(self.sinPowerEdit.text())
            cArgs = corrThres, minLen, thetaStep, deltaTh, wvlen, sinPow

            args = self.n, blocksInput, blocksInputS, blocksMask, intThr, cArgs
            self.localCorr = np.zeros(len(blocksInput))
            self.singleObj = Single(*args)
            self.updateFileStatus()
#            self.singleObj.signals.start.connect(self.updateFileStatus)
            self.singleObj.doneSignal.connect(self.updateOutput)
            self.workerThread = QtCore.QThread(self)
            self.singleObj.moveToThread(self.workerThread)
            self.workerThread.started.connect(self.singleObj.start)
            self.workerThread.start()

        else:
            self.corrResult.clear()
            self.ringResult.clear()

    def updateFileStatus(self):
        self.fileStatus.setText(os.path.split(self.filename)[1])

    def updateOutput(self, localCorr):

        self.workerThread.quit()

        # code for visualization of the output
        self.corrResult.clear()
        self.ringResult.clear()

        self.analyzed = True
        self.localCorr = localCorr

        # code for visualization of the output
        mag = np.array(self.inputData.shape)/self.n
        self.localCorrBig = np.repeat(self.localCorr, mag[0], 0)
        self.localCorrBig = np.repeat(self.localCorrBig, mag[1], 1)
        showIm = 100*np.fliplr(np.transpose(self.localCorrBig))
        self.corrResult.setImage(np.nan_to_num(showIm))

        self.corrThres = float(self.corrThresEdit.text())
        self.ringsBig = np.nan_to_num(self.localCorrBig) > self.corrThres
        self.ringsBig = self.ringsBig.astype(float)
        self.ringResult.setImage(np.fliplr(np.transpose(self.ringsBig)))

        if self.showCorrMapCheck.isChecked():
            plt.figure(figsize=(10, 8))
            data = self.localCorr.reshape(*self.n)
            data = np.flipud(data)
            maskedData = np.ma.array(data, mask=np.isnan(data))
            heatmap = plt.pcolor(maskedData, cmap='inferno')
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',)
            plt.colorbar(heatmap)
            plt.show()

    def updateBar(self, text):
        self.fileStatus.setText(text)

    def batch(self, tech):
        try:

            if tech == 'STORM':
                pxSize = np.float(self.STORMPxEdit.text())
                crop = 3*np.float(self.magnificationEdit.text())

            elif tech == 'STED':
                pxSize = np.float(self.STEDPxEdit.text())
                crop = 0

            filenames = utils.getFilenames('Load ' + tech + ' images',
                                           [('Tiff file', '.tif')],
                                           self.initialdir)

#            self.batchWorker = Worker(*args)
#            self.batchWorker.doneSignal.connect(self.updateOutput)
#            self.batchWorkerThread = QtCore.QThread(self)
#            self.batchWorker.moveToThread(self.batchWorkerThread)
#            self.batchWorkerThread.started.connect(self.batchWorker.start)
#            self.batchWorkerThread.start()
#            nfiles = len(filenames)
#            function(filenames[0])
#            corrArray = np.zeros((nfiles, self.n[0], self.n[1]))
#
#            # Expand correlation array so it matches data shape
#            corrExp = np.empty((nfiles, self.initShape[0], self.initShape[1]),
#                               dtype=np.single)
#            corrExp[:] = np.nan
#            ringsExp = np.empty((nfiles, self.initShape[0], self.initShape[1]),
#                                dtype=np.single)
#            ringsExp[:] = np.nan
#
#            self.iBatch = 0
#            self.worker.doneSignal.connect(self.stepBatch)

            intThr = np.float(self.intThresEdit.text())
            gaussSigma = np.float(self.sigmaEdit.text())
            corrThres = np.float(self.corrThresEdit.text())
            minLen = np.float(self.lineLengthEdit.text())/self.pxSize
            thetaStep = np.float(self.deltaAngleEdit.text())
            deltaTh = np.float(self.deltaAngleEdit.text())
            wvlen = np.float(self.wvlenEdit.text())/self.pxSize
            sinPow = np.float(self.sinPowerEdit.text())
            cArgs = corrThres, minLen, thetaStep, deltaTh, wvlen, sinPow

            self.batchObj = Batch(filenames, pxSize, crop, gaussSigma, intThr,
                                  cArgs)
            self.batchThread = QtCore.QThread(self)
            self.batchObj.startSignal.connect(self.updateBar)
            self.batchObj.moveToThread(self.batchThread)
            self.batchThread.started.connect(self.batchObj.start)
            self.batchThread.start()

#            path = os.path.split(filenames[0])[0]
##            folder = os.path.split(path)[1]
#            self.folderStatus.setText('Processing folder ' + path)
#            print('Processing folder', path)
#            t0 = time.time()
#            for i in np.arange(self.nfiles):
#                print(os.path.split(filenames[i])[1])
##                self.fileStatus.setText(os.path.split(filenames[i])[1])
#                function(filenames[i])
#                self.ringFinder(False, batch=True)
##                corrArray[i] = self.localCorr
##
##                bound = (np.array(self.initShape) - self.crop).astype(np.int)
##                corrExp[i, self.crop:bound[0],
##                        self.crop:bound[1]] = self.localCorrBig
##
##                # Save correlation values array
##                corrName = utils.insertSuffix(filenames[i], '_correlation')
##                tiff.imsave(corrName, corrExp[i], software='Gollum',
##                            imagej=True,
##                            resolution=(1000/self.pxSize, 1000/self.pxSize),
##                            metadata={'spacing': 1, 'unit': 'um'})
#
#            # Saving ring images
#            ringsExp[corrExp < self.corrThres] = 0
#            ringsExp[corrExp >= self.corrThres] = 1
#            for i in np.arange(nfiles):
#                # Save correlation values array
#                ringName = utils.insertSuffix(filenames[i], '_rings')
#                tiff.imsave(ringName, ringsExp[i], software='Gollum',
#                            imagej=True,
#                            resolution=(1000/self.pxSize, 1000/self.pxSize),
#                            metadata={'spacing': 1, 'unit': 'um'})
#
#            # plot histogram of the correlation values
#            y, x, _ = plt.hist(corrArray.flatten(), bins=60,
#                               range=(np.min(np.nan_to_num(corrArray)),
#                                      np.max(np.nan_to_num(corrArray))))
#            x = (x[1:] + x[:-1])/2
#
## ==============================================================================
##             # Code for saving images that rely on bulk analysis
##             ringsExp = np.zeros((nfiles, self.initShape[0],
##                                  self.initShape[1]),
##                                 dtype=np.single)
##             corrExp = np.zeros((nfiles, self.initShape[0],
##                                 self.initShape[1]),
##                                 dtype=np.single)
##             for i in np.arange(len(filenames)):
##                 expanded = np.repeat(np.repeat(corrArray[i], m[1], axis=1),
##                                      m[0], axis=0)
## #                corrExp[i, self.crop:self.initShape[0] - self.crop,
## #                        self.crop:self.initShape[1] - self.crop] = expanded
##                 tiff.imsave(utils.insertSuffix(filenames[i], '_correlation'),
##                             corrExp[i].astype(np.single), software='Gollum',
##                             imagej=True,
##                             resolution=(1000/self.pxSize, 1000/self.pxSize),
##                             metadata={'spacing': 1, 'unit': 'um'})
##
##                 limits = np.array(self.initShape) - self.crop
##                 rings = np.nan_to_num(corrArray[i]) > self.corrThres
##                 ringsExp[i, self.crop:limits[0],
##                          self.crop:limits[1]] = rings.astype(float)
##                 tiff.imsave(utils.insertSuffix(filenames[i], '_rings'),
##                             ringsExp[i], software='Gollum', imagej=True,
##                             resolution=(1000/self.pxSize, 1000/self.pxSize),
##                             metadata={'spacing': 1, 'unit': 'um'})
## ==============================================================================
#
#            # Plotting
#            plt.figure(0)
#            validCorr = corrArray[~np.isnan(corrArray)]
#            n = corrArray.size - np.count_nonzero(np.isnan(corrArray))
#            ringFrac = np.sum(validCorr > self.corrThres) / n
#            ringStd = math.sqrt(ringFrac*(1 - ringFrac)/n)
#            plt.bar(x, y, align='center', width=(x[1] - x[0]))
#            plt.plot((self.corrThres, self.corrThres), (0, np.max(y)), 'r--',
#                     linewidth=2)
#            text = 'ringFrac={0:.3f}\pm{1:.3f} \ncorrelation threshold={2:.2f}'
#            plt.text(0.8*plt.axis()[1], 0.8*plt.axis()[3],
#                     text.format(ringFrac, ringStd, self.corrThres),
#                     horizontalalignment='center', verticalalignment='center',
#                     bbox=dict(facecolor='white'))
#            plt.title("Correlations Histogram")
#            plt.xlabel("Value")
#            plt.ylabel("Frequency")
#            plt.savefig(os.path.join(path, folder + 'corr_hist'))
#            plt.close()
#
##            print('Done in {0:.0f} seconds'.format(time.time() - t0))
#            folder = os.path.split(path)[1]
#            text = 'Folder ' + folder + ' done in {0:.0f} seconds'
#            self.folderStatus.setText(text.format(time.time() - t0))
##            self.statusBar().showMessage('                 ')
#            self.fileStatus.setText('                 ')

        except IndexError:
            self.fileStatus.setText('No file selected!')

#    def stepBatch(self):
#
#        self.corrArray[i] = self.localCorr
#
#        bound = (np.array(self.initShape) - self.crop).astype(np.int)
#        self.corrExp[i, self.crop:bound[0],
#                     self.crop:bound[1]] = self.localCorrBig
#
#        # Save correlation values array
#        corrName = utils.insertSuffix(self.filenames[i], '_correlation')
#        tiff.imsave(corrName, self.corrExp[i], software='Gollum', imagej=True,
#                    resolution=(1000/self.pxSize, 1000/self.pxSize),
#                    metadata={'spacing': 1, 'unit': 'um'})

    def batchSTORM(self):
        self.batch('STORM')

    def batchSTED(self):
        self.batch('STED')


class WorkerSignals(QtCore.QObject):
    start = QtCore.pyqtSignal()
    done = QtCore.pyqtSignal(np.ndarray)


# Base example found in http://stackoverflow.com/a/13909749/2834480
class Worker(QtCore.QRunnable):

    def __init__(self, n, blocksInput, blocksInputS, blocksMask, intThr, cArgs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.signals = WorkerSignals()

        self.n = n
        self.blocksInput = blocksInput
        self.blocksInputS = blocksInputS
        self.blocksMask = blocksMask
        self.intThr = intThr
        self.cArgs = cArgs
        self.meanS = np.mean(self.blocksInputS)
        self.stdS = np.std(self.blocksInputS)

    def run(self):

        self.signals.start.emit()
        localCorr = np.zeros(len(self.blocksInput))

        # Single-core code
        for i in np.arange(len(self.blocksInput)):
            rings = False
            block = self.blocksInput[i]
            blockS = self.blocksInputS[i]
            mask = self.blocksMask[i]
            # Block may be excluded from the analysis for two reasons.
            # Firstly, because the intensity for all its pixels may be
            # too low. Secondly, because the part of the block that
            # belongs toa neuron may be below an arbitrary 30% of the
            # block. We apply intensity threshold to smoothed data so we
            # don't catch tiny bright spots outside neurons
            neuronFrac = 1 - np.sum(mask)/np.size(mask)
            thres = self.meanS + self.intThr*self.stdS
            if np.any(blockS > thres) and neuronFrac > 0.25:
                output = tools.corrMethod(block, mask, *self.cArgs)
                angle, corrTheta, corrMax, theta, phase, rings = output
                # Store results
                localCorr[i] = corrMax
            else:
                localCorr[i] = np.nan

        localCorr = localCorr.reshape(*self.n)
        self.signals.done.emit(localCorr)


class Single(QtCore.QObject):

    doneSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, n, blocksInput, blocksInputS, blocksMask, intThr, cArgs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.signals = WorkerSignals()

        self.n = n
        self.blocksInput = blocksInput
        self.blocksInputS = blocksInputS
        self.blocksMask = blocksMask
        self.intThr = intThr
        self.cArgs = cArgs
        self.meanS = np.mean(self.blocksInputS)
        self.stdS = np.std(self.blocksInputS)

    def sendResult(self, result):
        self.doneSignal.emit(result)

    def start(self):
        worker = Worker(self.n, self.blocksInput, self.blocksInputS,
                        self.blocksMask, self.intThr, self.cArgs)
        worker.signals.done.connect(self.sendResult)
        worker.run()


class Batch(QtCore.QObject):

    startSignal = QtCore.pyqtSignal(str)

    def __init__(self, files, pxSize, crop, gaussSigma, intThres, cArgs):
        super().__init__()

        self.files = files
        self.pxSize = pxSize
        self.crop = crop
        self.gaussSigma = gaussSigma/self.pxSize
        self.intThres = intThres
        self.cArgs = cArgs

        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(1)

        self.nfiles = len(files)
        self.subimgPxSize = 1000/self.pxSize

        # Get data shape and derivates
        im = Image.open(self.files[0])
        inputData = np.array(im).astype(np.float64)
        initShape = inputData.shape
        self.bound = (np.array(inputData.shape) - self.crop).astype(np.int)
        inputData = inputData[self.crop:self.bound[0], self.crop:self.bound[1]]
        dataShape = inputData.shape
        self.n = (np.array(dataShape)/self.subimgPxSize).astype(int)
        self.mag = np.array(dataShape)/self.n
        self.corrArray = np.zeros((self.nfiles, self.n[0], self.n[1]))
        self.path = os.path.split(self.files[0])[0]
        self.corrExp = np.empty((self.nfiles, initShape[0], initShape[1]),
                                dtype=np.single)
        self.corrExp[:] = np.nan
        self.ringsExp = np.empty((self.nfiles, initShape[0], initShape[1]),
                                 dtype=np.single)
        self.ringsExp[:] = np.nan

    def processResult(self, localCorr):
        self.corrArray[self.i] = localCorr

        localCorrBig = np.repeat(localCorr, self.mag[0], 0)
        localCorrBig = np.repeat(localCorrBig, self.mag[1], 1)

        self.corrExp[self.i, self.crop:self.bound[0],
                     self.crop:self.bound[1]] = localCorrBig

        # Save correlation values array
        corrName = utils.insertSuffix(self.files[self.i], '_correlation')
        tiff.imsave(corrName, self.corrExp[self.i], software='Gollum',
                    imagej=True, metadata={'spacing': 1, 'unit': 'um'},
                    resolution=(1000/self.pxSize, 1000/self.pxSize))

    def updateBar(self):
        self.startSignal.emit(os.path.split(self.files[self.i])[1])

    def start(self):

        t0 = time.time()
        for i in np.arange(self.nfiles):

            self.i = i

            # Image loading
            im = Image.open(self.files[i])
            inputData = np.array(im).astype(np.float64)
            inputData = inputData[self.crop:self.bound[0],
                                  self.crop:self.bound[1]]

            inputDataS = ndi.gaussian_filter(inputData, self.gaussSigma)
            meanS = np.mean(inputDataS)
            stdS = np.std(inputDataS)

            # binarization of image
            mask = inputDataS < meanS + self.intThres*stdS
            # shape the data into the subimg that we need for the analysis
            nblocks = np.array(inputData.shape)/self.n
            blocksInput = tools.blockshaped(inputData, *nblocks)
            blocksInputS = tools.blockshaped(inputDataS, *nblocks)
            blocksMask = tools.blockshaped(mask, *nblocks)

            worker = Worker(self.n, blocksInput, blocksInputS, blocksMask,
                            self.intThres, self.cArgs)
            worker.signals.start.connect(self.updateBar)
            worker.signals.done.connect(self.processResult)

            self.pool.start(worker)

        self.pool.waitForDone()


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = Gollum()
    win.show()
    app.exec_()
