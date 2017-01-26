# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:36:37 2016

@author: Luciano Masullo / Federico Barabas
"""

        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.pointsButton = QtGui.QPushButton('Points')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')

def FFTMethodGUI(self):
    print('FFT 2D analysis executed')

    thres = np.float(self.main.fftThrEdit.text())
    fft2output, coord, rlim, rings = tools.FFTMethod(self.selected, thres)
    rmin, rmax = rlim

    if rings:
        print('¡HAY ANILLOS!')
    else:
        print('NO HAY ANILLOS')

    # plot results
    self.fft2.clear()       # remove previous fft2
    self.fft2.addItem(self.FFT2img)
    self.fft2.setAspectLocked(True)
    self.FFT2img.setImage(fft2output)

    A = self.subImgSize    # size of the subimqge of interest

    # draw circles for visulization
    rminX = A*((rmax/100)*np.cos(np.linspace(0, 2*np.pi, 1000)) + 0.5)
    rminY = A*((rmax/100)*np.sin(np.linspace(0, 2*np.pi, 1000)) + 0.5)
    rmaxX = A*((rmin/100)*np.cos(np.linspace(0, 2*np.pi, 1000)) + 0.5)
    rmaxY = A*((rmin/100)*np.sin(np.linspace(0, 2*np.pi, 1000)) + 0.5)
    self.fft2.plot(rminX, rminY, pen=(0, 102, 204))
    self.fft2.plot(rmaxX, rmaxY, pen=(0, 102, 204))

    # plot local maxima
    self.fft2.plot(coord[:, 0], coord[:, 1], pen=None,
                   symbolBrush=(0, 102, 204), symbolPen='w')

    def pointsMethodGUI(self):

        print('Points analysis executed')

        # clear previous plot
        self.pointsPlot.clear()

        # set points analysis thereshold
        thres = np.float(self.main.pointsThrEdit.text())
        points, D, rings = tools.pointsMethod(self.selected, thres)

        if rings:
            print('¡HAY ANILLOS!')
            pen = pg.mkPen(color=(0, 255, 100), width=1,
                           style=QtCore.Qt.SolidLine, antialias=True)
            for d in D:
                self.pointsPlot.plot([d[0][0], d[1][0], d[2][0]],
                                     [d[0][1], d[1][1], d[2][1]], pen=pen,
                                     symbolBrush=(0, 204, 122), symbolPen='w')
        else:
            print('NO HAY ANILLOS')

        # plot results
        self.pointsPlot.addItem(self.pointsImg)
        self.pointsImg.setImage(self.selected)
        self.pointsPlot.plot(points[:, 0], points[:, 1], pen=None,
                             symbolBrush=(0, 204, 122), symbolPen='w')
