# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:51:26 2017

@author: Federico Barabas
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

class Grid:

    def __init__(self, viewbox, shape, n=[10, 10]):
        self.vb = viewbox
        self.n = n
        self.lines = []

        pen = pg.mkPen(color=(255, 255, 255), width=1, style=QtCore.Qt.DotLine,
                       antialias=True)
        self.rect = QtGui.QGraphicsRectItem(0, 0, shape[0], shape[1])
        self.rect.setPen(pen)
        self.vb.addItem(self.rect)
        self.lines.append(self.rect)

        step = np.array(shape)/self.n

        for i in np.arange(0, self.n[0] - 1):
            cx = step[0]*(i + 1)
            line = QtGui.QGraphicsLineItem(cx, 0, cx, shape[1])
            line.setPen(pen)
            self.vb.addItem(line)
            self.lines.append(line)

        for i in np.arange(0, self.n[1] - 1):
            cy = step[1]*(i + 1)
            line = QtGui.QGraphicsLineItem(0, cy, shape[0], cy)
            line.setPen(pen)
            self.vb.addItem(line)
            self.lines.append(line)


class SubImgROI(pg.ROI):

    def __init__(self, step, *args, **kwargs):
        super().__init__([0, 0], [0, 0], translateSnap=True, scaleSnap=True,
                         *args, **kwargs)
        self.step = step
        self.keyPos = (0, 0)
        self.addScaleHandle([1, 1], [0, 0], lockAspect=True)

    def moveUp(self):
        self.moveRoi(0, self.step)

    def moveDown(self):
        self.moveRoi(0, -self.step)

    def moveRight(self):
        self.moveRoi(self.step, 0)

    def moveLeft(self):
        self.moveRoi(-self.step, 0)

    def moveRoi(self, dx, dy):
        self.keyPos = (self.keyPos[0] + dx, self.keyPos[1] + dy)
        self.setPos(self.keyPos)
