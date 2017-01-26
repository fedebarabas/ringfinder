# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:45:14 2016

@author: Federico Barabas
"""

from pyqtgraph.Qt import QtGui
from ringfinder.ringFinder import Gollum


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = Gollum()
    win.show()
    app.exec_()
