# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:45:14 2016

@author: Federico Barabas
"""

from pyqtgraph.Qt import QtGui
from ringfinder.ringFinderDeveloper import GollumDeveloper


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = GollumDeveloper()
    win.show()
    app.exec_()
