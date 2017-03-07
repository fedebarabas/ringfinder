RingFinder
==========

Automated image analysis for finding the fraction of structure presenting a given pattern. Originally developed to identify the 190 nm periodicity in the actin/spectrin cytoskeleton of neurons.

Installation
~~~~~~~~~~~~

Ubuntu
^^^^^^

Run in terminal:

::

    ```
    $ sudo apt-get install python3-pip git
    $ sudo pip3 install tifffile pyqtgraph
    $ git clone https://github.com/fedebarabas/ringfinder
    ```

Windows
^^^^^^^

-  Install `WinPython
   3.4 <https://sourceforge.net/projects/winpython/files/>`__.
-  Browse to `Laboratory for Fluorescence
   Dynamics <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__ and download
   tifffile for Python 3.4 to
   ``$PATH\WinPython-64bit-3.4.4.1\python-3.4.4.amd64\``.
-  Open WinPython Command Prompt and run:

   ::

       $ pip install tifffile-2016.4.19-cp34-cp34m-win_amd64.whl

-  Clone `Tormenta repo <https://github.com/fedebarabas/ringfinder>`__.


Usage
~~~~~

The code includes two executable programs: ringFinder and ringFinderDeveloper. ringFinderDeveloper is used for testing the analysis on individual subregions of an image in order to fine tune the parameters of the program. Once these parameters are chosen so that the software correctly characterizes the image structure, ringFinder is then used for automatic bulk processing of as many images as necessary to gather significant statistics. 


-  Run ringFinderDeveloper to tune the detection parameters:

   ::

       $ python -m bin.ringFinderDeveloper

   Load one of the images. Tune these parameters:

   -  Gaussian filter sigma and intensity threshold: they are needed for discriminating image from background. A sigma of 100~nm and a discrimination intensity of half a standard deviation above the mean intensity works well for STORM and STED images of hippocampal neurons. 
   -  Size of subregions: used to locally characterize the similarity with the user-defined pattern. 1000 nm was used for identifying the 190 nm periodicity in the actin/spectrin cytoskeleton of neurons.
   -  Minimum area: subregions with less than a certain percentage of its area occupied by the imaged structure are discarded from further analysis to avoid undersampling. 20% usually works well.

   Once structures in your sample image are correctly identified, close ringFinderDeveloper. The configuration is automatically saved to a config file.
   
-  Run ringFinder:

   ::

       $ python -m bin.ringFinder
       
   Load one of the images and press "Run Analysis". Choose a threshold value by moving the slider until the software successfully discriminates subregions exhibiting the given structure from those that do not. This chosen value can be used in the subsequent bulk analysis of a large number of images taken from the sample under identical conditions (Run -> Analyze batch...).

Contact
~~~~~~~

Feel free to contact me with comments or suggestions. Use any part of
the code that suits your needs.

Federico Barabas fede.barabas[AT]gmail.com
