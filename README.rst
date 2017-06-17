RingFinder
==========

Automated image analysis for finding the fraction of structure presenting a given pattern. Originally developed to identify the 190 nm periodicity in the actin/spectrin cytoskeleton of neurons.

Installation
~~~~~~~~~~~~

Ubuntu
^^^^^^

Run in terminal:

::

    
    $ sudo apt-get install python3-pip git
    $ sudo pip3 install tifffile pyqtgraph
    $ git clone https://github.com/fedebarabas/ringfinder
    
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

The code includes two executable programs: ringFinder and ringFinderDeveloper. Their usage instructions are descibed below. 

ringFinderDeveloper 
^^^^^^^^^^^^^^^^^^^

ringFinderDeveloper is used for fine tuning of the analysis parameters so that the image structure is correctly characterized. These parameters are chosen from its performance on individual subregions of a testing image.

1. Open the program:

   ::

       $ python -m bin.ringFinderDeveloper

   .. image:: screenshots/developer1.png

   It is structured in two sections: input on the left and output on the right. 

2. Load one of the images. At the top of the input section, the **Load image** subsection allows to load either STORM or STED images provided its **Pixel size** and, for STORM, the final image **magnification** over the raw images. If your image was acquired through a different method, load it as a STED image with the appropiate pixel size. These instructions use the example STED image automatically loaded when the program starts.

3. Choose a region of interest size (**ROI size**) appropiate to the structure that you want to detect within your images. *Gollum* was developed with the structure observed in the spectrin distribution in neurons of 190 nm periodicity, and therefore a 1000 nm ROI size was chosen.

4. Neuron discrimination. The discrimination of the biological structure is carried out by first blurring the image with a gaussian kernel of a sigma given by the value in the **Sigma of gaussian filter** field. Then, all pixels with an intensity above **#sigmas threshold from mean** sigmas from the mean intensity are considered part of the structure of interest. Therefore, **Sigma of gaussian filter** is a function of the size of the structure and **#sigmas threshold from mean**, of the images's SNR. A sigma of 100~nm and a discrimination intensity of half a standard deviation above the mean intensity works well for STORM and STED images of hippocampal neurons. Check if the chosen values work correctly by pushing the **Intensity and neuron content discrimination** button.

5. Using the arrow keys, move the yellow subimage ROI to a region where the structure to be identified is clearly present. Choose suitable values for the **Ring periodicity** and (within the **Advanced** subsection), the **Direction lines min length**. The direction lines are detected at the edge between the structure and the background and are used to estimate the direction angle of the structure. For the actin/spectrin periodical structure in neurons, a periodicity of 180-190 and a minimum length of 300 works well. 

6. Push **Run analysis** button. The software will compare the subregion to a reference pattern for different angles and phases and it will show the output in the two right panels of the output section. 

   .. image:: screenshots/developer2.png

   The graph at the top plots the *Pearson coefficient* vs the angle of the reference pattern. The blue-shaded region points the angle range within where the maximum Pearson coefficient will be found. It is centered in the previously estimated direction angle. Below the plot, the pattern with the angle and phase that maximize the Pearson coefficient is shown on top of the data. They should match.

7. Take note of the maximum Pearson value in this region and compare it with the one found in regions with biological structure but without the periodical arrangement that you need to identify. Then, choose a **Discrimination threshold** that allows you to discriminate these two distinct cases. This threshold will be further tuned within the ringFinder program.

Once structures in your sample image are correctly discriminated, close ringFinderDeveloper. The chosen parameters are automatically saved to a config file.

ringFinder
^^^^^^^^^^

1. Open the program:

   ::

       $ python -m bin.ringFinder
       
   .. image:: screenshots/finder1.png
   
2. Load one of the images and press **Run Analysis**. 

3. Choose a threshold value by moving the **Discrimination threshold** slider until the software successfully discriminates subregions exhibiting the given structure from those that do not. For the actin/spectrin structure, we used 0.2 for STORM images and 0.17 for STED images. The highlighted regions indicate a Pearson value above the threshold. 

   .. image:: screenshots/finder2.png

4. Use the chosen value to automatically analyze an unlimited number of images taken from the sample under identical conditions. Do this from the **Run** section of the program bar. The analysis of a folder of 15 images takes ~ 120 s running on a computer with an Intel i5-4440 CPU @ 3.10GHz processor. The output of the program is located in a dedicated subfolder named **results** and includes an histogram of all Pearson values of the subregions of all analyzed images. 

   .. image:: screenshots/histogram.png

   The information in the textbox characterizes the set of analyzed images. In particular, **ringFrac** is the fraction of subimages exhibiting the specified structure. Also, for every single analyzed image, a binary one indicating the presence of the structure and an image in which the Pearson coefficient is encoded in each pixel's intensity are provided. They can be superimposed with the original data using ImageJ software.

Contact
~~~~~~~

Feel free to contact us with comments or suggestions. Use any part of
the code that suits your needs.

Federico Barabas
   fede.barabas[AT]gmail.com

Luciano Masullo
   lu.masullo[AT]gmail.com